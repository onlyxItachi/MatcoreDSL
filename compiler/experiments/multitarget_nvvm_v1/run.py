#!/usr/bin/env python3
"""Reproduce the bounded NVVM research lane using already-installed tools.

No dependency download or LLVM source build. Pass --execute for real GPU tests.
Artifacts are not accepted as product execution authority.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issuer", type=Path, required=True)
    parser.add_argument("--mlir-prefix", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    if args.sanitize and not args.execute:
        parser.error("--sanitize requires explicit --execute")
    source = Path(__file__).resolve().parent
    work = args.work_dir.resolve()
    if work == Path("/") or work == Path.home() or work == Path("/tmp") or Path("/tmp") in work.parents:
        parser.error("select a dedicated SSD-backed work directory, never /tmp or home root")
    work.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(work)], text=True).strip()
    if filesystem in ("tmpfs", "ramfs"):
        parser.error("research builds must not use a RAM-backed filesystem")
    if shutil.disk_usage(work).free < 1024**3:
        parser.error("require at least 1 GiB free before the small local build")
    temporary = work / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = dict(os.environ, TMPDIR=str(temporary), CUDA_CACHE_PATH=str(work / "cuda-cache"),
                       PYTHONDONTWRITEBYTECODE="1")
    commands = []

    def run(command, output=None):
        command = [str(item) for item in command]
        print("+", " ".join(command), flush=True)
        result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=120)
        commands.append({"command": command, "exit": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        (work / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        if output:
            Path(output).write_text(result.stdout)
        return result.stdout

    prefix = args.mlir_prefix.resolve()
    opt, translate = prefix / "bin/mlir-opt", prefix / "bin/mlir-translate"
    for tool in (opt, translate, "/usr/bin/llc-21", "/usr/bin/clang++-21"):
        version = run([tool, "--version"])
        if "21.1.8" not in version:
            raise RuntimeError(f"requires coherent 21.1.8 tools: {tool}")
    run(["/usr/local/cuda/bin/ptxas", "--version"])
    run([args.issuer.resolve(), "--output", work / "strict.ll"])
    structured = work / "strict.ll.structured.mlir"
    run([opt, structured,
         "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
         "-o", work / "bufferized.mlir"])
    run([opt, work / "bufferized.mlir", "--convert-linalg-to-parallel-loops",
         "--gpu-map-parallel-loops", "--convert-parallel-loops-to-gpu", "--canonicalize",
         "--gpu-launch-sink-index-computations", "--gpu-kernel-outlining", "--canonicalize",
         "-o", work / "outlined.mlir"])
    run([opt, work / "outlined.mlir",
         "--pass-pipeline=builtin.module(gpu.module(lower-affine,convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64},canonicalize,reconcile-unrealized-casts))",
         "-o", work / "nvvm.mlir"])
    run(["/usr/bin/clang++-21", "-std=c++17", "-O0", "-I" + str(prefix / "include"),
         "-I/usr/lib/llvm-21/include", source / "extract_device.cpp", "-L" + str(prefix / "lib"),
         "-Wl,-rpath," + str(prefix / "lib"), "-lMLIR", "-lLLVM-21", "-o", work / "extract-device"])
    run([work / "extract-device", work / "nvvm.mlir", work])
    for stage, suffix in (("fill", ""), ("matmul", "_0")):
        device = work / f"__matcore_strict_gemm_f32_v1_kernel{suffix}.device.mlir"
        run([translate, "--mlir-to-llvmir", device, "-o", work / (stage + ".ll")])
        run(["/usr/bin/llc-21", "-mtriple=nvptx64-nvidia-cuda", "-mcpu=sm_89", "-mattr=+ptx80",
             "-fp-contract=off", work / (stage + ".ll"), "-o", work / (stage + ".ptx")])
        run(["/usr/local/cuda/bin/ptxas", "--gpu-name", "sm_89", "--fmad=false",
             work / (stage + ".ptx"), "-o", work / (stage + ".cubin")])
        run(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", work / (stage + ".cubin")],
            output=work / (stage + ".sass"))
    ptx = (work / "matmul.ptx").read_text()
    sass = (work / "matmul.sass").read_text()
    if not all(instruction in ptx for instruction in ("mul.rn.f32", "add.rn.f32")):
        raise RuntimeError("lost separate nearest-even f32 arithmetic in PTX")
    if re.search(r"\b(?:fma|mad)\.[^\n]*f32|\.ftz\b", ptx) or "FFMA" in sass or ".FTZ" in sass:
        raise RuntimeError("unexpected contraction or flush-to-zero")
    if args.execute:
        result = run([sys.executable, "-B", source / "execute.py", work], output=work / "execution.txt")
        print(result, end="")
        if args.sanitize:
            # The suite deliberately tests a rejected launch; all Driver API
            # results are still checked by the harness itself.
            result = run(["/usr/local/cuda/bin/compute-sanitizer", "--tool", "memcheck",
                          "--error-exitcode", "99", "--leak-check", "full",
                          "--report-api-errors", "no", sys.executable, "-B",
                          source / "execute.py", work], output=work / "sanitizer.txt")
            print(result, end="")
    manifest = {
        "evidence": "research physical sm89 execution" if args.execute else "research compile-only",
        "product_execution_authority": False,
        "source": "closed canonical strict GEMM issuer; not an admitted user program",
        "gpu_schedule": "one block per output; one thread per block; serial increasing K",
        "compiler": "21.1.8", "chip": "sm_89", "ptx": "+ptx80",
        "sanitizer": "CUDA memcheck" if args.sanitize else "not run",
        "numerical": "positive-zero fill; separate f32 mul/add; no FMA, FTZ, reassociation",
        "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(work.iterdir()) if p.is_file() and p.name != "manifest.json"},
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Research evidence:", work / "manifest.json")


if __name__ == "__main__":
    main()
