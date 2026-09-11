#!/usr/bin/env python3
"""Sequential, no-timing leaf experiment using the real closed MLIR issuer.

This is not a source compiler, an IR execution-authority importer, or a runtime
registry. AArch64 override is explicitly compile-only on an x86 host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(assembly, symbols, imports, arm=False):
    expected = {"__matcore_strict_gemm_f32_v1",
                "_mlir_ciface___matcore_strict_gemm_f32_v1"}
    definitions = {line.split()[-1] for line in symbols.splitlines()
                   if re.search(r"\sT\s", line)}
    if definitions != expected or re.search(r"\s[tWw]\s", symbols):
        raise RuntimeError("issued strong function set changed")
    imported = {line.split()[-1] for line in imports.splitlines() if line.strip()}
    if imported - {"memset"}:
        raise RuntimeError(f"unbounded imports: {sorted(imported)}")
    if arm:
        if "elf64-littleaarch64" not in assembly:
            raise RuntimeError("not an AArch64 ELF object")
        if re.search(r"\b(fmla|fmls|fmadd|fmsub|fnmadd|fnmsub)\b", assembly):
            raise RuntimeError("strict object acquired fused arithmetic")
        if not all(re.search(rf"\b{op}\b", assembly) for op in ("fmul", "fadd")):
            raise RuntimeError("missing separate ARM multiply/add")
    else:
        if "elf64-x86-64" not in assembly:
            raise RuntimeError("not an x86-64 ELF object")
        if re.search(r"\bv?f(madd|msub|nmadd|nmsub)\w*", assembly):
            raise RuntimeError("strict object acquired fused arithmetic")
        if not all(re.search(rf"\bv?{op}(ss|ps)\b", assembly)
                   for op in ("mul", "add")):
            raise RuntimeError("missing separate x86 multiply/add")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issuer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tmp-dir", required=True, type=Path)
    parser.add_argument("--clang", default="/usr/bin/clang++-21")
    parser.add_argument("--objdump", default="/usr/bin/llvm-objdump-21")
    parser.add_argument("--nm", default="/usr/bin/llvm-nm-21")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("this initial campaign runner requires native Linux x86-64")
    output, temporary = args.output.resolve(), args.tmp_dir.resolve()
    for path in (output, temporary):
        if path == Path("/tmp") or Path("/tmp") in path.parents:
            parser.error("outputs and temporary files must not use /tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    commands = []

    def run(command, allowed=(0,)):
        process = subprocess.run([str(x) for x in command], env=env,
                                 text=True, capture_output=True, timeout=120)
        commands.append({"argv": [str(x) for x in command],
                         "returncode": process.returncode,
                         "stdout": process.stdout, "stderr": process.stderr})
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if process.returncode not in allowed:
            raise RuntimeError(f"command failed: {command}\n{process.stderr}")
        return process

    for tool in (args.clang, args.objdump, args.nm):
        if not re.search(r"\b21\.1\.8\b", run([tool, "--version"]).stdout):
            raise RuntimeError("all compiler/object tools must be exact 21.1.8")
    issuer = args.issuer.resolve(strict=True)
    issuer_hash = sha(issuer)
    ir = output / "issued-row-contiguous.ll"
    run([issuer, "--output", ir, "--schedule=row-contiguous"])
    manifest = Path(str(ir) + ".manifest").read_text()
    required = ["toolchain=21.1.8", "profile=strict_f32",
                "source_authority=none_builtin_primitive_only",
                "schedule=row-contiguous-mkn", "address_sanitizer=off",
                "target=x86_64-pc-linux-gnu", "llvm_sha256=" + sha(ir)]
    if not all(line in manifest.splitlines() for line in required):
        raise RuntimeError("issuer artifact does not match this experiment's exact profile")
    artifact_hash = sha(ir)
    fixture = Path(__file__).resolve().with_name("leaf_fixture.cpp")
    fixture_object = output / "fixture.o"
    run([args.clang, "-std=c++20", "-O2", "-fno-fast-math", "-ffp-contract=off",
         "-march=x86-64", "-c", fixture, "-o", fixture_object])
    variants = {
        "baseline": ["-march=x86-64", "-mno-avx", "-mno-fma"],
        "avx": ["-march=x86-64", "-mavx", "-mno-avx2", "-mno-fma"],
        "avx2": ["-march=x86-64", "-mavx2", "-mno-fma"],
        "avx2-fma": ["-march=x86-64", "-mavx2", "-mfma"],
        "avx512f": ["-march=x86-64", "-mavx512f", "-mno-fma"],
        "aarch64": ["--target=aarch64-unknown-linux-gnu", "-march=armv8-a"],
    }
    results = []
    for name, flags in variants.items():
        obj = output / f"{name}.o"
        run([args.clang, "-c", "-x", "ir", ir, "-O2", "-ffp-contract=off",
             *flags, "-o", obj])
        assembly = run([args.objdump, "-d", obj]).stdout
        (output / f"{name}.asm").write_text(assembly)
        inspect(assembly, run([args.nm, "--defined-only", obj]).stdout,
                run([args.nm, "--undefined-only", obj]).stdout, name == "aarch64")
        result = {"target_request": name, "object_sha256": sha(obj),
                  "requested_flags": flags, "strict_no_fma_object_check": "PASS",
                  "observed_ymm": bool(re.search(r"\bymm\d+\b", assembly)),
                  "observed_zmm": bool(re.search(r"\bzmm\d+\b", assembly))}
        if name == "aarch64":
            result["execution"] = "NOT_RUN_cross_compile_only"
            result["target_override"] = "experimental_not_production_issuer_authority"
        else:
            executable = output / f"{name}-fixture"
            run([args.clang, fixture_object, obj, "-o", executable])
            process = run([executable, name], allowed=(0, 77))
            result["execution"] = "PASS" if process.returncode == 0 else "SKIP_capability_or_fp"
            result["stdout"] = process.stdout
        results.append(result)
    if sha(issuer) != issuer_hash or sha(ir) != artifact_hash:
        raise RuntimeError("issuer or issued artifact changed during experiment")
    report = {"schema": "multitarget-cpu-leaf-experiment-v1",
              "source_execution_authority": "none",
              "issuer": str(issuer), "issuer_sha256": issuer_hash,
              "llvm_sha256": artifact_hash, "results": results}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
