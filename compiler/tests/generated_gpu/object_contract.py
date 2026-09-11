#!/usr/bin/env python3
"""Inspect the actual embedded device images; not an execution/correctness proof.

The compiler-issued structured witness and physical arithmetic tests are separate
obligations. These checks catch wrong-target artifacts, unexpected imports and
machine arithmetic that contradicts this initial strict recipe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_header(data, target):
    require(len(data) >= 64 and data[:7] == b"\x7fELF\x02\x01\x01", "not ELF64 little-endian")
    kind, machine = struct.unpack_from("<HH", data, 16)
    flags, = struct.unpack_from("<I", data, 48)
    if target == "nvvm":
        require(kind == 2 and machine == 190 and (flags >> 8) & 255 == 89,
                "not an sm_89 CUDA executable image")
    else:
        require(kind == 3 and machine == 224 and flags & 255 == 0x43,
                "not a gfx1150 AMDGPU shared image")


def check_assembly(assembly, target, stage):
    require("__matcore_strict_gemm_f32_v1_kernel" in assembly, "missing issued kernel")
    if target == "nvvm":
        require(re.search(r"\bsm_89\b", assembly), "wrong disassembled CUDA target")
        require(not re.search(r"\b(?:FFMA|HFMA2|HMMA|IMMA|MMA)\b|\.FTZ\b", assembly),
                "forbidden fused/matrix or flush-to-zero arithmetic")
        if stage == "gemm":
            require(re.search(r"\bFMUL\b", assembly) and re.search(r"\bFADD\b", assembly),
                    "missing separate f32 multiply/add")
    else:
        require("file format elf64-amdgpu" in assembly, "wrong disassembled AMDGPU format")
        require(not re.search(r"\bv_(?:fma|mad|mfma|wmma)[a-z0-9_]*\b", assembly),
                "forbidden fused/matrix arithmetic")
        if stage == "gemm":
            require(re.search(r"\bv_mul_f32(?:_e(?:32|64))?\b", assembly) and
                    re.search(r"\bv_add_f32(?:_e(?:32|64))?\b", assembly),
                    "missing separate f32 multiply/add")


def rejects(label, function):
    try:
        function()
    except ValueError:
        return label
    raise ValueError(f"negative control was accepted: {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--target", choices=("nvvm", "rocdl"), required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    commands, artifacts, negatives = [], [], []

    def run(command):
        command = list(map(str, command))
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        commands.append({"argv": command, "exit": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        require(result.returncode == 0, f"artifact inspection failed: {commands[-1]}")
        return result.stdout

    manifest = dict(line.split("=", 1) for line in
                    Path(str(args.prefix) + ".images.cpp.manifest").read_text().splitlines())
    require(manifest.get("target") == args.target, "embedding manifest target mismatch")
    for stage in ("fill", "gemm"):
        extension = "cubin" if args.target == "nvvm" else "hsaco"
        path = Path(f"{args.prefix}.{stage}.{extension}")
        data = path.read_bytes()
        check_header(data, args.target)
        digest = hashlib.sha256(data).hexdigest()
        require(manifest.get(stage + "_sha256") == digest and
                manifest.get(stage + "_bytes") == str(len(data)), "embedding identity mismatch")
        imports = run([args.llvm_bin / "llvm-nm", "--undefined-only", path])
        require(not imports.strip(), f"device image acquired imports: {imports}")
        symbols = run([args.llvm_bin / "llvm-nm", "--defined-only", path])
        require(re.search(r" T __matcore_strict_gemm_f32_v1_kernel$", symbols, re.M),
                "missing real kernel symbol")
        if args.target == "nvvm":
            require(args.cuobjdump is not None, "CUDA image requires cuobjdump")
            assembly = run([args.cuobjdump, "--dump-sass", path])
        else:
            assembly = run([args.llvm_bin / "llvm-objdump", "-d", path])
        check_assembly(assembly, args.target, stage)
        artifacts.append({"path": str(path), "sha256": digest, "bytes": len(data)})
        # These are validator falsifiers, not forged-image execution tests.
        forged = bytearray(data)
        struct.pack_into("<H", forged, 18, 62)  # CPU image in a GPU slot.
        negatives.append(rejects(stage + ":cpu-image", lambda: check_header(forged, args.target)))
        forged = bytearray(data)
        struct.pack_into("<I", forged, 48, 0)  # No qualified architecture.
        negatives.append(rejects(stage + ":wrong-architecture", lambda: check_header(forged, args.target)))
        forbidden = "FFMA R0, R1, R2, R3;" if args.target == "nvvm" else "v_fma_f32 v0, v1, v2, v3"
        negatives.append(rejects(stage + ":fused-arithmetic",
            lambda: check_assembly(assembly + "\n" + forbidden, args.target, stage)))
        if args.target == "nvvm":
            negatives.append(rejects(stage + ":flush-to-zero",
                lambda: check_assembly(assembly + "\nFMUL.FTZ R0, R1, R2;", args.target, stage)))
        if stage == "gemm":
            absent = re.sub(r"\bFMUL\b|\bv_mul_f32(?:_e(?:32|64))?\b", "REMOVED", assembly)
            negatives.append(rejects("gemm:missing-multiply",
                lambda: check_assembly(absent, args.target, stage)))
    args.report.write_text(json.dumps({"target": args.target, "artifacts": artifacts,
        "commands": commands, "negative_controls": negatives, "status": "PASS",
        "claim": "machine artifact and embedding identity; not execution or general FP proof"}, indent=2) + "\n")
    print(f"{args.target} object contract PASS: 2 images, {len(negatives)} negative controls")


if __name__ == "__main__":
    main()
