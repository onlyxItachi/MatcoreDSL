#!/usr/bin/env python3
"""Actual research images: target/ABI/import/arithmetic gates, not authority.

Inspection strategy follows generated_gpu/object_contract.py, but this image is
one serial pair kernel, not the production single-GEMM fill/GEMM image pair.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

KERNEL = "__matcore_research_strict_fused_pair_kernel"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def header(data, target):
    require(len(data) >= 64 and data[:7] == b"\x7fELF\x02\x01\x01", "not ELF64 little-endian")
    kind, machine = struct.unpack_from("<HH", data, 16)
    flags, = struct.unpack_from("<I", data, 48)
    require((kind, machine, (flags >> 8) & 255) == (2, 190, 89) if target == "nvvm" else
            (kind, machine, flags & 255) == (3, 224, 0x43), "wrong GPU target image")


def assembly(text, target):
    require(KERNEL in text, "missing actual pair kernel")
    if target == "nvvm":
        require(re.search(r"\bsm_89\b", text), "wrong SASS architecture")
        require(not re.search(r"\b(?:FFMA|HFMA2|HMMA|IMMA|MMA)\b|\.FTZ\b", text),
                "fused/matrix/FTZ arithmetic")
        require(re.search(r"\bFMUL\b", text) and re.search(r"\bFADD\b", text),
                "separate scalar f32 arithmetic absent")
    else:
        require("file format elf64-amdgpu" in text, "wrong AMDGPU architecture")
        require(not re.search(r"\bv_(?:fma|mad|mfma|wmma)[a-z0-9_]*\b", text),
                "fused/matrix arithmetic")
        require(re.search(r"\bv_mul_f32(?:_e(?:32|64))?\b", text) and
                re.search(r"\bv_add_f32(?:_e(?:32|64))?\b", text),
                "separate scalar f32 arithmetic absent")


def amd_descriptor(data):
    """21.1.8 AMDHSAKernelDescriptor.h offsets, not inferred from attributes."""
    shoff, = struct.unpack_from("<Q", data, 40)
    shentsize, shnum = struct.unpack_from("<HH", data, 58)
    require(shentsize == 64 and shoff + shentsize * shnum <= len(data), "bad section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
                for i in range(shnum)]
    matches = []
    for section in sections:
        if section[1] != 11:  # SHT_DYNSYM; no duplicate .symtab lookup.
            continue
        strings = sections[section[6]]
        names = data[strings[4]:strings[4]+strings[5]]
        require(section[9] == 24, "bad ELF64 dynamic symbol size")
        for offset in range(section[4], section[4]+section[5], 24):
            name, info, other, index, value, size = struct.unpack_from("<IBBHQQ", data, offset)
            text = names[name:].split(b"\0", 1)[0].decode()
            if text != KERNEL + ".kd":
                continue
            require(size == 64 and index < len(sections), "bad AMD kernel descriptor symbol")
            owner = sections[index]
            file_offset = owner[4] + value - owner[3]
            require(file_offset >= owner[4] and file_offset + 64 <= owner[4]+owner[5],
                    "kernel descriptor outside section")
            matches.append(file_offset)
    require(len(matches) == 1, "expected unique AMD kernel descriptor")
    offset = matches[0]
    group, private, kernarg = struct.unpack_from("<III", data, offset)
    rsrc1, = struct.unpack_from("<I", data, offset+48)
    require(group == private == 0, "hidden group/private tensor storage")
    require(((rsrc1 >> 12) & 3) == 0 and ((rsrc1 >> 16) & 3) == 3 and
            ((rsrc1 >> 23) & 1) == 1, "AMDHSA lacks nearest/gradual/IEEE f32 controls")
    return {"file_offset": offset, "compute_pgm_rsrc1": hex(rsrc1),
            "f32_round": "nearest-even", "f32_denorm": "flush-none", "ieee_mode": True,
            "group_segment_bytes": group, "private_segment_bytes": private,
            "kernarg_bytes_including_hidden": kernarg}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    args = parser.parse_args()
    records, negatives = [], []

    def command(argv):
        result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=30)
        require(result.returncode == 0, f"inspection failed: {argv}: {result.stderr}")
        return result.stdout

    def rejected(name, run):
        try:
            run()
        except ValueError:
            negatives.append(name)
            return
        raise ValueError(f"negative control accepted: {name}")

    for target, extension in (("nvvm", "cubin"), ("rocdl", "hsaco")):
        path = args.build / f"{target}.{extension}"
        data = path.read_bytes()
        header(data, target)
        imports = command(["/usr/lib/llvm-21/bin/llvm-nm", "--undefined-only", path])
        require(not imports.strip(), f"unexpected device imports: {imports}")
        symbols = command(["/usr/lib/llvm-21/bin/llvm-nm", "--defined-only", path])
        require(re.search(rf" T {KERNEL}$", symbols, re.M), "real kernel symbol absent")
        require(re.findall(r" T (\S+)$", symbols, re.M) == [KERNEL],
                "extra executable function export")
        text = command(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", path] if target == "nvvm"
                       else ["/usr/lib/llvm-21/bin/llvm-objdump", "-d", path])
        assembly(text, target)
        (args.build / f"{target}.assembly.txt").write_text(text)
        forged = bytearray(data)
        struct.pack_into("<H", forged, 18, 62)
        rejected(f"{target}:cpu-image", lambda: header(forged, target))
        forbidden = "FFMA R0, R1, R2, R3;" if target == "nvvm" else "v_fma_f32 v0,v1,v2,v3"
        rejected(f"{target}:FMA", lambda: assembly(text + "\n" + forbidden, target))
        if target == "nvvm":
            rejected("nvvm:FTZ", lambda: assembly(text + "\nFMUL.FTZ R0,R1,R2;", target))
        missing = re.sub(r"\bFMUL\b|\bv_mul_f32(?:_e(?:32|64))?\b", "REMOVED", text)
        rejected(f"{target}:missing-multiply", lambda: assembly(missing, target))
        llvm = (args.build / f"{target}.ll").read_text()
        require(len(re.findall(r" = fmul float ", llvm)) == 2 and
                len(re.findall(r" = fadd float ", llvm)) == 2, "not two strict LLVM sites")
        require(not re.search(r"\b(?:fast|contract|reassoc|noalias)\b", llvm), "LLVM permission appeared")
        descriptor = None
        if target == "rocdl":
            descriptor = amd_descriptor(data)
            for shift, label, value in ((12, "wrong-rounding", 1), (16, "FTZ", 0),
                                        (23, "not-IEEE", 0)):
                corrupted = bytearray(data)
                offset = descriptor["file_offset"] + 48
                rsrc1, = struct.unpack_from("<I", corrupted, offset)
                mask = (1 if shift == 23 else 3) << shift
                struct.pack_into("<I", corrupted, offset, (rsrc1 & ~mask) | (value << shift))
                rejected(f"rocdl:{label}-descriptor", lambda: amd_descriptor(corrupted))
        records.append({"target": target, "path": str(path), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(), "imports": imports,
                        "llvm_sha256": hashlib.sha256(llvm.encode()).hexdigest(),
                        "kernel": KERNEL, "expanded_memref_fields": 35,
                        "amdhsa_descriptor": descriptor})
    report = {"status": "PASS", "artifacts": records, "negative_controls": negatives,
              "authority": "none; research artifact inspection only"}
    (args.build / "object-contract.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS actual NVVM/ROCDL object gates; {len(negatives)} rejected inspection controls")


if __name__ == "__main__":
    main()
