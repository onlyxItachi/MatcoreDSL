#!/usr/bin/env python3
"""Require exact ZMM packed-FMA code in the isolated AVX-512 microkernel."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


SYMBOL = "matcore_cpu_packed_avx512_4x16_microkernel_f32_v1"


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--nm", default="nm")
    arguments = parser.parse_args()

    symbols = run([arguments.nm, "-C", arguments.artifact])
    if not re.search(rf"\b{re.escape(SYMBOL)}$", symbols, re.MULTILINE):
        print(f"missing exact AVX-512 microkernel symbol: {SYMBOL}", file=sys.stderr)
        return 1

    disassembly = run(
        [arguments.objdump, "-d", "--no-show-raw-insn", f"--disassemble={SYMBOL}",
         arguments.artifact]
    ).lower()
    zmm_count = len(re.findall(r"\bzmm[0-9]+\b", disassembly))
    packed_fma_count = len(re.findall(r"\bvfmadd(?:132|213|231)ps\b", disassembly))
    if zmm_count < 8 or packed_fma_count < 4:
        print(
            "AVX-512 microkernel was scalarized or does not contain the expected "
            f"packed body (zmm={zmm_count}, packed_fma={packed_fma_count})",
            file=sys.stderr,
        )
        print(disassembly, file=sys.stderr)
        return 1

    print(
        f"AVX-512 artifact verified: symbol={SYMBOL} zmm={zmm_count} "
        f"packed_fma={packed_fma_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
