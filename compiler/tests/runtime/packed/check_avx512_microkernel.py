#!/usr/bin/env python3
"""Require exact ZMM packed-FMA code in the isolated AVX-512 microkernel."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


CHECKED_SYMBOL = "matcore_cpu_packed_avx512_4x16_microkernel_f32_v1"
FULL_TILE_SYMBOL = (
    "matcore_internal_cpu_packed_avx512_4x32_full_microkernel_f32_m7"
)


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
    for symbol in (CHECKED_SYMBOL, FULL_TILE_SYMBOL):
        if not re.search(rf"\b{re.escape(symbol)}$", symbols, re.MULTILINE):
            print(f"missing exact AVX-512 microkernel symbol: {symbol}", file=sys.stderr)
            return 1

    checked_disassembly = run(
        [
            arguments.objdump,
            "-d",
            "--no-show-raw-insn",
            f"--disassemble={CHECKED_SYMBOL}",
            arguments.artifact,
        ]
    ).lower()
    checked_zmm_count = len(re.findall(r"\bzmm[0-9]+\b", checked_disassembly))
    checked_fma_count = len(
        re.findall(r"\bvfmadd(?:132|213|231)ps\b", checked_disassembly)
    )
    if checked_zmm_count < 8 or checked_fma_count < 4:
        print(
            "checked AVX-512 edge microkernel was scalarized or does not contain "
            "the expected packed body "
            f"(zmm={checked_zmm_count}, packed_fma={checked_fma_count})",
            file=sys.stderr,
        )
        print(checked_disassembly, file=sys.stderr)
        return 1

    full_disassembly = run(
        [
            arguments.objdump,
            "-d",
            "--no-show-raw-insn",
            f"--disassemble={FULL_TILE_SYMBOL}",
            arguments.artifact,
        ]
    ).lower()
    full_zmm_registers = set(re.findall(r"\bzmm[0-9]+\b", full_disassembly))
    full_fma_count = len(
        re.findall(r"\bvfmadd(?:132|213|231)ps\b", full_disassembly)
    )
    if len(full_zmm_registers) < 10 or full_fma_count < 8:
        print(
            "AVX-512 4x32 full-tile microkernel lost its eight-chain packed body "
            f"(distinct_zmm={len(full_zmm_registers)}, "
            f"packed_fma={full_fma_count})",
            file=sys.stderr,
        )
        print(full_disassembly, file=sys.stderr)
        return 1

    print(
        "AVX-512 artifacts verified: "
        f"checked_symbol={CHECKED_SYMBOL} checked_zmm={checked_zmm_count} "
        f"checked_fma={checked_fma_count} full_tile_symbol={FULL_TILE_SYMBOL} "
        f"full_tile_distinct_zmm={len(full_zmm_registers)} "
        f"full_tile_fma={full_fma_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
