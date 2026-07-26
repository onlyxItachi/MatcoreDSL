#!/usr/bin/env python3
"""Require exact YMM/FMA code in the checked and production AVX2 kernels."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


CHECKED_SYMBOL = "matcore_cpu_packed_avx2_4x16_microkernel_f32_v1"
FULL_TILE_SYMBOL = "matcore_cpu_packed_avx2_4x16_full_microkernel_f32_v2"


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def disassemble(arguments: argparse.Namespace, symbol: str) -> str:
    return run(
        [
            arguments.objdump,
            "-d",
            "-M",
            "intel",
            "--no-show-raw-insn",
            f"--disassemble={symbol}",
            arguments.artifact,
        ]
    ).lower()


def packed_fma_count(disassembly: str) -> int:
    return len(re.findall(r"\bvfmadd(?:132|213|231)ps\b", disassembly))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--nm", default="nm")
    arguments = parser.parse_args()

    symbols = run([arguments.nm, "-C", arguments.artifact])
    for symbol in (CHECKED_SYMBOL, FULL_TILE_SYMBOL):
        if not re.search(rf"\b{re.escape(symbol)}$", symbols, re.MULTILINE):
            print(f"missing exact AVX2 microkernel symbol: {symbol}", file=sys.stderr)
            return 1

    checked = disassemble(arguments, CHECKED_SYMBOL)
    checked_ymm = len(re.findall(r"\bymm[0-9]+\b", checked))
    checked_fma = packed_fma_count(checked)
    if checked_ymm < 16 or checked_fma < 8:
        print(
            "checked AVX2 edge microkernel lost its packed body "
            f"(ymm={checked_ymm}, packed_fma={checked_fma})",
            file=sys.stderr,
        )
        print(checked, file=sys.stderr)
        return 1

    full = disassemble(arguments, FULL_TILE_SYMBOL)
    full_ymm_registers = set(re.findall(r"\bymm[0-9]+\b", full))
    full_fma = packed_fma_count(full)
    sections = run([arguments.objdump, "-h", arguments.artifact]).lower()
    debug_artifact = ".debug_info" in sections
    stack_reference = re.search(r"\b(?:rsp|esp)\b", full) is not None
    if (
        len(full_ymm_registers) < 10
        or full_fma < 8
        or (stack_reference and not debug_artifact)
    ):
        print(
            "AVX2 full-tile microkernel lost its eight-chain packed body "
            f"(distinct_ymm={len(full_ymm_registers)}, "
            f"packed_fma={full_fma}, stack_reference={stack_reference}, "
            f"debug_artifact={debug_artifact})",
            file=sys.stderr,
        )
        print(full, file=sys.stderr)
        return 1

    print(
        "AVX2 artifacts verified: "
        f"checked_symbol={CHECKED_SYMBOL} checked_ymm={checked_ymm} "
        f"checked_fma={checked_fma} full_tile_symbol={FULL_TILE_SYMBOL} "
        f"full_tile_distinct_ymm={len(full_ymm_registers)} "
        f"full_tile_fma={full_fma} "
        f"stack_reference={stack_reference} debug_artifact={debug_artifact}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
