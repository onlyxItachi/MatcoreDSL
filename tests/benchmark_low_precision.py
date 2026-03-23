from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcore import matmul_kernel
from matcore.validation import (
    benchmark_bf16_storage_f32_matmul,
    benchmark_int8_i32_global_quantized_matmul,
    probe_backend_availability,
)


DEFAULT_TARGETS = ("x86-avx512", "nvidia-dgpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    return parser.parse_args()


def build_bf16_inputs(
    m: int, k: int, n: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lhs = rng.standard_normal((m, k), dtype=np.float32)
    rhs = rng.standard_normal((k, n), dtype=np.float32)
    return lhs, rhs


def build_int8_inputs(
    m: int, k: int, n: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lhs = rng.integers(-12, 12, size=(m, k), dtype=np.int8)
    rhs = rng.integers(-12, 12, size=(k, n), dtype=np.int8)
    return lhs, rhs


def format_header(args: argparse.Namespace) -> str:
    return (
        "MatCore low-precision benchmark: "
        f"{args.m}x{args.k} @ {args.k}x{args.n}, repeats={args.repeats}"
    )


def run_target(target: str, args: argparse.Namespace) -> None:
    if target == "nvidia-dgpu":
        ok, reason = probe_backend_availability(target)
        if not ok:
            print(f"{target}: SKIP ({reason})")
            return

    bf16_lhs, bf16_rhs = build_bf16_inputs(args.m, args.k, args.n, seed=args.seed)
    int8_lhs, int8_rhs = build_int8_inputs(args.m, args.k, args.n, seed=args.seed + 1)

    bf16_out, bf16_ms = benchmark_bf16_storage_f32_matmul(
        matmul_kernel,
        bf16_lhs,
        bf16_rhs,
        target=target,
        repeats=args.repeats,
    )
    bf16_checksum = float(np.sum(bf16_out.astype(np.float64)))
    print(
        f"{target} bf16-storage->f32: {bf16_ms:.3f} ms "
        f"(checksum={bf16_checksum:.6e})"
    )

    int8_out, int8_ms = benchmark_int8_i32_global_quantized_matmul(
        matmul_kernel,
        int8_lhs,
        int8_rhs,
        target=target,
        repeats=args.repeats,
        zero_point=2,
        scale=0.5,
    )
    int8_checksum = int(np.sum(int8_out.astype(np.int64)))
    print(
        f"{target} int8->int32 (global zp=2): {int8_ms:.3f} ms "
        f"(checksum={int8_checksum})"
    )


def main() -> None:
    args = parse_args()
    print(format_header(args))
    for target in args.targets:
        try:
            run_target(target, args)
        except Exception as exc:
            print(f"{target}: FAILED ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
