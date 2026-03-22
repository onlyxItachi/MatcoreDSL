from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcore import (
    benchmark_matcore_matmul,
    benchmark_numpy_matmul,
    check_matmul_correctness,
    matmul_kernel,
)

DEFAULT_TARGETS = ("x86-auto", "x86-avx2", "x86-avx512", "nvidia-dgpu")


def parse_dtype(name: str) -> np.dtype:
    lowered = name.strip().lower()
    if lowered == "float16":
        return np.dtype(np.float16)
    if lowered == "float32":
        return np.dtype(np.float32)
    if lowered == "bfloat16":
        return np.dtype("bfloat16")
    raise ValueError(f"Unsupported dtype '{name}'")


def build_inputs(
    m: int,
    k: int,
    n: int,
    *,
    dtype: np.dtype,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lhs = rng.standard_normal((m, k), dtype=np.float32).astype(dtype)
    rhs = rng.standard_normal((k, n), dtype=np.float32).astype(dtype)
    return lhs, rhs


def run_correctness_probe(lhs: np.ndarray, rhs: np.ndarray, target: str) -> None:
    probe_m = min(lhs.shape[0], 32)
    probe_k = min(lhs.shape[1], 32)
    probe_n = min(rhs.shape[1], 32)
    probe_lhs = lhs[:probe_m, :probe_k].copy()
    probe_rhs = rhs[:probe_k, :probe_n].copy()
    report = check_matmul_correctness(
        matmul_kernel,
        probe_lhs,
        probe_rhs,
        target=target,
    )
    print(
        f"  probe[{target}]: exact={report.exact_match} "
        f"max_abs={report.max_abs_error:.6e} max_rel={report.max_rel_error:.6e}"
    )


def format_speedup(numpy_ms: float, other_ms: float) -> str:
    if other_ms <= 0.0:
        return "n/a"
    return f"{numpy_ms / other_ms:.2f}x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260322)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    lhs, rhs = build_inputs(args.m, args.k, args.n, dtype=dtype, seed=args.seed)

    print(
        f"MatCore vs NumPy benchmark: "
        f"{args.m}x{args.k} @ {args.k}x{args.n}, dtype={dtype.name}, repeats={args.repeats}"
    )

    numpy_out, numpy_ms = benchmark_numpy_matmul(lhs, rhs, repeats=args.repeats)
    numpy_checksum = float(np.sum(numpy_out.astype(np.float64)))
    print(f"numpy: {numpy_ms:.3f} ms (checksum={numpy_checksum:.6e})")

    for target in args.targets:
        try:
            run_correctness_probe(lhs, rhs, target)
            matcore_out, matcore_ms = benchmark_matcore_matmul(
                matmul_kernel,
                lhs,
                rhs,
                target=target,
                repeats=args.repeats,
            )
            checksum = float(np.sum(matcore_out.astype(np.float64)))
            print(
                f"matcore[{target}]: {matcore_ms:.3f} ms "
                f"(speedup={format_speedup(numpy_ms, matcore_ms)}, checksum={checksum:.6e})"
            )
        except Exception as exc:
            print(f"matcore[{target}]: FAILED ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
