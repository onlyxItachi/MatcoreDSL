from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from matcore import check_matmul_correctness, matmul_kernel


def make_inputs(
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


def run_case(
    *,
    target: str,
    dtype: np.dtype,
    shape: tuple[int, int, int],
    seed: int,
) -> None:
    m, k, n = shape
    lhs, rhs = make_inputs(m, k, n, dtype=dtype, seed=seed)
    out = np.zeros((m, n), dtype=dtype)
    report = check_matmul_correctness(
        matmul_kernel,
        lhs,
        rhs,
        target=target,
        out=out,
    )
    assert report.zero_copy_output
    assert report.exact_match, report
    print(
        f"{target} {np.dtype(dtype).name} {m}x{k}@{k}x{n}: "
        f"exact={report.exact_match} elapsed={report.elapsed_ms:.3f} ms"
    )


def maybe_bfloat16_dtype() -> np.dtype | None:
    try:
        return np.dtype("bfloat16")
    except TypeError:
        return None


def main() -> None:
    cases: list[tuple[str, np.dtype, tuple[int, int, int], int]] = [
        ("x86-auto", np.dtype(np.float32), (2, 2, 2), 7),
        ("x86-auto", np.dtype(np.float32), (9, 13, 5), 11),
        ("x86-avx2", np.dtype(np.float16), (16, 16, 16), 13),
        ("x86-avx512", np.dtype(np.float16), (24, 16, 12), 17),
    ]

    bfloat16_dtype = maybe_bfloat16_dtype()
    if bfloat16_dtype is not None:
        cases.append(("x86-auto", bfloat16_dtype, (8, 10, 6), 19))

    for target, dtype, shape, seed in cases:
        run_case(target=target, dtype=dtype, shape=shape, seed=seed)


if __name__ == "__main__":
    main()
