from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from matcore import check_matmul_correctness, matmul_kernel
from matcore.validation import (
    check_bf16_storage_f32_output_correctness,
    check_int8_i32_global_quant_correctness,
    probe_backend_availability,
    run_fp8_case_with_capability,
)


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


def run_int8_i32_global_quant_case(*, target: str) -> None:
    lhs = np.array([[4, -3, 2], [1, 0, -2]], dtype=np.int8)
    rhs = np.array([[2, -1], [3, 4], [-2, 1]], dtype=np.int8)
    report = check_int8_i32_global_quant_correctness(
        matmul_kernel,
        lhs,
        rhs,
        target=target,
        zero_point=2,
        scale=0.5,
    )
    assert report.zero_copy_output
    assert report.exact_match, report
    print(
        f"{target} int8->int32 2x3@3x2 (zp=2): "
        f"exact={report.exact_match} elapsed={report.elapsed_ms:.3f} ms"
    )


def run_bf16_storage_case(*, target: str) -> None:
    lhs_f32 = np.array([[1.5, -2.0], [0.25, 3.0]], dtype=np.float32)
    rhs_f32 = np.array([[2.0, 0.5], [-1.0, 4.0]], dtype=np.float32)
    out = np.zeros((2, 2), dtype=np.float32)
    report = check_bf16_storage_f32_output_correctness(
        matmul_kernel,
        lhs_f32,
        rhs_f32,
        target=target,
        out=out,
        atol=1e-3,
        rtol=1e-3,
    )
    assert report.zero_copy_output
    print(
        f"{target} bfloat16-storage->float32 2x2@2x2: "
        f"max_abs={report.max_abs_error:.6e} elapsed={report.elapsed_ms:.3f} ms"
    )


def run_fp8_capability_case(*, target: str) -> None:
    status, payload = run_fp8_case_with_capability(target)
    if status == "ok":
        out = np.asarray(payload, dtype=np.float16)
        assert out.shape == (2, 2)
        assert np.isfinite(out).all()
        print(f"{target} fp8_e4m3fn: ok")
        return

    message = str(payload)
    assert "float8" in message.lower() or "f8e4m3fn" in message.lower()
    print(f"{target} fp8_e4m3fn: unsupported-clear ({message.splitlines()[0]})")


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

    run_int8_i32_global_quant_case(target="x86-avx512")
    run_bf16_storage_case(target="x86-avx512")
    run_fp8_capability_case(target="x86-avx512")

    nvidia_ok, nvidia_reason = probe_backend_availability("nvidia-dgpu")
    if nvidia_ok:
        run_bf16_storage_case(target="nvidia-dgpu")
        run_int8_i32_global_quant_case(target="nvidia-dgpu")
        run_fp8_capability_case(target="nvidia-dgpu")
    else:
        assert nvidia_reason is not None and len(nvidia_reason) > 0
        print(f"nvidia-dgpu unavailable, skipping bf16/int8/fp8 checks: {nvidia_reason}")


if __name__ == "__main__":
    main()
