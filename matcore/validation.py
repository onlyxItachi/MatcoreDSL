from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .frontend import MatCoreKernel, mc


@mc.kernel
def matmul_kernel(a, b, c):
    lhs = mc.load(a)
    rhs = mc.load(b)
    out = mc.matmul(lhs, rhs)
    mc.store(c, out)


@dataclass(frozen=True)
class MatmulCorrectnessReport:
    target: str
    dtype: str
    lhs_shape: tuple[int, int]
    rhs_shape: tuple[int, int]
    atol: float
    rtol: float
    max_abs_error: float
    max_rel_error: float
    exact_match: bool
    zero_copy_output: bool
    elapsed_ms: float


def _dtype_name(array: np.ndarray) -> str:
    return np.dtype(array.dtype).name


def _default_tolerances(dtype_name: str) -> tuple[float, float]:
    if dtype_name == "float16":
        return (1e-2, 1e-2)
    if dtype_name == "bfloat16":
        return (2e-2, 2e-2)
    return (1e-5, 1e-5)


def _as_error_dtype(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def make_reference_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_arr = np.ascontiguousarray(lhs)
    rhs_arr = np.ascontiguousarray(rhs)

    if lhs_arr.ndim != 2 or rhs_arr.ndim != 2:
      raise ValueError("reference matmul expects rank-2 tensors")
    if lhs_arr.shape[1] != rhs_arr.shape[0]:
      raise ValueError("reference matmul requires lhs.shape[1] == rhs.shape[0]")
    if lhs_arr.dtype != rhs_arr.dtype:
      raise TypeError("reference matmul requires matching lhs/rhs dtypes")

    out = np.zeros((lhs_arr.shape[0], rhs_arr.shape[1]), dtype=lhs_arr.dtype)
    scratch = np.empty_like(out)
    for reduction_idx in range(lhs_arr.shape[1]):
        np.multiply(
            lhs_arr[:, reduction_idx : reduction_idx + 1],
            rhs_arr[reduction_idx : reduction_idx + 1, :],
            out=scratch,
            casting="same_kind",
        )
        np.add(out, scratch, out=out, casting="same_kind")
    return out


def check_matmul_correctness(
    kernel: MatCoreKernel,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    target: str = "x86-auto",
    out: np.ndarray | None = None,
    atol: float | None = None,
    rtol: float | None = None,
) -> MatmulCorrectnessReport:
    lhs_arr = np.ascontiguousarray(lhs)
    rhs_arr = np.ascontiguousarray(rhs)
    if lhs_arr.ndim != 2 or rhs_arr.ndim != 2:
        raise ValueError("MatCore correctness check expects rank-2 tensors")
    if lhs_arr.shape[1] != rhs_arr.shape[0]:
        raise ValueError("lhs.shape[1] must equal rhs.shape[0] for matmul")
    if lhs_arr.dtype != rhs_arr.dtype:
        raise TypeError("MatCore correctness check requires matching lhs/rhs dtypes")

    result = out
    if result is None:
        result = np.zeros((lhs_arr.shape[0], rhs_arr.shape[1]), dtype=lhs_arr.dtype)
    else:
        if result.dtype != lhs_arr.dtype:
            raise TypeError("output dtype must match the input dtype")
        if result.shape != (lhs_arr.shape[0], rhs_arr.shape[1]):
            raise ValueError("output shape must match the matmul result shape")
        if not result.flags.c_contiguous:
            raise ValueError("output tensor must be C-contiguous")

    expected = make_reference_matmul(lhs_arr, rhs_arr)
    before_ptr = result.__array_interface__["data"][0]
    start = perf_counter()
    mc.launch(kernel, lhs_arr, rhs_arr, result, target=target)
    elapsed_ms = (perf_counter() - start) * 1000.0
    after_ptr = result.__array_interface__["data"][0]

    dtype_name = _dtype_name(lhs_arr)
    default_atol, default_rtol = _default_tolerances(dtype_name)
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol

    result_err = _as_error_dtype(result)
    expected_err = _as_error_dtype(expected)
    abs_diff = np.abs(result_err - expected_err)
    denom = np.maximum(np.abs(expected_err), np.finfo(result_err.dtype).eps)
    rel_diff = abs_diff / denom

    np.testing.assert_allclose(result_err, expected_err, atol=atol, rtol=rtol)
    return MatmulCorrectnessReport(
        target=target,
        dtype=dtype_name,
        lhs_shape=tuple(int(dim) for dim in lhs_arr.shape),
        rhs_shape=tuple(int(dim) for dim in rhs_arr.shape),
        atol=atol,
        rtol=rtol,
        max_abs_error=float(abs_diff.max()) if abs_diff.size else 0.0,
        max_rel_error=float(rel_diff.max()) if rel_diff.size else 0.0,
        exact_match=bool(np.array_equal(result, expected)),
        zero_copy_output=before_ptr == after_ptr,
        elapsed_ms=elapsed_ms,
    )


def benchmark_numpy_matmul(lhs: np.ndarray, rhs: np.ndarray, repeats: int) -> tuple[np.ndarray, float]:
    lhs_arr = np.ascontiguousarray(lhs)
    rhs_arr = np.ascontiguousarray(rhs)
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    output = lhs_arr @ rhs_arr
    start = perf_counter()
    for _ in range(repeats):
        output = lhs_arr @ rhs_arr
    elapsed_ms = (perf_counter() - start) * 1000.0 / float(repeats)
    return output, elapsed_ms


def benchmark_matcore_matmul(
    kernel: MatCoreKernel,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    target: str,
    repeats: int,
) -> tuple[np.ndarray, float]:
    lhs_arr = np.ascontiguousarray(lhs)
    rhs_arr = np.ascontiguousarray(rhs)
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    out = np.zeros((lhs_arr.shape[0], rhs_arr.shape[1]), dtype=lhs_arr.dtype)
    mc.launch(kernel, lhs_arr, rhs_arr, out, target=target)

    start = perf_counter()
    for _ in range(repeats):
        mc.launch(kernel, lhs_arr, rhs_arr, out, target=target)
    elapsed_ms = (perf_counter() - start) * 1000.0 / float(repeats)
    return out, elapsed_ms
