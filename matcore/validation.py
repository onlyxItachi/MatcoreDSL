from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

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


_FP8_CLEAR_FAILURE_MESSAGES: tuple[str, ...] = (
    "float8_e4m3fn matmul is currently limited to nvidia-dgpu",
    "float8_e4m3fn matmul requires float32 output/accumulation",
    "float8_e4m3fn matmul requires native nvidia fp8 tensor-core support",
    "float8_e4m3fn matmul is eligible for nvidia fp8 wgmma on sm_90+",
    "float8_e4m3fn matmul is not eligible for nvidia fp8 wgmma",
    "float8_e4m3fn matmul requires a dedicated native nvidia fp8 wgmma lowering path",
)


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


def _require_rank2(lhs: np.ndarray, rhs: np.ndarray, *, op_name: str) -> None:
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError(f"{op_name} expects rank-2 tensors")
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"{op_name} requires lhs.shape[1] == rhs.shape[0]")


def make_reference_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_arr = np.ascontiguousarray(lhs)
    rhs_arr = np.ascontiguousarray(rhs)

    _require_rank2(lhs_arr, rhs_arr, op_name="reference matmul")
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


def make_reference_int8_i32_matmul(
    lhs: np.ndarray, rhs: np.ndarray, *, zero_point: int
) -> np.ndarray:
    lhs_i8 = np.ascontiguousarray(lhs, dtype=np.int8)
    rhs_i8 = np.ascontiguousarray(rhs, dtype=np.int8)
    _require_rank2(lhs_i8, rhs_i8, op_name="int8->int32 reference matmul")

    zp = int(zero_point)
    lhs_i32 = lhs_i8.astype(np.int32) - zp
    rhs_i32 = rhs_i8.astype(np.int32) - zp
    return lhs_i32 @ rhs_i32


def float32_to_bf16_storage(values: np.ndarray) -> np.ndarray:
    values_f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits_u32 = values_f32.view(np.uint32)
    rounded = bits_u32 + np.uint32(0x7FFF) + ((bits_u32 >> 16) & np.uint32(1))
    return (rounded >> 16).astype(np.uint16)


def bf16_storage_to_float32(storage: np.ndarray) -> np.ndarray:
    storage_u16 = np.ascontiguousarray(storage, dtype=np.uint16)
    bits_u32 = storage_u16.astype(np.uint32) << np.uint32(16)
    return bits_u32.view(np.float32)


def make_bf16_logical_tensor(values: np.ndarray) -> tuple[Any, np.ndarray]:
    storage = float32_to_bf16_storage(values)
    return mc.asdtype(storage, "bfloat16"), storage


def probe_backend_availability(target: str) -> tuple[bool, str | None]:
    lhs = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float16)
    rhs = np.array([[0.5, -2.0], [1.0, 4.0]], dtype=np.float16)
    out = np.zeros((2, 2), dtype=np.float16)
    try:
        mc.launch(matmul_kernel, lhs, rhs, out, target=target)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def is_clear_fp8_unavailable_message(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in _FP8_CLEAR_FAILURE_MESSAGES)


def run_fp8_case_with_capability(target: str) -> tuple[str, str | np.ndarray]:
    lhs_storage = np.array([[0x3C, 0x40], [0x44, 0x38]], dtype=np.uint8)
    rhs_storage = np.array([[0x3C, 0x44], [0x40, 0x38]], dtype=np.uint8)
    lhs = mc.asdtype(lhs_storage, "float8_e4m3fn")
    rhs = mc.asdtype(rhs_storage, "float8_e4m3fn")
    out = np.zeros((2, 2), dtype=np.float32)

    try:
        mc.launch(matmul_kernel, lhs, rhs, out, target=target)
        return "ok", out.copy()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if not is_clear_fp8_unavailable_message(message):
            raise AssertionError(
                f"FP8 failure message for target '{target}' is not clear: {message}"
            ) from exc
        return "unsupported-clear", message


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


def check_int8_i32_global_quant_correctness(
    kernel: MatCoreKernel,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    target: str = "x86-avx512",
    zero_point: int = 0,
    scale: float = 1.0,
    out: np.ndarray | None = None,
) -> MatmulCorrectnessReport:
    lhs_i8 = np.ascontiguousarray(lhs, dtype=np.int8)
    rhs_i8 = np.ascontiguousarray(rhs, dtype=np.int8)
    _require_rank2(lhs_i8, rhs_i8, op_name="int8->int32 matmul")

    result = out
    expected_shape = (lhs_i8.shape[0], rhs_i8.shape[1])
    if result is None:
        result = np.zeros(expected_shape, dtype=np.int32)
    else:
        if result.dtype != np.int32:
            raise TypeError("int8->int32 correctness check requires int32 output tensor")
        if result.shape != expected_shape:
            raise ValueError("output shape must match the matmul result shape")
        if not result.flags.c_contiguous:
            raise ValueError("output tensor must be C-contiguous")

    expected = make_reference_int8_i32_matmul(lhs_i8, rhs_i8, zero_point=zero_point)
    lhs_q = mc.asdtype(lhs_i8, "int8", scale=scale, zero_point=zero_point)
    rhs_q = mc.asdtype(rhs_i8, "int8", scale=scale, zero_point=zero_point)

    before_ptr = result.__array_interface__["data"][0]
    start = perf_counter()
    mc.launch(
        kernel,
        lhs_q,
        rhs_q,
        result,
        target=target,
        quant={"scale": float(scale), "zero_point": int(zero_point), "enabled": True},
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    after_ptr = result.__array_interface__["data"][0]

    abs_diff = np.abs(result.astype(np.int64) - expected.astype(np.int64))
    denom = np.maximum(np.abs(expected.astype(np.float64)), 1.0)
    rel_diff = abs_diff.astype(np.float64) / denom

    np.testing.assert_array_equal(result, expected)
    return MatmulCorrectnessReport(
        target=target,
        dtype="int8->int32",
        lhs_shape=tuple(int(dim) for dim in lhs_i8.shape),
        rhs_shape=tuple(int(dim) for dim in rhs_i8.shape),
        atol=0.0,
        rtol=0.0,
        max_abs_error=float(abs_diff.max()) if abs_diff.size else 0.0,
        max_rel_error=float(rel_diff.max()) if rel_diff.size else 0.0,
        exact_match=bool(np.array_equal(result, expected)),
        zero_copy_output=before_ptr == after_ptr,
        elapsed_ms=elapsed_ms,
    )


def check_bf16_storage_f32_output_correctness(
    kernel: MatCoreKernel,
    lhs_values_f32: np.ndarray,
    rhs_values_f32: np.ndarray,
    *,
    target: str = "x86-avx512",
    out: np.ndarray | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> MatmulCorrectnessReport:
    lhs_f32 = np.ascontiguousarray(lhs_values_f32, dtype=np.float32)
    rhs_f32 = np.ascontiguousarray(rhs_values_f32, dtype=np.float32)
    _require_rank2(lhs_f32, rhs_f32, op_name="bf16-storage->f32 matmul")

    lhs_tensor, lhs_storage = make_bf16_logical_tensor(lhs_f32)
    rhs_tensor, rhs_storage = make_bf16_logical_tensor(rhs_f32)
    lhs_effective = bf16_storage_to_float32(lhs_storage)
    rhs_effective = bf16_storage_to_float32(rhs_storage)
    expected = lhs_effective @ rhs_effective

    result = out
    expected_shape = (lhs_f32.shape[0], rhs_f32.shape[1])
    if result is None:
        result = np.zeros(expected_shape, dtype=np.float32)
    else:
        if result.dtype != np.float32:
            raise TypeError(
                "bf16-storage->f32 correctness check requires float32 output tensor"
            )
        if result.shape != expected_shape:
            raise ValueError("output shape must match the matmul result shape")
        if not result.flags.c_contiguous:
            raise ValueError("output tensor must be C-contiguous")

    before_ptr = result.__array_interface__["data"][0]
    start = perf_counter()
    mc.launch(kernel, lhs_tensor, rhs_tensor, result, target=target)
    elapsed_ms = (perf_counter() - start) * 1000.0
    after_ptr = result.__array_interface__["data"][0]

    result_err = _as_error_dtype(result)
    expected_err = _as_error_dtype(expected)
    abs_diff = np.abs(result_err - expected_err)
    denom = np.maximum(np.abs(expected_err), np.finfo(result_err.dtype).eps)
    rel_diff = abs_diff / denom

    np.testing.assert_allclose(result_err, expected_err, atol=atol, rtol=rtol)
    return MatmulCorrectnessReport(
        target=target,
        dtype="bfloat16-storage->float32",
        lhs_shape=tuple(int(dim) for dim in lhs_f32.shape),
        rhs_shape=tuple(int(dim) for dim in rhs_f32.shape),
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


def benchmark_int8_i32_global_quantized_matmul(
    kernel: MatCoreKernel,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    target: str,
    repeats: int,
    zero_point: int,
    scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    lhs_i8 = np.ascontiguousarray(lhs, dtype=np.int8)
    rhs_i8 = np.ascontiguousarray(rhs, dtype=np.int8)
    _require_rank2(lhs_i8, rhs_i8, op_name="int8->int32 benchmark matmul")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    out = np.zeros((lhs_i8.shape[0], rhs_i8.shape[1]), dtype=np.int32)
    lhs_q = mc.asdtype(lhs_i8, "int8", scale=scale, zero_point=zero_point)
    rhs_q = mc.asdtype(rhs_i8, "int8", scale=scale, zero_point=zero_point)
    quant = {"scale": float(scale), "zero_point": int(zero_point), "enabled": True}

    mc.launch(kernel, lhs_q, rhs_q, out, target=target, quant=quant)
    start = perf_counter()
    for _ in range(repeats):
        mc.launch(kernel, lhs_q, rhs_q, out, target=target, quant=quant)
    elapsed_ms = (perf_counter() - start) * 1000.0 / float(repeats)
    return out, elapsed_ms


def benchmark_bf16_storage_f32_matmul(
    kernel: MatCoreKernel,
    lhs_values_f32: np.ndarray,
    rhs_values_f32: np.ndarray,
    *,
    target: str,
    repeats: int,
) -> tuple[np.ndarray, float]:
    lhs_f32 = np.ascontiguousarray(lhs_values_f32, dtype=np.float32)
    rhs_f32 = np.ascontiguousarray(rhs_values_f32, dtype=np.float32)
    _require_rank2(lhs_f32, rhs_f32, op_name="bf16-storage->f32 benchmark matmul")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    lhs_tensor, _ = make_bf16_logical_tensor(lhs_f32)
    rhs_tensor, _ = make_bf16_logical_tensor(rhs_f32)
    out = np.zeros((lhs_f32.shape[0], rhs_f32.shape[1]), dtype=np.float32)

    mc.launch(kernel, lhs_tensor, rhs_tensor, out, target=target)
    start = perf_counter()
    for _ in range(repeats):
        mc.launch(kernel, lhs_tensor, rhs_tensor, out, target=target)
    elapsed_ms = (perf_counter() - start) * 1000.0 / float(repeats)
    return out, elapsed_ms
