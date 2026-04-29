"""Fusion route and unsupported-pattern contracts."""
from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import matcore as mc
from matcore import frontend


def _assert_rejected_before_backend(
    name: str,
    call: Callable[[], object],
    exc_type: type[BaseException],
    expected_substring: str,
) -> None:
    original_get_native_module = frontend._get_native_module

    def fail_native_dispatch() -> object:
        raise AssertionError(f"{name} reached native dispatch unexpectedly")

    frontend._get_native_module = fail_native_dispatch
    try:
        try:
            call()
        except exc_type as exc:
            message = str(exc)
            assert expected_substring.lower() in message.lower(), message
            print(f"  {name}: unsupported-clear ({message.splitlines()[0]})")
        else:
            raise AssertionError(f"Expected {name} to raise {exc_type.__name__}")
    finally:
        frontend._get_native_module = original_get_native_module


def test_family_a_scalar_epilogue_rejected_before_backend() -> None:
    """Family A scalar epilogues are unsupported until tracer broadcasting exists."""

    @mc.fused
    def gemm_scalar_bias(A, B):
        return (A @ B) + 1.0

    A = np.zeros((4, 3), dtype=np.float32)
    B = np.zeros((3, 5), dtype=np.float32)
    _assert_rejected_before_backend(
        "family_a_scalar_epilogue",
        lambda: gemm_scalar_bias(A, B),
        TypeError,
        "scalar broadcasting",
    )


def test_family_b_scalar_glue_rejected_before_backend() -> None:
    """Family B scalar glue is unsupported until tracer broadcasting exists."""

    @mc.fused
    def chain_scalar_glue(A, B, W):
        return (((A @ B) + 1.0) @ W)

    A = np.zeros((4, 3), dtype=np.float32)
    B = np.zeros((3, 5), dtype=np.float32)
    W = np.zeros((5, 2), dtype=np.float32)
    _assert_rejected_before_backend(
        "family_b_scalar_glue",
        lambda: chain_scalar_glue(A, B, W),
        TypeError,
        "scalar broadcasting",
    )


def test_fused_matmul_mismatched_input_dtypes_rejected_before_backend() -> None:
    """Fused matmul requires matching lhs/rhs dtypes at trace time."""

    @mc.fused
    def gemm_relu(A, B):
        return mc.relu(A @ B)

    A = np.zeros((4, 3), dtype=np.float32)
    B = np.zeros((3, 5), dtype=np.float16)
    _assert_rejected_before_backend(
        "fused_mismatched_matmul_dtypes",
        lambda: gemm_relu(A, B),
        TypeError,
        "matching lhs/rhs dtypes",
    )


def test_fused_multi_output_rejected_before_backend() -> None:
    """The fused frontend currently exposes one output tensor per call."""

    @mc.fused
    def two_outputs(A, B):
        product = A @ B
        return mc.relu(product), product

    A = np.zeros((4, 3), dtype=np.float32)
    B = np.zeros((3, 5), dtype=np.float32)
    _assert_rejected_before_backend(
        "fused_multi_output",
        lambda: two_outputs(A, B),
        ValueError,
        "exactly one graph output",
    )


def test_elementwise_before_matmul_rejected_by_fusion_emitter() -> None:
    """Analyzer may discover this region, but the emitter must reject it clearly."""

    @mc.fused
    def sin_then_matmul(A, B):
        return mc.sin(A) @ B

    A = np.zeros((4, 3), dtype=np.float32)
    B = np.zeros((3, 5), dtype=np.float32)
    try:
        sin_then_matmul(A, B)
    except RuntimeError as exc:
        message = str(exc)
        assert "elementwise-before-matmul fusion is not implemented" in message, message
        print(f"  elementwise_before_matmul: unsupported-clear ({message.splitlines()[0]})")
    else:
        raise AssertionError("Expected elementwise-before-matmul fusion to be rejected")


def main() -> None:
    test_family_a_scalar_epilogue_rejected_before_backend()
    test_family_b_scalar_glue_rejected_before_backend()
    test_fused_matmul_mismatched_input_dtypes_rejected_before_backend()
    test_fused_multi_output_rejected_before_backend()
    test_elementwise_before_matmul_rejected_by_fusion_emitter()
    print("Fusion contract tests passed")


if __name__ == "__main__":
    main()
