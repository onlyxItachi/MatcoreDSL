"""Tests for Family B: matmul → pointwise → matmul tile chain."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import matcore as mc
from matcore.frontend import FusionTraceBuilder


def _check_close(name: str, result: np.ndarray, expected: np.ndarray, tol: float) -> None:
    err = float(np.max(np.abs(result - expected)))
    status = "PASS" if err < tol else "FAIL"
    print(f"  {name}: max_err={err:.6f} [{status}]")
    assert err < tol, f"{name} max_err={err:.6f} >= tol={tol}"


def test_trace_family_b():
    """Verify matmul → relu → matmul traces correctly."""
    builder = FusionTraceBuilder()
    A = builder.add_input("A", np.random.randn(32, 16).astype(np.float32))
    B = builder.add_input("B", np.random.randn(16, 32).astype(np.float32))
    W = builder.add_input("W", np.random.randn(32, 24).astype(np.float32))
    graph = builder.finish(mc.relu(A @ B) @ W)

    ops = [n["op"] for n in graph["nodes"]]
    assert "matmul" in ops, f"Missing matmul in {ops}"
    assert ops.count("matmul") == 2, f"Expected 2 matmuls in {ops}"
    print(f"  trace_family_b: PASS (ops={ops})")


def test_trace_family_b_binary():
    """Verify matmul → binary glue → matmul traces correctly."""
    builder = FusionTraceBuilder()
    A = builder.add_input("A", np.random.randn(32, 16).astype(np.float32))
    B = builder.add_input("B", np.random.randn(16, 32).astype(np.float32))
    bias = builder.add_input("bias", np.random.randn(32, 32).astype(np.float32))
    W = builder.add_input("W", np.random.randn(32, 24).astype(np.float32))
    graph = builder.finish(((A @ B) + bias) @ W)

    ops = [n["op"] for n in graph["nodes"]]
    assert ops == ["matmul", "add", "matmul"], f"Unexpected ops {ops}"
    assert graph["input_values"] == [0, 1, 2, 3]
    print(f"  trace_family_b_binary: PASS (ops={ops})")


def test_gemm_relu_gemm(M: int = 32, N1: int = 32, K: int = 16, N2: int = 24):
    """matmul → relu → matmul fusion."""

    @mc.fused
    def chain_relu(A, B, W):
        return mc.relu(A @ B) @ W

    A = np.random.randn(M, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N1).astype(np.float32) * 0.1
    W = np.random.randn(N1, N2).astype(np.float32) * 0.1

    result = chain_relu(A, B, W)
    expected = np.maximum(A @ B, 0) @ W
    _check_close(f"gemm_relu_gemm ({M}x{K}x{N1}x{N2})", result, expected, tol=1e-2)


def test_gemm_exp_gemm(M: int = 32, N1: int = 32, K: int = 16, N2: int = 24):
    """matmul → exp → matmul fusion."""

    @mc.fused
    def chain_exp(A, B, W):
        return mc.exp(A @ B) @ W

    A = np.random.randn(M, K).astype(np.float32) * 0.05
    B = np.random.randn(K, N1).astype(np.float32) * 0.05
    W = np.random.randn(N1, N2).astype(np.float32) * 0.1

    result = chain_exp(A, B, W)
    expected = np.exp(A @ B) @ W
    _check_close(f"gemm_exp_gemm ({M}x{K}x{N1}x{N2})", result, expected, tol=1e-2)


def test_gemm_add_gemm(M: int = 32, N1: int = 32, K: int = 16, N2: int = 24):
    """matmul → dense bias add → matmul fusion."""

    @mc.fused
    def chain_add(A, B, bias, W):
        return ((A @ B) + bias) @ W

    A = np.random.randn(M, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N1).astype(np.float32) * 0.1
    bias = np.random.randn(M, N1).astype(np.float32) * 0.1
    W = np.random.randn(N1, N2).astype(np.float32) * 0.1

    result = chain_add(A, B, bias, W)
    expected = ((A @ B) + bias) @ W
    _check_close(f"gemm_add_gemm ({M}x{K}x{N1}x{N2})", result, expected, tol=1e-2)


def test_bias_minus_gemm_gemm(M: int = 32, N1: int = 32, K: int = 16, N2: int = 24):
    """matmul → dense bias minus intermediate → matmul fusion."""

    @mc.fused
    def chain_bias_minus(A, B, bias, W):
        return (bias - (A @ B)) @ W

    A = np.random.randn(M, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N1).astype(np.float32) * 0.1
    bias = np.random.randn(M, N1).astype(np.float32) * 0.1
    W = np.random.randn(N1, N2).astype(np.float32) * 0.1

    result = chain_bias_minus(A, B, bias, W)
    expected = (bias - (A @ B)) @ W
    _check_close(f"bias_minus_gemm_gemm ({M}x{K}x{N1}x{N2})", result, expected, tol=1e-2)


if __name__ == "__main__":
    print("=== Family B Tests ===")
    test_trace_family_b()
    test_trace_family_b_binary()
    test_gemm_relu_gemm()
    test_gemm_exp_gemm()
    test_gemm_add_gemm()
    test_bias_minus_gemm_gemm()
    print("\nAll Family B tests passed! ✓")
