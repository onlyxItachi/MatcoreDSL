"""Tests for GPU elementwise operations including min/max."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import after path setup
import matcore as mc


def test_unary_op(name, np_func, dtype=np.float32, atol=1e-4):
    """Test a unary elementwise GPU op against numpy reference."""
    x = np.random.randn(64, 64).astype(dtype)
    if name in ("sqrt", "log"):
        x = np.abs(x) + 0.01  # ensure positive

    gpu_func = getattr(mc, f"{name}_gpu")
    result = gpu_func(x)
    expected = np_func(x)

    err = np.max(np.abs(result - expected))
    status = "PASS" if err < atol else "FAIL"
    print(f"  {name:12s}: max_err={err:.6f} [{status}]")
    assert err < atol, f"{name} failed: max_err={err}"


def test_binary_op(name, np_func, dtype=np.float32, atol=1e-4):
    """Test a binary elementwise GPU op against numpy reference."""
    a = np.random.randn(64, 64).astype(dtype)
    b = np.random.randn(64, 64).astype(dtype)
    if name == "div":
        b = np.abs(b) + 0.01  # avoid div by zero

    gpu_func = getattr(mc, f"{name}_gpu")
    result = gpu_func(a, b)
    expected = np_func(a, b)

    err = np.max(np.abs(result - expected))
    status = "PASS" if err < atol else "FAIL"
    print(f"  {name:12s}: max_err={err:.6f} [{status}]")
    assert err < atol, f"{name} failed: max_err={err}"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


if __name__ == "__main__":
    print("=== Unary Elementwise GPU Tests ===")
    test_unary_op("exp", np.exp)
    test_unary_op("log", np.log)
    test_unary_op("sqrt", np.sqrt)
    test_unary_op("tanh", np.tanh)
    test_unary_op("sigmoid", sigmoid)
    test_unary_op("gelu", gelu, atol=1e-3)
    test_unary_op("relu", lambda x: np.maximum(x, 0))
    test_unary_op("neg", np.negative)
    test_unary_op("abs", np.abs)

    print("\n=== Binary Elementwise GPU Tests ===")
    test_binary_op("add", np.add)
    test_binary_op("sub", np.subtract)
    test_binary_op("mul", np.multiply)
    test_binary_op("div", np.divide)
    test_binary_op("min", np.minimum)
    test_binary_op("max", np.maximum)

    print("\n=== FP16 Tests ===")
    test_unary_op("exp", np.exp, dtype=np.float16, atol=1e-2)
    test_unary_op("relu", lambda x: np.maximum(x, 0), dtype=np.float16, atol=1e-3)
    test_binary_op("add", np.add, dtype=np.float16, atol=1e-3)
    test_binary_op("min", np.minimum, dtype=np.float16, atol=1e-3)
    test_binary_op("max", np.maximum, dtype=np.float16, atol=1e-3)

    print("\n=== Shape Tests ===")
    for shape in [(1, 16), (32, 32), (128, 64), (1, 1024)]:
        x = np.random.randn(*shape).astype(np.float32)
        result = mc.relu_gpu(x)
        expected = np.maximum(x, 0)
        err = np.max(np.abs(result - expected))
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  relu {str(shape):12s}: max_err={err:.6f} [{status}]")
        assert err < 1e-5

    print("\nAll elementwise GPU tests passed! ✓")
