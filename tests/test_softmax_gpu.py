"""Tests for GPU softmax operation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matcore as mc


def softmax_ref(x):
    """Row-wise softmax reference."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    max_vals = np.max(x, axis=-1, keepdims=True)
    exp_vals = np.exp(x - max_vals)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def test_softmax(shape, dtype=np.float32, atol=1e-4):
    x = np.random.randn(*shape).astype(dtype)
    result = mc.softmax_gpu(x)
    expected = softmax_ref(x)
    err = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    status = "PASS" if err < atol else "FAIL"
    print(f"  softmax {str(shape):15s} {str(dtype):10s}: max_err={err:.6f} [{status}]")
    assert err < atol, f"softmax {shape} failed: max_err={err}"


if __name__ == "__main__":
    print("=== Softmax GPU Tests ===")
    test_softmax((1, 64))
    test_softmax((32, 32))
    test_softmax((64, 128))
    test_softmax((128, 256))
    test_softmax((1, 1024))

    print("\n=== Softmax FP16 ===")
    test_softmax((32, 32), dtype=np.float16, atol=1e-2)
    test_softmax((64, 64), dtype=np.float16, atol=1e-2)

    print("\nAll softmax GPU tests passed! ✓")
