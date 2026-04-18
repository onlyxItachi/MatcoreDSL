"""End-to-end Family C fusion tests: softmax(Q @ K.T) @ V on GPU."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matcore as mc


def test_attention_small():
    """Simple attention: softmax(Q @ K.T) @ V."""

    @mc.fused
    def attention(Q, K, V):
        return mc.softmax(Q @ K.T) @ V

    M, N, D = 16, 16, 8
    Q = np.random.randn(M, D).astype(np.float32) * 0.1
    K = np.random.randn(N, D).astype(np.float32) * 0.1
    V = np.random.randn(N, D).astype(np.float32) * 0.1

    result = attention(Q, K, V)

    scores = Q @ K.T / np.sqrt(D)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    expected = probs @ V

    err = np.max(np.abs(result - expected))
    tol = 1e-3
    print(f"  attention ({M}x{N}x{D}): max_err={err:.6f}")
    assert err < tol


if __name__ == "__main__":
    print("=== Family C End-to-End GPU Tests ===")
    test_attention_small()
    print("\nAll Family C GPU tests passed! ✓")
