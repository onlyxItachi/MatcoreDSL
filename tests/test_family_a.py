"""End-to-end Family A fusion tests: GEMM + pointwise epilogue on GPU."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matcore as mc

def test_gemm_relu(M=64, N=64, K=32):
    """GEMM + ReLU fusion."""
    @mc.fused
    def gemm_relu(A, B):
        return mc.relu(A @ B)

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    result = gemm_relu(A, B)

    expected = np.maximum(A @ B, 0)
    err = np.max(np.abs(result - expected))
    tol = 1e-3
    status = "PASS" if err < tol else "FAIL"
    print(f"  gemm_relu ({M}x{K}x{N}): max_err={err:.6f} [{status}]")
    assert err < tol, f"gemm_relu failed: {err}"

def test_gemm_gelu(M=64, N=64, K=32):
    """GEMM + GELU fusion."""
    @mc.fused
    def gemm_gelu(A, B):
        return mc.gelu(A @ B)

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    result = gemm_gelu(A, B)

    C = A @ B
    expected = 0.5 * C * (1 + np.tanh(np.sqrt(2/np.pi) * (C + 0.044715 * C**3)))
    err = np.max(np.abs(result - expected))
    tol = 1e-3
    status = "PASS" if err < tol else "FAIL"
    print(f"  gemm_gelu ({M}x{K}x{N}): max_err={err:.6f} [{status}]")
    assert err < tol, f"gemm_gelu failed: {err}"

def test_gemm_exp(M=32, N=32, K=16):
    """GEMM + exp fusion."""
    @mc.fused
    def gemm_exp(A, B):
        return mc.exp(A @ B)

    A = np.random.randn(M, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N).astype(np.float32) * 0.1
    result = gemm_exp(A, B)

    expected = np.exp(A @ B)
    err = np.max(np.abs(result - expected))
    tol = 1e-3
    status = "PASS" if err < tol else "FAIL"
    print(f"  gemm_exp ({M}x{K}x{N}): max_err={err:.6f} [{status}]")
    assert err < tol, f"gemm_exp failed: {err}"

if __name__ == "__main__":
    print("=== Family A End-to-End GPU Tests ===")
    test_gemm_relu()
    test_gemm_gelu()
    test_gemm_exp()
    print("\nAll Family A GPU tests passed! ✓")
