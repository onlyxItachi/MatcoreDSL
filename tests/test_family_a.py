"""End-to-end Family A fusion tests: GEMM + pointwise epilogue on GPU."""
import importlib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matcore as mc


def _assert_family_a_no_fallback_route():
    """Assert the last native compile used the fused route when stats are exposed."""
    try:
        native = importlib.import_module("matcore._matcore_native")
    except ImportError:
        print("  route stats unavailable")
        return

    get_stats = getattr(native, "get_compilation_stats", None)
    if not callable(get_stats):
        print("  route stats unavailable")
        return

    info = get_stats()
    route = str(info.get("route", "")) if info else ""
    if not route:
        print("  route stats unavailable")
        return

    route_lower = route.lower()
    assert "fused" in route_lower, f"expected fused Family A route, got {route!r}"
    assert "fallback" not in route_lower, f"unexpected fallback route: {route!r}"
    print(f"  route: {route} [PASS]")


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
    _assert_family_a_no_fallback_route()

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
    _assert_family_a_no_fallback_route()

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
    _assert_family_a_no_fallback_route()

def test_gemm_add(M=64, N=64, K=32):
    """GEMM + dense bias add fusion."""
    @mc.fused
    def gemm_add(A, B, bias):
        return (A @ B) + bias

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(M, N).astype(np.float32)
    result = gemm_add(A, B, bias)

    expected = (A @ B) + bias
    err = np.max(np.abs(result - expected))
    tol = 1e-3
    status = "PASS" if err < tol else "FAIL"
    print(f"  gemm_add ({M}x{K}x{N}): max_err={err:.6f} [{status}]")
    assert err < tol, f"gemm_add failed: {err}"
    _assert_family_a_no_fallback_route()

def test_gemm_add_relu(M=64, N=64, K=32):
    """GEMM + dense bias add + ReLU fusion."""
    @mc.fused
    def gemm_add_relu(A, B, bias):
        return mc.relu((A @ B) + bias)

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(M, N).astype(np.float32)
    result = gemm_add_relu(A, B, bias)

    expected = np.maximum((A @ B) + bias, 0)
    err = np.max(np.abs(result - expected))
    tol = 1e-3
    status = "PASS" if err < tol else "FAIL"
    print(f"  gemm_add_relu ({M}x{K}x{N}): max_err={err:.6f} [{status}]")
    assert err < tol, f"gemm_add_relu failed: {err}"
    _assert_family_a_no_fallback_route()

def test_bias_minus_gemm(M=64, N=64, K=32):
    """Dense bias - GEMM fusion preserves operand order."""
    @mc.fused
    def bias_minus_gemm(A, B, bias):
        return bias - (A @ B)

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(M, N).astype(np.float32)
    result = bias_minus_gemm(A, B, bias)

    expected = bias - (A @ B)
    err = np.max(np.abs(result - expected))
    tol = 1e-3
    status = "PASS" if err < tol else "FAIL"
    print(f"  bias_minus_gemm ({M}x{K}x{N}): max_err={err:.6f} [{status}]")
    assert err < tol, f"bias_minus_gemm failed: {err}"
    _assert_family_a_no_fallback_route()

if __name__ == "__main__":
    print("=== Family A End-to-End GPU Tests ===")
    test_gemm_relu()
    test_gemm_gelu()
    test_gemm_exp()
    test_gemm_add()
    test_gemm_add_relu()
    test_bias_minus_gemm()
    print("\nAll Family A GPU tests passed! ✓")
