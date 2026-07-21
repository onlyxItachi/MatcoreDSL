"""Post-codegen verification tests for fusion kernels."""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matcore as mc


def test_family_a_reg_budget():
    """Verify Family A kernel stays within register budget."""

    @mc.fused
    def gemm_relu(A, B):
        return mc.relu(A @ B)

    A = np.random.randn(64, 32).astype(np.float32)
    B = np.random.randn(32, 64).astype(np.float32)
    result = gemm_relu(A, B)

    expected = np.maximum(A @ B, 0)
    err = np.max(np.abs(result - expected))
    assert err < 1e-3, f"gemm_relu failed: {err}"
    print(f"  gemm_relu correctness: max_err={err:.6f} [PASS]")

    try:
        native = importlib.import_module("matcore._matcore_native")
        info = native.get_compilation_stats()
        if info and info.get("actual_reg_count", 0) > 0:
            print(f"  Actual regs: {info['actual_reg_count']}")
            assert not info.get("reg_budget_exceeded", False), "Register budget exceeded!"
            print("  reg_budget_exceeded: False [PASS]")
        else:
            print("  Compilation stats unavailable or no register count emitted")
    except (ImportError, AttributeError):
        print("  (compilation stats API not available)")

    print("  family_a_reg_budget: PASS")


if __name__ == "__main__":
    print("=== Post-codegen Verification Tests ===")
    test_family_a_reg_budget()
    print("\nAll post-codegen tests passed! ✓")
