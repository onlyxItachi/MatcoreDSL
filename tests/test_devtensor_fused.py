import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matcore as mc


def test_fused_devtensor():
    @mc.fused
    def gemm_relu(A, B):
        return mc.relu(A @ B)

    M, K, N = 64, 32, 64
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)

    # Test with numpy (existing path)
    result_np = gemm_relu(A_np, B_np)
    expected = np.maximum(A_np @ B_np, 0)
    assert np.max(np.abs(result_np - expected)) < 1e-3

    # Test with DeviceTensor
    dA = mc.to_device(A_np)
    dB = mc.to_device(B_np)
    result_dev = gemm_relu(dA, dB)

    # result_dev should be a DeviceTensor
    assert hasattr(result_dev, '_matcore_device_tensor'), "Result should be DeviceTensor"
    result_host = result_dev.to_host()
    err = np.max(np.abs(result_host - expected))
    print(f"DeviceTensor fused: max_err={err:.6f} {'PASS' if err < 1e-3 else 'FAIL'}")
    assert err < 1e-3


if __name__ == "__main__":
    test_fused_devtensor()
    print("DeviceTensor fused test PASSED")
