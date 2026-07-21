"""Test sub-tiling correctness at various matrix sizes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matcore as mc

@mc.kernel
def matmul(A, B, C):
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)

TARGET = "nvidia-dgpu:sm_89"

for N in [16, 32, 48, 64, 128, 256]:
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)
    try:
        mc.launch(matmul, A, B, C, target=TARGET)
        expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        atol = np.max(np.abs(C.astype(np.float32) - expected.astype(np.float32)))
        status = "PASS" if atol < 1.0 else "FAIL"
        print(f"{N}x{N}: atol={atol:.6f} {status}")
    except Exception as e:
        print(f"{N}x{N}: ERROR - {e}")
