"""Test multi-warp path correctness for V4.
Sizes must be multiples of 64 with grid >= 24 to trigger multi-warp.
Minimum: 384x384 → (384/64)^2 = 36 blocks.
"""
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

# Multi-warp eligible sizes: multiples of 64 with (N/64)^2 >= 24
# 384: grid=36, 448: grid=49, 512: grid=64, 640: grid=100, 768: grid=144
sizes = [384, 448, 512, 640, 768]

for N in sizes:
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)
    
    try:
        mc.launch(matmul, A, B, C, target=TARGET)
        
        ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        max_err = np.max(np.abs(C.astype(np.float32) - ref.astype(np.float32)))
        # f16 tolerance scales with N (accumulation errors)
        tol = max(1.0, N * 0.005)
        ok = max_err < tol
        print(f"{N}x{N}: max_err={max_err:.4f} tol={tol:.1f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            diff = np.abs(C.astype(np.float32) - ref.astype(np.float32))
            print(f"  diff stats: mean={np.mean(diff):.4f} median={np.median(diff):.4f} p99={np.percentile(diff, 99):.4f}")
    except Exception as e:
        print(f"{N}x{N}: ERROR - {e}")
