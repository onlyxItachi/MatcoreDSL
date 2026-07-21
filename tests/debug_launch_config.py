"""Quick launch config debug for different sizes."""
import numpy as np, sys
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc

@mc.kernel
def matmul(A, B, C):
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)

for N in [32, 64, 128, 256, 512, 1024]:
    print(f"\n=== {N}x{N} fp16 ===", flush=True)
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)
    mc.launch(matmul, A, B, C, target="nvidia-dgpu")
    print(f"  Result OK: C[0,0]={C[0,0]:.4f}")
