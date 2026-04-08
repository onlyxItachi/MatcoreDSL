import numpy as np, sys
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc

@mc.kernel
def matmul(A, B, C):
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)

A = np.random.randn(64,64).astype(np.float16)
B = np.random.randn(64,64).astype(np.float16)
C = np.zeros((64,64), dtype=np.float16)
mc.launch(matmul, A, B, C, target="nvidia-dgpu")
print(f"OK C[0,0]={C[0,0]:.4f}")
