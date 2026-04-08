"""Dump IR snapshots for 256x256 fp16 matmul to check mma.sync."""
import numpy as np, sys, os
os.environ["MATCORE_DEBUG"] = "1"
os.environ["MATCORE_DEBUG_DIR"] = "/tmp/matcore_ir_dump"
os.environ["MATCORE_DEBUG_SESSION"] = "mma_check"
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc

@mc.kernel
def matmul(A, B, C):
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)

N = 256
A = np.random.randn(N, N).astype(np.float16)
B = np.random.randn(N, N).astype(np.float16)
C = np.zeros((N, N), dtype=np.float16)
mc.launch(matmul, A, B, C, target="nvidia-dgpu")
print(f"Done. C[0,0]={C[0,0]:.4f}")
print("IR snapshots written to /tmp/matcore_ir_dump/mma_check/")
