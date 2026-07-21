"""Minimal graph capture test."""
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

N = 16
A = np.random.randn(N, N).astype(np.float16)
B = np.random.randn(N, N).astype(np.float16)
C = np.zeros((N, N), dtype=np.float16)

dA = mc.to_device(A)
dB = mc.to_device(B)
dC = mc.to_device(C)

# Normal plan first
plan_normal = mc.create_plan(matmul, dA, dB, dC, target="nvidia-dgpu:sm_89")
mc.execute_plan(plan_normal, dA, dB, dC)
result = dC.to_host()
print(f"Normal plan OK: sum={np.sum(result):.4f}")

# Graph plan
print("Creating graph plan...")
plan_graph = mc.create_plan(matmul, dA, dB, dC, target="nvidia-dgpu:sm_89", graph_mode=True)
print("Executing (capture)...")
dC.zero_()
mc.execute_plan(plan_graph, dA, dB, dC)
result2 = dC.to_host()
print(f"Graph capture OK: sum={np.sum(result2):.4f}")

# Graph replay
dC.zero_()
mc.execute_plan(plan_graph, dA, dB, dC)
result3 = dC.to_host()
print(f"Graph replay OK: sum={np.sum(result3):.4f}")

dA.free(); dB.free(); dC.free()
print("All graph tests passed!")
