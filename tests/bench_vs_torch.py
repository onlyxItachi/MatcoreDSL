"""Head-to-head: MatcoreDSL graph replay vs PyTorch cuBLAS."""
import numpy as np, time, sys, torch
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc

@mc.kernel
def matmul(A, B, C):
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)

WARMUP, ITERS = 10, 50
sizes = [32, 64, 128, 256, 512, 1024]

print("=" * 80)
print("MatcoreDSL vs PyTorch — REAL HEAD-TO-HEAD (RTX 4060 Laptop, fp16)")
print("=" * 80)
print()
hdr = f"{'Size':>6} | {'MC graph':>9} {'Torch':>9} | {'MC TF/s':>9} {'Torch TF/s':>10} | {'Gap':>5}"
print(hdr)
print("-" * 75)

for N in sizes:
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)
    flops = 2 * N**3

    # MatCore graph path
    dA = mc.to_device(A)
    dB = mc.to_device(B)
    dC = mc.to_device(C)
    try:
        gp = mc.create_plan(matmul, dA, dB, dC, target="nvidia-dgpu", graph_mode=True)
        mc.execute_plan(gp, dA, dB, dC)  # capture
        for _ in range(WARMUP):
            mc.execute_plan(gp, dA, dB, dC)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            mc.execute_plan(gp, dA, dB, dC)
        t_mc = (time.perf_counter() - t0) / ITERS * 1000
    except Exception as e:
        print(f"  {N:>4}² | MC ERROR: {e}")
        t_mc = None

    mc_tflops = flops / (t_mc/1000) / 1e12 if t_mc else 0

    # PyTorch
    tA = torch.from_numpy(A).cuda()
    tB = torch.from_numpy(B).cuda()
    for _ in range(WARMUP):
        tC = torch.matmul(tA, tB)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        tC = torch.matmul(tA, tB)
    torch.cuda.synchronize()
    t_pt = (time.perf_counter() - t0) / ITERS * 1000
    pt_tflops = flops / (t_pt/1000) / 1e12

    if t_mc:
        gap = pt_tflops / mc_tflops if mc_tflops > 0 else 999
        print(f"  {N:>4}² | {t_mc:>7.3f}ms {t_pt:>7.3f}ms | {mc_tflops:>8.4f} {pt_tflops:>9.4f}  | {gap:>4.1f}x")

print("=" * 80)
print()
print("MatCore uses mma.sync tensor cores for fp16. The remaining gap is")
print("tiling strategy: MatCore tiles 16x8 (1 warp), cuBLAS tiles 128x128+")
print("with multi-warp cooperation and shared-memory double-buffering.")
