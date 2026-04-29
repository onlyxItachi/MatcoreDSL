"""Extended power-of-2 benchmark: 128² through 65536² (or until OOM)."""
import numpy as np
import time, sys, traceback
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

@mc.fused
def fused_matmul(A, B):
    return A @ B

def bench_matcore_fused(N, warmup=5, iters=20):
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    dA = mc.to_device(A)
    dB = mc.to_device(B)
    # warmup (includes JIT)
    for _ in range(warmup):
        dC = fused_matmul(dA, dB)
    if HAS_TORCH:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dC = fused_matmul(dA, dB)
    if HAS_TORCH:
        torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) / iters * 1000
    flops = 2 * N**3
    tflops = flops / (t_ms / 1000) / 1e12
    return t_ms, tflops

def bench_pytorch(N, warmup=5, iters=20):
    if not HAS_TORCH:
        return None, None
    A = torch.randn(N, N, dtype=torch.float16, device='cuda')
    B = torch.randn(N, N, dtype=torch.float16, device='cuda')
    for _ in range(warmup):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) / iters * 1000
    flops = 2 * N**3
    tflops = flops / (t_ms / 1000) / 1e12
    return t_ms, tflops

# Powers of 2 from 128 to 65536
sizes = [2**k for k in range(7, 17)]  # 128, 256, 512, ..., 65536

print("=" * 82)
print("Extended Power-of-2 Benchmark: MatcoreDSL (fused) vs PyTorch")
print("=" * 82)
print(f"{'Size':>8} | {'MC time':>10} {'MC TFLOP/s':>11} | {'PT time':>10} {'PT TFLOP/s':>11} | {'gap':>6}")
print(f"{'':>8} | {'(ms)':>10} {'':>11} | {'(ms)':>10} {'':>11} | {'':>6}")
print("-" * 82)

for N in sizes:
    mem_gb = 3 * N * N * 2 / 1e9  # 3 matrices × FP16
    sys.stdout.write(f"{N:>6}² | ")
    sys.stdout.flush()

    # MatCore
    mc_ms, mc_tf = None, None
    try:
        mc_ms, mc_tf = bench_matcore_fused(N, warmup=3, iters=max(3, min(20, 50000 // max(1, N))))
    except Exception as e:
        err = str(e)[:40]
        pass

    # PyTorch
    pt_ms, pt_tf = None, None
    try:
        pt_ms, pt_tf = bench_pytorch(N, warmup=3, iters=max(3, min(20, 50000 // max(1, N))))
    except Exception as e:
        pass

    mc_ms_s = f"{mc_ms:10.3f}" if mc_ms else "       OOM"
    mc_tf_s = f"{mc_tf:11.4f}" if mc_tf else "        N/A"
    pt_ms_s = f"{pt_ms:10.3f}" if pt_ms else "       OOM"
    pt_tf_s = f"{pt_tf:11.4f}" if pt_tf else "        N/A"
    if mc_tf and pt_tf:
        gap = f"{mc_tf/pt_tf:5.2f}x"
    else:
        gap = "  N/A "
    print(f"{mc_ms_s} {mc_tf_s} | {pt_ms_s} {pt_tf_s} | {gap}")

    # Clean up GPU memory
    if HAS_TORCH:
        torch.cuda.empty_cache()
    import gc; gc.collect()

    # If both OOM, stop
    if mc_ms is None and pt_ms is None:
        print(f"  (stopping — both OOM at {N}², needs ~{mem_gb:.1f} GB)")
        break

print("=" * 82)
