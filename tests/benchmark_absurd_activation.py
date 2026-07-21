"""
Benchmark: MatcoreDSL Turing-Complete Fusion vs PyTorch Unfused Kernels
=======================================================================
MatcoreDSL fuses matmul + exotic epilogue chain into minimal kernel launches.
PyTorch (no torch.compile) must launch SEPARATE kernels for each op,
paying full GMEM round-trips each time.
"""
import sys, time, numpy as np
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc
from matcore.device_tensor import DeviceTensor

import torch
torch.backends.cudnn.benchmark = True

# ── helpers ──────────────────────────────────────────────────────────
def cuda_sync_time(fn, warmup=5, iters=20):
    """Time a CUDA function using events, excluding JIT."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))
    return np.median(times), np.mean(times)

# ── MatcoreDSL fused: matmul + relu + gelu + tanh + sigmoid ─────────
@mc.fused
def chaos_fused(A, B):
    x = A @ B
    return mc.sigmoid(mc.tanh(mc.gelu(mc.relu(x))))

# ── Benchmark ────────────────────────────────────────────────────────
sizes = [1024, 2048, 4096]

print("=" * 80)
print("  MatcoreDSL Fused Epilogue vs PyTorch Unfused Kernels (FP16)")
print("  Activation: sigmoid(tanh(gelu(relu(A @ B))))")
print("  PyTorch uses 5 separate kernel launches (cuBLAS + 4 elementwise)")
print("  MatcoreDSL fuses everything into minimal kernel launches")
print("=" * 80)

for N in sizes:
    print(f"\n{'─'*60}")
    print(f"  Size: {N}×{N}  (FP16, {2*N*N*N/1e12:.3f} TFLOP matmul)")
    print(f"{'─'*60}")

    # ── PyTorch reference (unfused, 5 separate kernels) ──────────
    tA = torch.randn(N, N, dtype=torch.float16, device="cuda")
    tB = torch.randn(N, N, dtype=torch.float16, device="cuda")

    def pytorch_unfused():
        x = torch.mm(tA, tB)       # kernel 1: cuBLAS
        x = torch.relu(x)          # kernel 2: elementwise
        x = torch.nn.functional.gelu(x)  # kernel 3: elementwise
        x = torch.tanh(x)          # kernel 4: elementwise
        x = torch.sigmoid(x)       # kernel 5: elementwise
        return x

    pt_med, pt_avg = cuda_sync_time(pytorch_unfused)

    # ── MatcoreDSL fused (JIT warmup excluded) ───────────────────
    npA = tA.cpu().numpy()
    npB = tB.cpu().numpy()
    dA = mc.to_device(npA)
    dB = mc.to_device(npB)

    # JIT warmup (excluded from timing)
    _ = chaos_fused(dA, dB)
    torch.cuda.synchronize()

    def matcore_fused():
        return chaos_fused(dA, dB)

    mc_med, mc_avg = cuda_sync_time(matcore_fused)

    # ── Results ──────────────────────────────────────────────────
    flops = 2 * N * N * N
    pt_tflops = flops / (pt_med * 1e-3) / 1e12
    mc_tflops = flops / (mc_med * 1e-3) / 1e12
    speedup = pt_med / mc_med

    print(f"  PyTorch (5 kernels):  {pt_med:7.2f} ms  ({pt_tflops:5.1f} TFLOPS)")
    print(f"  MatcoreDSL (fused):   {mc_med:7.2f} ms  ({mc_tflops:5.1f} TFLOPS)")
    print(f"  Speedup:              {speedup:.2f}x {'🏆 MatcoreDSL' if speedup > 1 else '⚠️  PyTorch'}")

print(f"\n{'='*80}")
print("  NOTE: PyTorch pays 4 extra GMEM round-trips (read+write each op).")
print("  MatcoreDSL fuses the entire epilogue chain into the matmul output path.")
print("=" * 80)
