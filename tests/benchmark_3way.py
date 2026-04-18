"""3-way benchmark: MatcoreDSL fused vs PyTorch naive vs torch.compile (Triton)"""
import sys, numpy as np
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc
import torch
torch.backends.cudnn.benchmark = True

def cuda_time(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times)

# ── MatcoreDSL fused ──
@mc.fused
def chaos_fused(A, B):
    x = A @ B
    return mc.sigmoid(mc.tanh(mc.gelu(mc.relu(x))))

# ── torch.compile fused ──
@torch.compile(mode="max-autotune")
def pytorch_compiled(A, B):
    x = torch.mm(A, B)
    x = torch.relu(x)
    x = torch.nn.functional.gelu(x)
    x = torch.tanh(x)
    x = torch.sigmoid(x)
    return x

def pytorch_naive(A, B):
    x = torch.mm(A, B)
    x = torch.relu(x)
    x = torch.nn.functional.gelu(x)
    x = torch.tanh(x)
    x = torch.sigmoid(x)
    return x

print("=" * 80)
print("  3-Way Benchmark: sigmoid(tanh(gelu(relu(A @ B)))) — FP16")
print("  1) PyTorch naive    (cuBLAS + 4 separate kernels)")
print("  2) torch.compile    (cuBLAS + Triton fused epilogue)")
print("  3) MatcoreDSL       (MMA + fused epilogue)")
print("=" * 80)

for N in [1024, 2048, 4096]:
    print(f"\n{'─'*70}")
    print(f"  {N}×{N}  FP16  ({2*N*N*N/1e12:.3f} TFLOP)")
    print(f"{'─'*70}")

    tA = torch.randn(N, N, dtype=torch.float16, device="cuda")
    tB = torch.randn(N, N, dtype=torch.float16, device="cuda")

    # PyTorch naive
    t_naive = cuda_time(lambda: pytorch_naive(tA, tB))

    # torch.compile (Triton) — warmup includes JIT
    print("  [torch.compile warming up...]")
    for _ in range(5):
        pytorch_compiled(tA, tB)
    torch.cuda.synchronize()
    t_compiled = cuda_time(lambda: pytorch_compiled(tA, tB))

    # MatcoreDSL
    npA = tA.cpu().numpy()
    npB = tB.cpu().numpy()
    dA = mc.to_device(npA)
    dB = mc.to_device(npB)
    _ = chaos_fused(dA, dB)  # JIT warmup
    torch.cuda.synchronize()
    t_mc = cuda_time(lambda: chaos_fused(dA, dB))

    flops = 2 * N * N * N
    def tflops(ms): return flops / (ms * 1e-3) / 1e12

    print(f"  PyTorch naive:      {t_naive:7.2f} ms  ({tflops(t_naive):5.1f} TFLOPS)")
    print(f"  torch.compile:      {t_compiled:7.2f} ms  ({tflops(t_compiled):5.1f} TFLOPS)")
    print(f"  MatcoreDSL fused:   {t_mc:7.2f} ms  ({tflops(t_mc):5.1f} TFLOPS)")
    print(f"  vs naive:           {t_naive/t_mc:.2f}x")
    print(f"  vs torch.compile:   {t_compiled/t_mc:.2f}x")

print(f"\n{'='*80}")
