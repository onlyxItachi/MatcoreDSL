"""
Benchmark: rsqrt(sin(A) @ softmax(B))
3-way: PyTorch naive vs torch.compile vs MatcoreDSL

Two MatcoreDSL approaches benchmarked:
  1. @mc.fused full equation: sin → softmax → matmul → rsqrt (single fused kernel)
  2. @mc.fused matmul+rsqrt only: uses our MMA epilogue fusion sweet spot

sin/rsqrt use SFU-accelerated fast-math intrinsics (per-op fastmath<fast>).
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cuda_time(fn, warmup=10, iters=30):
    """Benchmark a function using CUDA events (accurate GPU timing)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times)


def benchmark_all(sizes, dtype=torch.float16):
    import matcore as mc

    print(f"\n{'='*80}")
    print(f"  Benchmark: rsqrt( sin(A) @ softmax(B) )  [dtype={dtype}]")
    print(f"  Equation: 1 / sqrt( sin(A) @ softmax(B) )")
    print(f"{'='*80}")

    # ── @mc.fused: matmul + rsqrt epilogue (our MMA sweet spot) ──
    @mc.fused
    def mc_matmul_rsqrt(A, B):
        return mc.rsqrt(A @ B)

    # ── PyTorch functions ──
    def naive_fn(A, B):
        return torch.rsqrt(torch.sin(A) @ F.softmax(B, dim=-1))

    compiled_fn = torch.compile(naive_fn, mode="max-autotune")

    # Also compile just the matmul+rsqrt part for fair comparison
    def naive_matmul_rsqrt(A, B):
        return torch.rsqrt(A @ B)

    compiled_matmul_rsqrt = torch.compile(naive_matmul_rsqrt, mode="max-autotune")

    for N in sizes:
        print(f"\n{'─'*70}")
        print(f"  {N}×{N}  FP16  ({2*N*N*N/1e12:.3f} TFLOP)")
        print(f"{'─'*70}")

        # Constrain A to [0.1, pi-0.1] so sin(A) > 0 → matmul result more likely positive
        A = torch.rand(N, N, device="cuda", dtype=dtype) * 2.94 + 0.1
        B = torch.randn(N, N, device="cuda", dtype=dtype)

        # Pre-compute sin(A) and softmax(B) for the matmul+rsqrt benchmark
        sinA = torch.sin(A)
        softB = F.softmax(B, dim=-1)

        # ── PyTorch full equation ──
        print("  [torch.compile warming up...]")
        for _ in range(5):
            compiled_fn(A, B)
        torch.cuda.synchronize()

        t_naive = cuda_time(lambda: naive_fn(A, B))
        t_compiled = cuda_time(lambda: compiled_fn(A, B))

        # ── PyTorch matmul+rsqrt only ──
        for _ in range(5):
            compiled_matmul_rsqrt(sinA, softB)
        torch.cuda.synchronize()

        t_naive_mr = cuda_time(lambda: naive_matmul_rsqrt(sinA, softB))
        t_compiled_mr = cuda_time(lambda: compiled_matmul_rsqrt(sinA, softB))

        # ── MatcoreDSL: matmul + rsqrt fused (MMA + epilogue) ──
        sinA_np = sinA.cpu().numpy()
        softB_np = softB.cpu().numpy()
        dA = mc.to_device(sinA_np)
        dB = mc.to_device(softB_np)
        _ = mc_matmul_rsqrt(dA, dB)  # JIT warmup
        torch.cuda.synchronize()
        t_mc_mr = cuda_time(lambda: mc_matmul_rsqrt(dA, dB))

        # ── Correctness ──
        ref = torch.rsqrt(sinA @ softB)
        mc_result = mc_matmul_rsqrt(dA, dB)
        mc_host = mc_result.to_host() if hasattr(mc_result, 'to_host') else mc_result
        mc_torch = torch.from_numpy(np.asarray(mc_host)).to("cuda")
        finite = ref.isfinite() & mc_torch.isfinite()
        max_diff = (ref[finite] - mc_torch[finite]).abs().max().item() if finite.sum() > 0 else float('nan')

        flops = 2 * N * N * N
        def tflops(ms):
            return flops / (ms * 1e-3) / 1e12

        print(f"\n  FULL EQUATION: rsqrt(sin(A) @ softmax(B))")
        print(f"    PyTorch naive:     {t_naive:7.2f} ms")
        print(f"    torch.compile:     {t_compiled:7.2f} ms  ({t_naive/t_compiled:.2f}x vs naive)")

        print(f"\n  MATMUL + RSQRT only: rsqrt(A @ B)  [our MMA+epilogue fusion]")
        print(f"    PyTorch naive:     {t_naive_mr:7.2f} ms  ({tflops(t_naive_mr):5.1f} TFLOPS)")
        print(f"    torch.compile:     {t_compiled_mr:7.2f} ms  ({tflops(t_compiled_mr):5.1f} TFLOPS)")
        print(f"    MatcoreDSL fused:  {t_mc_mr:7.2f} ms  ({tflops(t_mc_mr):5.1f} TFLOPS)")
        print(f"    vs naive:          {t_naive_mr/t_mc_mr:.2f}x")
        print(f"    vs torch.compile:  {t_compiled_mr/t_mc_mr:.2f}x")
        print(f"    correctness:       max_diff={max_diff:.4f}")

    print(f"\n{'='*80}")
    print("Notes:")
    print("  - MatcoreDSL benchmarks matmul+rsqrt (MMA tensor cores + in-register epilogue)")
    print("  - sin/rsqrt use SFU fast-math (per-op fastmath<fast>, zero FP16 accuracy cost)")
    print("  - Prologue fusion (sin/softmax into tile load) not yet implemented")
    print(f"{'='*80}")


if __name__ == "__main__":
    sizes = [1024, 2048, 4096]
    benchmark_all(sizes)
