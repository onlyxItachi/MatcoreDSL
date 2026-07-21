"""Fair fusion benchmark: large sizes, JIT excluded, GPU-resident PyTorch tensors.

Key differences from bench_fusion_suite.py:
- Much larger matrix sizes to stress compute (1024, 2048, 4096)
- JIT compilation excluded: first call compiles, timing starts after warmup
- PyTorch uses pre-allocated CUDA tensors (no CPU→GPU copy in timing loop)
- CUDA events for precise GPU timing (not wall-clock)
- MatcoreDSL still pays CPU→GPU copy per call (inherent to current API)
  so we also measure a "kernel-only" estimate by subtracting transfer time
"""
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matcore as mc
import torch
import torch.nn.functional as F

WARMUP = 10
ITERS = 50
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clear_cache():
    cache = os.path.join(REPO_ROOT, ".matcore_cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache)


def torch_gpu_time(fn, warmup=WARMUP, iters=ITERS):
    """Time a PyTorch GPU function using CUDA events (excludes launch overhead)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def matcore_time(fn, *args, warmup=WARMUP, iters=ITERS):
    """Time MatcoreDSL: warmup absorbs JIT, then measure execution only."""
    # Warmup (absorbs JIT compilation)
    for _ in range(warmup):
        fn(*args)

    # Timed iterations (JIT cached, measures execution + host→device copy)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters, out


# ─── Family A: GEMM + epilogue ───────────────────────────────────────────────

def make_family_a(op, tag):
    if op == "relu":
        def kernel(A, B): return mc.relu(A @ B)
    elif op == "gelu":
        def kernel(A, B): return mc.gelu(A @ B)
    elif op == "exp":
        def kernel(A, B): return mc.exp(A @ B)
    else:
        raise ValueError(op)
    kernel.__name__ = f"gemm_{op}_{tag}"
    return mc.fused(kernel)


def run_family_a():
    print("\n━━━ Family A: GEMM + Epilogue ━━━")
    print(f"{'Op':<5}{'Size':<16}{'MatcoreDSL':>12}{'PyTorch':>12}{'Speedup':>10}{'GFLOPS/s':>10}{'max_err':>10}")
    print("─" * 75)

    sizes = [(256, 128, 256), (512, 256, 512), (1024, 512, 1024),
             (2048, 1024, 2048)]
    ops = ["relu", "gelu", "exp"]
    speedups = []

    clear_cache()

    for op in ops:
        for M, K, N in sizes:
            scale = 0.1 if op == "exp" else 1.0
            A_np = (np.random.randn(M, K).astype(np.float32) * scale)
            B_np = (np.random.randn(K, N).astype(np.float32) * scale)

            A_t = torch.from_numpy(A_np).cuda()
            B_t = torch.from_numpy(B_np).cuda()

            # NumPy reference
            C = A_np @ B_np
            if op == "relu":
                expected = np.maximum(C, 0.0)
                torch_fn = lambda: torch.relu(A_t @ B_t)
            elif op == "gelu":
                expected = 0.5 * C * (1 + np.tanh(np.sqrt(2/np.pi) * (C + 0.044715 * C**3)))
                torch_fn = lambda: F.gelu(A_t @ B_t)
            else:
                expected = np.exp(C)
                torch_fn = lambda: torch.exp(A_t @ B_t)

            # PyTorch (GPU-resident, CUDA events)
            torch_ms = torch_gpu_time(torch_fn)

            # MatcoreDSL
            flops = 2.0 * M * K * N + M * N  # matmul + elementwise
            mc_ms = "ERR"
            speedup = "N/A"
            gflops = "N/A"
            max_err = "N/A"

            try:
                fused = make_family_a(op, f"{M}_{K}_{N}")
                mc_ms, out = matcore_time(fused, A_np, B_np)
                max_err = float(np.max(np.abs(out - expected)))
                speedup = torch_ms / mc_ms
                gflops = flops / (mc_ms / 1000) / 1e9
                speedups.append(speedup)
            except Exception as e:
                print(f"  [ERROR] {op} {M}x{K}x{N}: {e}")
                continue

            size_str = f"{M}x{K}x{N}"
            sp_str = f"{speedup:.2f}x" if isinstance(speedup, float) else speedup
            gf_str = f"{gflops:.1f}" if isinstance(gflops, float) else gflops
            mc_str = f"{mc_ms:.3f}" if isinstance(mc_ms, float) else mc_ms
            print(f"{op:<5}{size_str:<16}{mc_str:>12}{torch_ms:>12.3f}{sp_str:>10}{gf_str:>10}{max_err:>10.6f}")

    return speedups


# ─── Family B: tile chain ────────────────────────────────────────────────────

def make_family_b(tag):
    def kernel(A, B, W): return mc.relu(A @ B) @ W
    kernel.__name__ = f"chain_relu_{tag}"
    return mc.fused(kernel)


def run_family_b():
    print("\n━━━ Family B: Tile Chain (matmul→relu→matmul) ━━━")
    print(f"{'Size':<20}{'MatcoreDSL':>12}{'PyTorch':>12}{'Speedup':>10}{'max_err':>10}")
    print("─" * 64)

    sizes = [(64, 32, 64, 48), (128, 64, 128, 96), (256, 128, 256, 192),
             (512, 256, 512, 384)]
    speedups = []

    clear_cache()

    for M, K, N1, N2 in sizes:
        A_np = np.random.randn(M, K).astype(np.float32) * 0.1
        B_np = np.random.randn(K, N1).astype(np.float32) * 0.1
        W_np = np.random.randn(N1, N2).astype(np.float32) * 0.1

        A_t = torch.from_numpy(A_np).cuda()
        B_t = torch.from_numpy(B_np).cuda()
        W_t = torch.from_numpy(W_np).cuda()

        expected = np.maximum(A_np @ B_np, 0.0) @ W_np

        # PyTorch eager: two cuBLAS calls + relu — materialize intermediate in VRAM
        torch_fn = lambda: torch.relu(A_t @ B_t) @ W_t
        torch_ms = torch_gpu_time(torch_fn)

        mc_ms = "ERR"
        speedup = "N/A"
        max_err = "N/A"

        try:
            fused = make_family_b(f"{M}_{K}_{N1}_{N2}")
            mc_ms, out = matcore_time(fused, A_np, B_np, W_np)
            max_err = float(np.max(np.abs(out - expected)))
            speedup = torch_ms / mc_ms
            speedups.append(speedup)
        except Exception as e:
            print(f"  [ERROR] {M}x{K}x{N1}x{N2}: {e}")
            continue

        size_str = f"{M}x{K}x{N1}x{N2}"
        sp_str = f"{speedup:.2f}x" if isinstance(speedup, float) else speedup
        mc_str = f"{mc_ms:.3f}" if isinstance(mc_ms, float) else mc_ms
        print(f"{size_str:<20}{mc_str:>12}{torch_ms:>12.3f}{sp_str:>10}{max_err:>10.6f}")

    return speedups


# ─── Family C: attention ─────────────────────────────────────────────────────

def make_family_c(tag):
    def kernel(Q, K, V): return mc.softmax(Q @ K.T) @ V
    kernel.__name__ = f"attention_{tag}"
    return mc.fused(kernel)


def run_family_c():
    print("\n━━━ Family C: Attention softmax(Q·Kᵀ)·V ━━━")
    print(f"{'Size':<16}{'MatcoreDSL':>12}{'PyTorch':>12}{'Torch SDPA':>12}{'vs Eager':>10}{'vs SDPA':>10}{'max_err':>10}")
    print("─" * 82)

    sizes = [(32, 32, 16), (64, 64, 32), (128, 128, 64), (256, 256, 128)]
    speedups_eager = []
    speedups_sdpa = []

    clear_cache()

    for M, N, D in sizes:
        Q_np = np.random.randn(M, D).astype(np.float32) * 0.1
        K_np = np.random.randn(N, D).astype(np.float32) * 0.1
        V_np = np.random.randn(N, D).astype(np.float32) * 0.1

        Q_t = torch.from_numpy(Q_np).cuda()
        K_t = torch.from_numpy(K_np).cuda()
        V_t = torch.from_numpy(V_np).cuda()

        # NumPy reference
        scores = Q_np @ K_np.T
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        expected = probs @ V_np

        # PyTorch naive eager
        torch_eager_fn = lambda: torch.softmax(Q_t @ K_t.T, dim=-1) @ V_t
        torch_eager_ms = torch_gpu_time(torch_eager_fn)

        # PyTorch SDPA (needs batch dims: 1×1×M×D)
        Q_sdpa = Q_t.unsqueeze(0).unsqueeze(0)  # (1,1,M,D)
        K_sdpa = K_t.unsqueeze(0).unsqueeze(0)
        V_sdpa = V_t.unsqueeze(0).unsqueeze(0)
        torch_sdpa_fn = lambda: F.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa, scale=1.0)
        torch_sdpa_ms = torch_gpu_time(torch_sdpa_fn)

        mc_ms = "ERR"
        sp_eager = "N/A"
        sp_sdpa = "N/A"
        max_err = "N/A"

        try:
            fused = make_family_c(f"{M}_{N}_{D}")
            mc_ms, out = matcore_time(fused, Q_np, K_np, V_np)
            max_err = float(np.max(np.abs(out - expected)))
            sp_eager = torch_eager_ms / mc_ms
            sp_sdpa = torch_sdpa_ms / mc_ms
            speedups_eager.append(sp_eager)
            speedups_sdpa.append(sp_sdpa)
        except Exception as e:
            print(f"  [ERROR] {M}x{N}x{D}: {e}")
            continue

        size_str = f"{M}x{N}x{D}"
        mc_str = f"{mc_ms:.3f}" if isinstance(mc_ms, float) else mc_ms
        spe = f"{sp_eager:.2f}x" if isinstance(sp_eager, float) else sp_eager
        sps = f"{sp_sdpa:.2f}x" if isinstance(sp_sdpa, float) else sp_sdpa
        print(f"{size_str:<16}{mc_str:>12}{torch_eager_ms:>12.3f}{torch_sdpa_ms:>12.3f}{spe:>10}{sps:>10}{max_err:>10.6f}")

    return speedups_eager, speedups_sdpa


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 82)
    print("  MatcoreDSL Fusion Benchmark — Fair Comparison")
    print(f"  RTX 4060 Laptop (sm_89), float32")
    print(f"  Warmup={WARMUP} (absorbs JIT), Iters={ITERS}")
    print(f"  PyTorch: CUDA events timing, GPU-resident tensors")
    print(f"  MatcoreDSL: wall-clock (includes host→device copy per call)")
    print("=" * 82)

    a_speedups = run_family_a()
    b_speedups = run_family_b()
    c_eager, c_sdpa = run_family_c()

    print("\n" + "=" * 82)
    print("  Summary")
    print("=" * 82)
    if a_speedups:
        print(f"  Family A avg speedup vs PyTorch eager: {sum(a_speedups)/len(a_speedups):.2f}x  (n={len(a_speedups)})")
    if b_speedups:
        print(f"  Family B avg speedup vs PyTorch eager: {sum(b_speedups)/len(b_speedups):.2f}x  (n={len(b_speedups)})")
    if c_eager:
        print(f"  Family C avg speedup vs PyTorch eager: {sum(c_eager)/len(c_eager):.2f}x  (n={len(c_eager)})")
    if c_sdpa:
        print(f"  Family C avg speedup vs PyTorch SDPA:  {sum(c_sdpa)/len(c_sdpa):.2f}x  (n={len(c_sdpa)})")

    print()
    print("  Note: MatcoreDSL times include host→device memcpy per call.")
    print("  PyTorch times are GPU-only (CUDA events, data pre-staged on GPU).")
    print("  A true apples-to-apples comparison needs device-resident tensors")
    print("  in MatcoreDSL (planned for next phase).")
    print("=" * 82)
