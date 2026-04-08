"""Full benchmark: MatcoreDSL vs PyTorch across sizes.

Shows where graph mode shines and where we stand vs PyTorch.
"""
import numpy as np
import time
import sys
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc

# Try importing torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

@mc.kernel
def matmul(A, B, C):
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)

def bench_matcore(N, warmup=5, iters=50):
    """Benchmark all 3 MatCore paths."""
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)

    # --- Path 1: mc.launch() (full JIT every time) ---
    for _ in range(warmup):
        mc.launch(matmul, A, B, C, target="nvidia-dgpu")
    t0 = time.perf_counter()
    for _ in range(iters):
        mc.launch(matmul, A, B, C, target="nvidia-dgpu")
    t_launch = (time.perf_counter() - t0) / iters * 1000  # ms

    # --- Path 2: execute_plan() (cached plan, no graph) ---
    dA = mc.to_device(A)
    dB = mc.to_device(B)
    dC = mc.to_device(C)
    plan = mc.create_plan(matmul, dA, dB, dC, target="nvidia-dgpu")
    for _ in range(warmup):
        mc.execute_plan(plan, dA, dB, dC)
    t0 = time.perf_counter()
    for _ in range(iters):
        mc.execute_plan(plan, dA, dB, dC)
    t_plan = (time.perf_counter() - t0) / iters * 1000

    # --- Path 3: graph replay ---
    dA2 = mc.to_device(A)
    dB2 = mc.to_device(B)
    dC2 = mc.to_device(C)
    gplan = mc.create_plan(matmul, dA2, dB2, dC2, target="nvidia-dgpu", graph_mode=True)
    mc.execute_plan(gplan, dA2, dB2, dC2)  # capture
    for _ in range(warmup):
        mc.execute_plan(gplan, dA2, dB2, dC2)  # replay warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        mc.execute_plan(gplan, dA2, dB2, dC2)
    t_graph = (time.perf_counter() - t0) / iters * 1000

    # Compute TFLOP/s (2*N^3 FLOPs for matmul)
    flops = 2 * N**3
    tflops_launch = flops / (t_launch / 1000) / 1e12
    tflops_plan   = flops / (t_plan / 1000) / 1e12
    tflops_graph  = flops / (t_graph / 1000) / 1e12

    return t_launch, t_plan, t_graph, tflops_launch, tflops_plan, tflops_graph

def bench_pytorch(N, warmup=5, iters=50):
    """Benchmark PyTorch matmul (tensors already on GPU)."""
    if not HAS_TORCH:
        return None, None
    A = torch.randn(N, N, dtype=torch.float16, device='cuda')
    B = torch.randn(N, N, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Timed
    t0 = time.perf_counter()
    for _ in range(iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) / iters * 1000
    
    flops = 2 * N**3
    tflops = flops / (t_ms / 1000) / 1e12
    return t_ms, tflops

# ============================================================
print("=" * 80)
print("MatcoreDSL Full Performance Report")
print("=" * 80)
print()

sizes = [32, 64, 128, 256, 512, 1024]
results = []

for N in sizes:
    print(f"Benchmarking {N}x{N}...", flush=True)
    t_launch, t_plan, t_graph, tf_launch, tf_plan, tf_graph = bench_matcore(N)
    t_torch, tf_torch = bench_pytorch(N)
    results.append((N, t_launch, t_plan, t_graph, tf_launch, tf_plan, tf_graph, t_torch, tf_torch))

print()
print("=" * 80)
print(f"{'Size':>6} | {'launch':>8} {'plan':>8} {'graph':>8} {'PyTorch':>8} | {'graph':>8} {'PyTorch':>8} | {'ratio':>6}")
print(f"{'':>6} | {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} | {'TFLOP/s':>8} {'TFLOP/s':>8} | {'gap':>6}")
print("-" * 80)

for N, t_l, t_p, t_g, tf_l, tf_p, tf_g, t_t, tf_t in results:
    torch_str = f"{t_t:8.3f}" if t_t else "   N/A  "
    torch_tf  = f"{tf_t:8.4f}" if tf_t else "   N/A  "
    if tf_t and tf_g:
        gap = f"{tf_t/tf_g:5.1f}x"
    else:
        gap = "  N/A "
    print(f"{N:>5}² | {t_l:8.3f} {t_p:8.3f} {t_g:8.3f} {torch_str} | {tf_g:8.4f} {torch_tf} | {gap}")

print("=" * 80)
print()
print("Legend:")
print("  launch  = mc.launch() — full JIT dispatch every call")
print("  plan    = execute_plan() — cached compilation, device-resident tensors")  
print("  graph   = CUDA graph replay — captured stream, zero dispatch overhead")
print("  PyTorch = torch.matmul() on pre-allocated CUDA tensors")
print("  gap     = how many X PyTorch is faster than our best (graph)")
print()

# Summary
best = results[-1]  # largest size
N = best[0]
if best[7]:  # has torch
    print(f"At {N}x{N}: MatCore graph = {best[5]:.4f} TFLOP/s, PyTorch = {best[8]:.4f} TFLOP/s")
    print(f"  → PyTorch is {best[8]/best[5]:.1f}x faster at large sizes (kernel efficiency gap)")
    print(f"  → At small sizes, MatCore graph closes the dispatch overhead gap")
    
    # Find where we're closest
    min_gap = 999
    min_N = 0
    for N, t_l, t_p, t_g, tf_l, tf_p, tf_g, t_t, tf_t in results:
        if tf_t and tf_g:
            g = tf_t / tf_g
            if g < min_gap:
                min_gap = g
                min_N = N
    print(f"  → Closest gap: {min_gap:.1f}x at {min_N}x{min_N}")
