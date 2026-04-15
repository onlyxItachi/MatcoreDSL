import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import numpy as np
import torch
import torch.nn.functional as F
import matcore as mc


WARMUP = 5
ITERS = 20
SIZES = [
    (16, 16, 8),
    (32, 32, 16),
    (64, 64, 32),
    (128, 128, 64),
]


@mc.fused
def matcore_attention(Q, K, V):
    return mc.softmax(Q @ K.T) @ V


def numpy_attention(Q, K, V):
    scores = Q @ K.T
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs @ V


def time_matcore(Q, K, V):
    for _ in range(WARMUP):
        matcore_attention(Q, K, V)
    t0 = time.perf_counter()
    out = None
    for _ in range(ITERS):
        out = matcore_attention(Q, K, V)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / ITERS, out


def time_torch_naive(Q_t, K_t, V_t):
    for _ in range(WARMUP):
        _ = torch.softmax(Q_t @ K_t.T, dim=-1) @ V_t
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(ITERS):
        out = torch.softmax(Q_t @ K_t.T, dim=-1) @ V_t
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / ITERS, out


def time_torch_sdpa(Q_t, K_t, V_t):
    q = Q_t.unsqueeze(0)
    k = K_t.unsqueeze(0)
    v = V_t.unsqueeze(0)
    for _ in range(WARMUP):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(ITERS):
        out = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / ITERS, out.squeeze(0)


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float, np.floating)) else str(v)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.")
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(repo_root, ".matcore_cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    print("=== MatcoreDSL Fused Attention vs PyTorch ===")
    print("Size (MxNxD) | MatcoreDSL (ms) | Torch Naive (ms) | Torch SDPA (ms) | max_err")

    for M, N, D in SIZES:
        Q = (np.random.randn(M, D).astype(np.float32) * 0.1)
        K = (np.random.randn(N, D).astype(np.float32) * 0.1)
        V = (np.random.randn(N, D).astype(np.float32) * 0.1)

        expected = numpy_attention(Q, K, V)

        Q_t = torch.from_numpy(Q).to("cuda", dtype=torch.float32)
        K_t = torch.from_numpy(K).to("cuda", dtype=torch.float32)
        V_t = torch.from_numpy(V).to("cuda", dtype=torch.float32)

        torch_naive_ms, _ = time_torch_naive(Q_t, K_t, V_t)
        torch_sdpa_ms, _ = time_torch_sdpa(Q_t, K_t, V_t)

        matcore_ms = "ERR"
        max_err = "N/A"

        try:
            matcore_ms, matcore_out = time_matcore(Q, K, V)
            max_err = float(np.max(np.abs(matcore_out - expected)))
        except Exception as e:
            print(f"{M}x{N}x{D}: MatcoreDSL failed: {e}")

        size_str = f"{M}x{N}x{D}"
        print(
            f"{size_str:12} | {fmt(matcore_ms):14} | {fmt(torch_naive_ms):16} | "
            f"{fmt(torch_sdpa_ms):15} | {fmt(max_err)}"
        )


if __name__ == "__main__":
    main()
