import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matcore as mc
import torch

WARMUP = 5
ITERS = 20
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clear_cache():
    cache = os.path.join(REPO_ROOT, ".matcore_cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache)


def numpy_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def numpy_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def time_matcore(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    t0 = time.perf_counter()
    out = None
    for _ in range(ITERS):
        out = fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / ITERS, out


def time_torch(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(ITERS):
        out = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / ITERS, out


def make_family_a_fused(op, tag):
    if op == "relu":
        def kernel(A, B):
            return mc.relu(A @ B)
    elif op == "gelu":
        def kernel(A, B):
            return mc.gelu(A @ B)
    elif op == "exp":
        def kernel(A, B):
            return mc.exp(A @ B)
    else:
        raise ValueError(f"Unsupported op: {op}")
    kernel.__name__ = f"gemm_{op}_{tag}"
    return mc.fused(kernel)


def make_family_b_fused(tag):
    def kernel(A, B, W):
        return mc.relu(A @ B) @ W
    kernel.__name__ = f"chain_relu_{tag}"
    return mc.fused(kernel)


def make_family_c_fused(tag):
    def kernel(Q, K, V):
        return mc.softmax(Q @ K.T) @ V
    kernel.__name__ = f"attention_{tag}"
    return mc.fused(kernel)


def fmt_time_ms(v):
    return f"{v:>12.3f}" if isinstance(v, (float, np.floating)) else f"{v:>12}"


def fmt_speedup(v):
    return f"{v:>6.2f}x" if isinstance(v, (float, np.floating)) else f"{v:>6}"


def fmt_err(v):
    return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)


def run_family_a():
    print("\n--- Family A: GEMM + Epilogue ---")
    print("Op   | Size         | MatcoreDSL (ms) | PyTorch (ms) | Speedup | max_err")

    sizes = [(64, 32, 64), (128, 64, 128), (256, 128, 256), (512, 256, 512)]
    ops = ["relu", "gelu", "exp"]
    speedups = []

    clear_cache()

    for op in ops:
        for M, K, N in sizes:
            scale = 0.1 if op == "exp" else 1.0
            A = (np.random.randn(M, K).astype(np.float32) * scale)
            B = (np.random.randn(K, N).astype(np.float32) * scale)

            A_t = torch.from_numpy(A).to("cuda", dtype=torch.float32)
            B_t = torch.from_numpy(B).to("cuda", dtype=torch.float32)

            if op == "relu":
                expected = np.maximum(A @ B, 0.0)
                torch_callable = lambda: torch.relu(A_t @ B_t)
            elif op == "gelu":
                C = A @ B
                expected = numpy_gelu(C)
                torch_callable = lambda: torch.nn.functional.gelu(A_t @ B_t)
            else:
                expected = np.exp(A @ B)
                torch_callable = lambda: torch.exp(A_t @ B_t)

            torch_ms, _ = time_torch(torch_callable)

            matcore_ms = "ERR"
            speedup = "N/A"
            max_err = "N/A"

            try:
                fused = make_family_a_fused(op, f"{M}_{K}_{N}")
                matcore_ms, out = time_matcore(fused, A, B)
                max_err = float(np.max(np.abs(out - expected)))
                if max_err > 1e-3:
                    print(f"  [warn] Family A {op} {M}x{K}x{N} max_err={max_err:.6f} > 1e-3")
                speedup = torch_ms / matcore_ms
                speedups.append(speedup)
            except Exception as e:
                print(f"  [error] Matcore Family A {op} {M}x{K}x{N} failed: {e}")

            size_str = f"{M}x{K}x{N}"
            print(
                f"{op:<4} | {size_str:<12} | {fmt_time_ms(matcore_ms)} |"
                f" {torch_ms:>11.3f} | {fmt_speedup(speedup)} | {fmt_err(max_err)}"
            )

    return speedups


def run_family_b():
    print("\n--- Family B: Tile Chain ---")
    print("Op   | Size            | MatcoreDSL (ms) | PyTorch (ms) | Speedup | max_err")

    sizes = [(32, 16, 32, 24), (64, 32, 64, 48), (128, 64, 128, 96)]
    speedups = []

    clear_cache()

    for M, K, N1, N2 in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N1).astype(np.float32)
        W = np.random.randn(N1, N2).astype(np.float32)

        expected = np.maximum(A @ B, 0.0) @ W

        A_t = torch.from_numpy(A).to("cuda", dtype=torch.float32)
        B_t = torch.from_numpy(B).to("cuda", dtype=torch.float32)
        W_t = torch.from_numpy(W).to("cuda", dtype=torch.float32)

        torch_ms, _ = time_torch(lambda: torch.relu(A_t @ B_t) @ W_t)

        matcore_ms = "ERR"
        speedup = "N/A"
        max_err = "N/A"

        try:
            fused = make_family_b_fused(f"{M}_{K}_{N1}_{N2}")
            matcore_ms, out = time_matcore(fused, A, B, W)
            max_err = float(np.max(np.abs(out - expected)))
            if max_err > 1e-2:
                print(f"  [warn] Family B relu {M}x{K}x{N1}x{N2} max_err={max_err:.6f} > 1e-2")
            speedup = torch_ms / matcore_ms
            speedups.append(speedup)
        except Exception as e:
            print(f"  [error] Matcore Family B relu {M}x{K}x{N1}x{N2} failed: {e}")

        size_str = f"{M}x{K}x{N1}x{N2}"
        print(
            f"relu | {size_str:<15} | {fmt_time_ms(matcore_ms)} |"
            f" {torch_ms:>11.3f} | {fmt_speedup(speedup)} | {fmt_err(max_err)}"
        )

    return speedups


def run_family_c():
    print("\n--- Family C: Attention ---")
    print("Op        | Size      | MatcoreDSL (ms) | PyTorch (ms) | Speedup | max_err")

    sizes = [(16, 16, 8), (32, 32, 16), (64, 64, 32)]
    speedups = []

    clear_cache()

    for M, N, D in sizes:
        Q = np.random.randn(M, D).astype(np.float32) * 0.1
        K = np.random.randn(N, D).astype(np.float32) * 0.1
        V = np.random.randn(N, D).astype(np.float32) * 0.1

        probs = numpy_softmax(Q @ K.T)
        expected = probs @ V

        Q_t = torch.from_numpy(Q).to("cuda", dtype=torch.float32)
        K_t = torch.from_numpy(K).to("cuda", dtype=torch.float32)
        V_t = torch.from_numpy(V).to("cuda", dtype=torch.float32)

        torch_ms, _ = time_torch(lambda: torch.softmax(Q_t @ K_t.T, dim=-1) @ V_t)

        matcore_ms = "ERR"
        speedup = "N/A"
        max_err = "N/A"

        try:
            fused = make_family_c_fused(f"{M}_{N}_{D}")
            matcore_ms, out = time_matcore(fused, Q, K, V)
            max_err = float(np.max(np.abs(out - expected)))
            if max_err > 1e-2:
                print(f"  [warn] Family C attention {M}x{N}x{D} max_err={max_err:.6f} > 1e-2")
            speedup = torch_ms / matcore_ms
            speedups.append(speedup)
        except Exception as e:
            print(f"  [error] Matcore Family C attention {M}x{N}x{D} failed: {e}")

        size_str = f"{M}x{N}x{D}"
        print(
            f"attention | {size_str:<9} | {fmt_time_ms(matcore_ms)} |"
            f" {torch_ms:>11.3f} | {fmt_speedup(speedup)} | {fmt_err(max_err)}"
        )

    return speedups


def avg_or_nan(vals):
    return float(np.mean(vals)) if vals else float("nan")


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark suite.")
        return

    np.random.seed(0)

    print("=========================================")
    print("  MatcoreDSL Fusion Benchmark Suite")
    print("  RTX 4060 Laptop (sm_89), float32")
    print("=========================================")

    a_speedups = run_family_a()
    b_speedups = run_family_b()
    c_speedups = run_family_c()

    print("\n=========================================")
    print("  Summary")
    print("=========================================")

    a_avg = avg_or_nan(a_speedups)
    b_avg = avg_or_nan(b_speedups)
    c_avg = avg_or_nan(c_speedups)

    a_str = f"{a_avg:.2f}x" if np.isfinite(a_avg) else "N/A"
    b_str = f"{b_avg:.2f}x" if np.isfinite(b_avg) else "N/A"
    c_str = f"{c_avg:.2f}x" if np.isfinite(c_avg) else "N/A"

    print(f"Family A avg speedup: {a_str}")
    print(f"Family B avg speedup: {b_str}")
    print(f"Family C avg speedup: {c_str}")


if __name__ == "__main__":
    main()
