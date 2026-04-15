"""Fair fusion benchmark with device-resident tensors on both sides.

Both MatcoreDSL and PyTorch keep data on GPU. Both use wall-clock timing
after warmup (absorbs JIT). This is the apples-to-apples comparison.
"""
import os, sys, time, shutil, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matcore as mc
from matcore.device_tensor import DeviceTensor, to_device

import torch
import torch.nn.functional as F

WARMUP = 20
ITERS = 100
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clear_cache():
    cache = os.path.join(REPO_ROOT, ".matcore_cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache)


def torch_gpu_time(fn, warmup=WARMUP, iters=ITERS):
    """Time PyTorch using CUDA events (GPU-only timing)."""
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


def matcore_devtensor_time(fn, *dev_args, warmup=WARMUP, iters=ITERS):
    """Time MatcoreDSL with device-resident tensors (no H→D copy in loop)."""
    for _ in range(warmup):
        fn(*dev_args)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*dev_args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return statistics.median(times), out


def matcore_host_time(fn, *np_args, warmup=WARMUP, iters=ITERS):
    """Time MatcoreDSL with host arrays (includes H→D copy)."""
    for _ in range(warmup):
        fn(*np_args)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*np_args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return statistics.median(times), out


# ─── Family A: GEMM + epilogue ───────────────────────────────────────────────

def run_family_a():
    print("\n━━━ Family A: GEMM + Epilogue (DeviceTensor) ━━━")
    print(f"{'Op':<5}{'Size':<16}{'MC DevTens':>12}{'MC Host':>12}{'PyTorch':>12}{'DT/Torch':>10}{'GFLOPS':>10}{'err':>10}")
    print("─" * 87)

    sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    ops_config = [
        ("relu", mc.relu, lambda t: torch.relu(t)),
        ("gelu", mc.gelu, lambda t: F.gelu(t)),
        ("exp",  mc.exp,  lambda t: torch.exp(t)),
    ]
    speedups = []

    clear_cache()

    for op_name, mc_op, torch_op in ops_config:
        for M, K, N in sizes:
            scale = 0.1 if op_name == "exp" else 1.0
            A_np = (np.random.randn(M, K) * scale).astype(np.float32)
            B_np = (np.random.randn(K, N) * scale).astype(np.float32)

            # PyTorch GPU-resident
            A_t = torch.from_numpy(A_np).cuda()
            B_t = torch.from_numpy(B_np).cuda()
            torch_fn = lambda _a=A_t, _b=B_t, _op=torch_op: _op(_a @ _b)
            torch_ms = torch_gpu_time(torch_fn)

            # MatcoreDSL fused
            @mc.fused
            def fused_op(A, B, _op=mc_op):
                return _op(A @ B)
            fused_op.__name__ = f"gemm_{op_name}_{M}_{K}_{N}"

            # Host path
            mc_host_ms, _ = matcore_host_time(fused_op, A_np, B_np)

            # DeviceTensor path
            dA = to_device(A_np)
            dB = to_device(B_np)
            try:
                mc_dev_ms, out_dt = matcore_devtensor_time(fused_op, dA, dB)

                # Verify correctness
                result = out_dt.to_host() if isinstance(out_dt, DeviceTensor) else out_dt
                C = A_np @ B_np
                if op_name == "relu":
                    expected = np.maximum(C, 0)
                elif op_name == "gelu":
                    expected = C * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (C + 0.044715 * C**3)))
                else:
                    expected = np.exp(C)
                max_err = float(np.max(np.abs(result - expected)))

                flops = 2.0 * M * K * N + M * N
                gflops = flops / (mc_dev_ms / 1000) / 1e9
                sp = torch_ms / mc_dev_ms
                speedups.append(sp)

                print(f"{op_name:<5}{M}x{K}x{N:<9}{mc_dev_ms:>12.3f}{mc_host_ms:>12.3f}{torch_ms:>12.3f}{sp:>9.2f}x{gflops:>10.1f}{max_err:>10.6f}")
            except Exception as e:
                print(f"{op_name:<5}{M}x{K}x{N:<9} ERROR: {e}")
            finally:
                dA.free()
                dB.free()

    return speedups


# ─── Standalone Matmul Comparison ────────────────────────────────────────────

def run_standalone_matmul():
    """Standalone matmul via @mc.kernel + mc.launch (DeviceTensor)."""
    print("\n━━━ Standalone Matmul (@mc.kernel, DeviceTensor) ━━━")
    print(f"{'Size':<16}{'MC DevTens':>12}{'MC Host':>12}{'PyTorch':>12}{'DT/Torch':>10}{'TFLOPS':>10}")
    print("─" * 72)

    @mc.kernel
    def matmul_k(A, B, C):
        a = mc.load(A)
        b = mc.load(B)
        c = mc.matmul(a, b)
        mc.store(c, C)

    sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    speedups = []

    clear_cache()

    for M, K, N in sizes:
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)
        C_np = np.zeros((M, N), dtype=np.float32)
        tgt = "nvidia-dgpu:sm_89"

        # PyTorch
        A_t = torch.from_numpy(A_np).cuda()
        B_t = torch.from_numpy(B_np).cuda()
        torch_fn = lambda _a=A_t, _b=B_t: _a @ _b
        torch_ms = torch_gpu_time(torch_fn)

        # Host path warmup + timing
        for _ in range(WARMUP):
            mc.launch(matmul_k, A_np, B_np, C_np, target=tgt)
        times_host = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            mc.launch(matmul_k, A_np, B_np, C_np, target=tgt)
            t1 = time.perf_counter()
            times_host.append((t1 - t0) * 1000.0)
        mc_host_ms = statistics.median(times_host)

        # DeviceTensor path
        dA = to_device(A_np)
        dB = to_device(B_np)
        dC = to_device(C_np)
        try:
            for _ in range(WARMUP):
                mc.launch(matmul_k, dA, dB, dC, target=tgt)
            times_dev = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                mc.launch(matmul_k, dA, dB, dC, target=tgt)
                t1 = time.perf_counter()
                times_dev.append((t1 - t0) * 1000.0)
            mc_dev_ms = statistics.median(times_dev)

            flops = 2.0 * M * K * N
            tflops = flops / (mc_dev_ms / 1000) / 1e12
            sp = torch_ms / mc_dev_ms
            speedups.append(sp)
            print(f"{M}x{K}x{N:<9}{mc_dev_ms:>12.3f}{mc_host_ms:>12.3f}{torch_ms:>12.3f}{sp:>9.2f}x{tflops:>10.4f}")
        except Exception as e:
            print(f"{M}x{K}x{N:<9} ERROR: {e}")
        finally:
            dA.free()
            dB.free()
            dC.free()

    return speedups


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 87)
    print("  MatcoreDSL Fusion Benchmark — Device-Resident Tensors")
    print(f"  RTX 4060 Laptop (sm_89), float32")
    print(f"  Warmup={WARMUP} (absorbs JIT), Iters={ITERS}")
    print(f"  Both sides: GPU-resident data, wall-clock after warmup")
    print("=" * 87)

    matmul_sp = run_standalone_matmul()
    a_sp = run_family_a()

    print("\n" + "=" * 87)
    print("  Summary")
    print("=" * 87)
    if matmul_sp:
        print(f"  Standalone matmul avg vs PyTorch: {sum(matmul_sp)/len(matmul_sp):.2f}x  (n={len(matmul_sp)})")
    if a_sp:
        print(f"  Family A avg vs PyTorch:          {sum(a_sp)/len(a_sp):.2f}x  (n={len(a_sp)})")
    print("=" * 87)
