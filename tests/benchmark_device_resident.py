"""Benchmark: Device-Resident Tensors vs Host-Path Launch.

Measures the warm-launch overhead with and without device-resident tensors
to validate the 60× PyTorch gap closure.

Targets:
  - 256×256 f16 warm launch:  <0.05ms  (currently ~0.68ms with host path)
  - TFLOP/s:                  >0.5     (currently ~0.049)
"""

from __future__ import annotations

import time
import statistics

import numpy as np

import matcore as mc_module

mc = mc_module.mc


@mc.kernel
def matmul_kernel(A, B, C):
    """Correct MatCore matmul kernel pattern: load → compute → store."""
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)


def benchmark_host_path(kernel, A, B, C, *, warmup: int = 5, repeats: int = 100):
    """Benchmark using standard numpy arrays (host path with H→D→H copies)."""
    for _ in range(warmup):
        mc.launch(kernel, A, B, C, target="nvidia-dgpu:sm_89")

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mc.launch(kernel, A, B, C, target="nvidia-dgpu:sm_89")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms

    return times


def benchmark_device_path(kernel, dA, dB, dC, *, warmup: int = 5, repeats: int = 100):
    """Benchmark using DeviceTensors (device path, no H→D→H copies)."""
    for _ in range(warmup):
        mc.launch(kernel, dA, dB, dC, target="nvidia-dgpu:sm_89")

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mc.launch(kernel, dA, dB, dC, target="nvidia-dgpu:sm_89")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms

    return times


def compute_tflops(m: int, n: int, k: int, time_ms: float) -> float:
    """Compute TFLOP/s for a matmul of shape (m,k) × (k,n).
    
    Args:
        m, n, k: Matrix dimensions
        time_ms: Execution time in milliseconds
    
    Returns:
        TFLOP/s
    """
    flops = 2.0 * m * n * k
    time_s = time_ms / 1000.0
    return flops / time_s / 1e12 if time_s > 0 else 0.0


def run_single_benchmark(M, N, K, repeats=100, warmup=5):
    """Run a complete benchmark for a single matrix size."""
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    # --- Host Path ---
    host_times = benchmark_host_path(matmul_kernel, A, B, C, warmup=warmup, repeats=repeats)
    host_median = statistics.median(host_times)
    host_min = min(host_times)
    host_tflops = compute_tflops(M, N, K, host_median)

    # --- Device Path ---
    dA = mc.to_device(A)
    dB = mc.to_device(B)
    dC = mc.to_device(C)

    try:
        device_times = benchmark_device_path(matmul_kernel, dA, dB, dC, warmup=warmup, repeats=repeats)
        device_median = statistics.median(device_times)
        device_min = min(device_times)
        device_tflops = compute_tflops(M, N, K, device_median)

        # Verify correctness
        result = dC.to_host()
        expected = A.astype(np.float32) @ B.astype(np.float32)
        max_err = float(np.max(np.abs(result.astype(np.float32) - expected)))

        return {
            "M": M,
            "N": N,
            "K": K,
            "host_median_ms": host_median,
            "host_min_ms": host_min,
            "host_tflops": host_tflops,
            "device_median_ms": device_median,
            "device_min_ms": device_min,
            "device_tflops": device_tflops,
            "speedup": host_median / device_median if device_median > 0 else float("inf"),
            "max_error": max_err,
        }
    finally:
        dA.free()
        dB.free()
        dC.free()


def main():
    """Run comprehensive benchmarks across multiple matrix sizes."""
    print("=" * 80)
    print("MatCore Device-Resident Tensor Benchmark")
    print("=" * 80)
    print(f"{'Size':<12} {'Host Median':<15} {'Host Min':<15} {'Device Median':<15} {'Device Min':<15} {'Speedup':<10} {'Error':<12}")
    print("-" * 100)

    # Test multiple sizes: 16x16, 64x64, 128x128, 256x256, 512x512
    test_sizes = [
        (16, 16, 16),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ]

    results = []
    for M, N, K in test_sizes:
        try:
            result = run_single_benchmark(M, N, K, repeats=100, warmup=5)
            results.append(result)

            size_str = f"{M}×{N}×{K}"
            speedup_str = f"{result['speedup']:.1f}×"
            error_str = f"{result['max_error']:.2e}"

            print(f"{size_str:<12} {result['host_median_ms']:<15.3f} {result['host_min_ms']:<15.3f} {result['device_median_ms']:<15.3f} {result['device_min_ms']:<15.3f} {speedup_str:<10} {error_str:<12}")
        except Exception as e:
            print(f"{M}×{N}×{K:<6} ERROR: {e}")

    # Print detailed summary for 256x256 (the target size)
    print("\n" + "=" * 80)
    print("Detailed Summary for 256×256 (Target Size)")
    print("=" * 80)

    target_result = next((r for r in results if r["M"] == 256), None)
    if target_result:
        print(f"\nHost Path (H→D→H copies):")
        print(f"  Median latency: {target_result['host_median_ms']:.3f} ms")
        print(f"  Min latency:    {target_result['host_min_ms']:.3f} ms")
        print(f"  TFLOP/s:        {target_result['host_tflops']:.4f}")

        print(f"\nDevice Path (tensors stay on GPU):")
        print(f"  Median latency: {target_result['device_median_ms']:.3f} ms")
        print(f"  Min latency:    {target_result['device_min_ms']:.3f} ms")
        print(f"  TFLOP/s:        {target_result['device_tflops']:.4f}")

        print(f"\nComparison:")
        print(f"  Speedup:        {target_result['speedup']:.1f}×")
        print(f"  Max error:      {target_result['max_error']:.2e}")

        print(f"\nTarget Validation:")
        if target_result["device_median_ms"] < 0.05:
            print(f"  ✅ <0.05ms:    PASS ({target_result['device_median_ms']:.3f} ms)")
        else:
            print(f"  ❌ <0.05ms:    FAIL ({target_result['device_median_ms']:.3f} ms)")

        if target_result["device_tflops"] > 0.5:
            print(f"  ✅ >0.5 TFLOP/s: PASS ({target_result['device_tflops']:.4f})")
        else:
            print(f"  ❌ >0.5 TFLOP/s: FAIL ({target_result['device_tflops']:.4f})")

    print("\n" + "=" * 80)
    print("TFLOP/s Summary Across All Sizes")
    print("=" * 80)
    for result in results:
        size_str = f"{result['M']}×{result['N']}×{result['K']}"
        print(f"{size_str:<12} Host: {result['host_tflops']:<8.4f} TFLOP/s  |  Device: {result['device_tflops']:<8.4f} TFLOP/s  |  Speedup: {result['speedup']:<6.1f}×")


if __name__ == "__main__":
    main()
