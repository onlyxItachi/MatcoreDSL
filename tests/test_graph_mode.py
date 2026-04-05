"""Test CUDA graph capture/replay via mc.create_plan(graph_mode=True)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matcore as mc

@mc.kernel
def matmul(A, B, C):
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)

TARGET = "nvidia-dgpu:sm_89"

def test_graph_correctness():
    """Verify graph replay produces correct results."""
    N = 64
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)

    dA = mc.to_device(A)
    dB = mc.to_device(B)
    dC = mc.to_device(C)

    # Create plan with graph_mode
    plan = mc.create_plan(matmul, dA, dB, dC, target=TARGET, graph_mode=True)

    # First execute — captures graph
    mc.execute_plan(plan, dA, dB, dC)
    result1 = dC.to_host()

    # Second execute — replays graph
    dC.zero_()
    mc.execute_plan(plan, dA, dB, dC)
    result2 = dC.to_host()

    expected = A.astype(np.float32) @ B.astype(np.float32)
    expected = expected.astype(np.float16)

    atol = np.max(np.abs(result1.astype(np.float32) - expected.astype(np.float32)))
    print(f"Graph correctness: atol={atol:.6f} (capture)")
    assert atol < 1.0, f"Graph capture result incorrect: atol={atol}"

    atol2 = np.max(np.abs(result2.astype(np.float32) - expected.astype(np.float32)))
    print(f"Graph correctness: atol={atol2:.6f} (replay)")
    assert atol2 < 1.0, f"Graph replay result incorrect: atol={atol2}"

    dA.free(); dB.free(); dC.free()
    print("PASS: Graph correctness test")


def test_graph_pointer_validation():
    """Verify graph rejects changed pointers."""
    N = 32
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)

    dA = mc.to_device(A)
    dB = mc.to_device(B)
    dC = mc.to_device(C)

    plan = mc.create_plan(matmul, dA, dB, dC, target=TARGET, graph_mode=True)
    mc.execute_plan(plan, dA, dB, dC)  # capture

    # Create new tensors with different pointers
    dA2 = mc.to_device(A)
    try:
        mc.execute_plan(plan, dA2, dB, dC)  # should fail
        assert False, "Should have raised on pointer mismatch"
    except RuntimeError as e:
        assert "pointer changed" in str(e).lower() or "data pointer changed" in str(e)
        print(f"PASS: Pointer validation works: {e}")

    dA.free(); dB.free(); dC.free(); dA2.free()


def test_no_graph_still_works():
    """Plans without graph_mode still work normally."""
    N = 32
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)

    dA = mc.to_device(A)
    dB = mc.to_device(B)
    dC = mc.to_device(C)

    plan = mc.create_plan(matmul, dA, dB, dC, target=TARGET)
    mc.execute_plan(plan, dA, dB, dC)
    result = dC.to_host()

    expected = A.astype(np.float32) @ B.astype(np.float32)
    atol = np.max(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
    print(f"No-graph plan: atol={atol:.6f}")
    assert atol < 1.0

    # Execute again with different data (same pointers)
    dC.zero_()
    mc.execute_plan(plan, dA, dB, dC)
    result2 = dC.to_host()
    atol2 = np.max(np.abs(result2.astype(np.float32) - expected.astype(np.float32)))
    assert atol2 < 1.0

    dA.free(); dB.free(); dC.free()
    print("PASS: No-graph plan test")


def benchmark_graph_vs_plan():
    """Benchmark graph replay vs normal plan execution."""
    N = 256
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    C = np.zeros((N, N), dtype=np.float16)

    dA = mc.to_device(A)
    dB = mc.to_device(B)
    dC = mc.to_device(C)

    # Normal plan
    plan_normal = mc.create_plan(matmul, dA, dB, dC, target=TARGET)
    # Warm up
    for _ in range(5):
        mc.execute_plan(plan_normal, dA, dB, dC)

    NUM_ITER = 100
    t0 = time.perf_counter()
    for _ in range(NUM_ITER):
        mc.execute_plan(plan_normal, dA, dB, dC)
    t_plan = (time.perf_counter() - t0) / NUM_ITER * 1000

    # Graph plan
    plan_graph = mc.create_plan(matmul, dA, dB, dC, target=TARGET, graph_mode=True)
    mc.execute_plan(plan_graph, dA, dB, dC)  # capture
    # Warm up replays
    for _ in range(5):
        mc.execute_plan(plan_graph, dA, dB, dC)

    t0 = time.perf_counter()
    for _ in range(NUM_ITER):
        mc.execute_plan(plan_graph, dA, dB, dC)
    t_graph = (time.perf_counter() - t0) / NUM_ITER * 1000

    # Also benchmark mc.launch for comparison
    for _ in range(5):
        mc.launch(matmul, dA, dB, dC, target=TARGET)

    t0 = time.perf_counter()
    for _ in range(NUM_ITER):
        mc.launch(matmul, dA, dB, dC, target=TARGET)
    t_launch = (time.perf_counter() - t0) / NUM_ITER * 1000

    print(f"\n{'='*60}")
    print(f"Benchmark: {N}x{N} f16 matmul ({NUM_ITER} iterations)")
    print(f"{'='*60}")
    print(f"  mc.launch():       {t_launch:.3f} ms")
    print(f"  execute_plan():    {t_plan:.3f} ms  ({t_launch/t_plan:.1f}x vs launch)")
    print(f"  graph replay:      {t_graph:.3f} ms  ({t_launch/t_graph:.1f}x vs launch)")
    print(f"  graph vs plan:     {t_plan/t_graph:.1f}x speedup")
    print(f"{'='*60}")

    dA.free(); dB.free(); dC.free()


if __name__ == "__main__":
    print("=== CUDA Graph Mode Tests ===\n")

    test_no_graph_still_works()
    print()

    test_graph_correctness()
    print()

    test_graph_pointer_validation()
    print()

    benchmark_graph_vs_plan()
    print("\nAll graph tests passed!")
