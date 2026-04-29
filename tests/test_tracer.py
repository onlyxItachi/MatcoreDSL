"""Tests for the fusion tracer system."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matcore as mc
from matcore.frontend import FusionTraceBuilder, TraceNode, TraceValue, TracerTensor


def test_simple_matmul_trace():
    """Test tracing a simple matmul."""
    builder = FusionTraceBuilder()
    Q = builder.add_input("Q", np.zeros((64, 32), dtype=np.float16))
    K = builder.add_input("K", np.zeros((64, 32), dtype=np.float16))
    result = Q @ K.T
    graph = builder.finish(result)

    assert graph["version"] == "graph_v2"
    assert len(graph["values"]) == 3
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["op"] == "matmul"
    assert graph["nodes"][0]["attrs"]["transpose_rhs"] is True
    assert graph["input_values"] == [0, 1]
    assert graph["output_values"] == [2]
    assert graph["values"][2]["shape"] == [64, 64]
    print("  simple_matmul_trace: PASS")


def test_attention_trace():
    """Test tracing attention pattern: softmax(Q @ K.T) @ V"""
    builder = FusionTraceBuilder()
    Q = builder.add_input("Q", np.zeros((64, 32), dtype=np.float16))
    K = builder.add_input("K", np.zeros((64, 32), dtype=np.float16))
    V = builder.add_input("V", np.zeros((64, 32), dtype=np.float16))

    scores = Q @ K.T
    attn = builder.add_softmax(scores)
    result = attn @ V
    graph = builder.finish(result)

    assert len(graph["values"]) == 6
    assert len(graph["nodes"]) == 3
    assert graph["nodes"][0]["op"] == "matmul"
    assert graph["nodes"][1]["op"] == "softmax"
    assert graph["nodes"][2]["op"] == "matmul"
    assert len(graph["topo_order"]) == 3
    print("  attention_trace: PASS")


def test_elementwise_trace():
    """Test tracing elementwise ops."""
    builder = FusionTraceBuilder()
    A = builder.add_input("A", np.zeros((32, 32), dtype=np.float32))
    B = builder.add_input("B", np.zeros((32, 32), dtype=np.float32))

    C = A + B
    D = builder.add_elementwise_unary("relu", C)
    graph = builder.finish(D)

    assert len(graph["nodes"]) == 2
    assert graph["nodes"][0]["op"] == "add"
    assert graph["nodes"][1]["op"] == "relu"
    print("  elementwise_trace: PASS")


def test_fused_decorator():
    """Test @mc.fused decorator executes and returns output."""

    @mc.fused
    def gemm_relu(A, B):
        return mc.relu(A @ B)

    A = np.random.randn(64, 32).astype(np.float32)
    B = np.random.randn(32, 64).astype(np.float32)

    out = gemm_relu(A, B)
    expected = np.maximum(A @ B, 0)
    assert out.shape == (64, 64)
    assert out.dtype == np.float32
    assert np.max(np.abs(out - expected)) < 1e-3
    print("  fused_decorator: PASS")


def test_fused_gemm_epilogue():
    """Test fused GEMM + ReLU output shape and stability."""

    @mc.fused
    def gemm_relu(A, B):
        return mc.relu(A @ B)

    A = np.random.randn(64, 32).astype(np.float32)
    B = np.random.randn(32, 64).astype(np.float32)

    out = gemm_relu(A, B)
    assert out.shape == (64, 64)
    assert np.isfinite(out).all()
    print("  fused_gemm_epilogue: PASS")


def test_topo_order():
    """Test topological ordering is correct."""
    builder = FusionTraceBuilder()
    A = builder.add_input("A", np.zeros((32, 32), dtype=np.float32))
    B = builder.add_input("B", np.zeros((32, 32), dtype=np.float32))

    C = A @ B
    D = builder.add_elementwise_unary("exp", C)
    E = builder.add_elementwise_unary("relu", D)
    graph = builder.finish(E)

    assert graph["topo_order"] == [0, 1, 2], f"Expected [0,1,2] got {graph['topo_order']}"
    print("  topo_order: PASS")


def test_logical_dtype_trace():
    """Logical dtype wrappers should trace their logical dtype, not storage dtype."""
    builder = FusionTraceBuilder()
    bf16 = mc.asdtype(np.zeros((8, 8), dtype=np.uint16), "bfloat16")
    fp8 = mc.asdtype(np.zeros((8, 8), dtype=np.uint8), "float8_e4m3fn")

    lhs = builder.add_input("lhs", bf16)
    rhs = builder.add_input("rhs", fp8)
    graph = builder.finish((lhs, rhs))

    assert graph["values"][0]["dtype"] == "bfloat16"
    assert graph["values"][1]["dtype"] == "float8_e4m3fn"
    print("  logical_dtype_trace: PASS")


def test_matmul_trace_dtype_inference():
    """Tracer matmul should infer the same output dtypes as launch-time validation."""
    builder = FusionTraceBuilder()
    lhs = builder.add_input("lhs", mc.asdtype(np.zeros((4, 6), dtype=np.int8), "int8"))
    rhs = builder.add_input("rhs", mc.asdtype(np.zeros((6, 5), dtype=np.int8), "int8"))
    graph = builder.finish(lhs @ rhs)
    assert graph["values"][-1]["dtype"] == "int32"

    builder_fp8 = FusionTraceBuilder()
    fp8_lhs = builder_fp8.add_input(
        "lhs", mc.asdtype(np.zeros((4, 6), dtype=np.uint8), "float8_e4m3fn")
    )
    fp8_rhs = builder_fp8.add_input(
        "rhs", mc.asdtype(np.zeros((6, 5), dtype=np.uint8), "float8_e4m3fn")
    )
    graph_fp8 = builder_fp8.finish(fp8_lhs @ fp8_rhs)
    assert graph_fp8["values"][-1]["dtype"] == "float32"
    print("  matmul_trace_dtype_inference: PASS")


def test_cast_trace():
    """Tracer-aware cast should emit canonical target dtypes."""
    builder = FusionTraceBuilder()
    lhs = builder.add_input("lhs", np.zeros((4, 4), dtype=np.float16))
    casted = builder.add_cast(lhs, "bf16")
    graph = builder.finish(casted)

    assert graph["nodes"][-1]["op"] == "cast"
    assert graph["nodes"][-1]["attrs"]["target_dtype"] == "bfloat16"
    assert graph["values"][-1]["dtype"] == "bfloat16"
    print("  cast_trace: PASS")


if __name__ == "__main__":
    print("=== Tracer Unit Tests ===")
    test_simple_matmul_trace()
    test_attention_trace()
    test_elementwise_trace()
    test_fused_decorator()
    test_fused_gemm_epilogue()
    test_topo_order()
    test_logical_dtype_trace()
    test_matmul_trace_dtype_inference()
    test_cast_trace()
    print("\nAll tracer tests passed! ✓")
