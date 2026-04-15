"""Test fusion analysis on traced graphs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matcore as mc
from matcore.frontend import FusionTraceBuilder


def test_family_a_classification():
    """matmul -> relu should be classified as Family A."""
    builder = FusionTraceBuilder()
    A = builder.add_input("A", np.zeros((64, 32), dtype=np.float32))
    B = builder.add_input("B", np.zeros((32, 64), dtype=np.float32))
    C = A @ B
    D = builder.add_elementwise_unary("relu", C)
    graph = builder.finish(D)
    assert graph["nodes"][0]["op"] == "matmul"
    assert graph["nodes"][1]["op"] == "relu"
    print("  family_a_classification: PASS")


def test_family_c_classification():
    """matmul -> softmax -> matmul should be classified as Family C."""
    builder = FusionTraceBuilder()
    Q = builder.add_input("Q", np.zeros((64, 32), dtype=np.float16))
    K = builder.add_input("K", np.zeros((64, 32), dtype=np.float16))
    V = builder.add_input("V", np.zeros((64, 32), dtype=np.float16))
    S = Q @ K.T
    P = builder.add_softmax(S)
    O = P @ V
    graph = builder.finish(O)
    ops = [n["op"] for n in graph["nodes"]]
    assert ops == ["matmul", "softmax", "matmul"]
    print("  family_c_classification: PASS")


def test_escape_analysis():
    """Intermediate values with single consumer should not escape."""
    builder = FusionTraceBuilder()
    A = builder.add_input("A", np.zeros((64, 32), dtype=np.float32))
    B = builder.add_input("B", np.zeros((32, 64), dtype=np.float32))
    C = A @ B
    D = builder.add_elementwise_unary("relu", C)
    graph = builder.finish(D)

    assert len(graph["values"][2]["consumers"]) == 1
    assert graph["values"][2]["kind"] == "intermediate"
    assert graph["values"][3]["kind"] == "output"
    print("  escape_analysis: PASS")


def test_long_chain():
    """Long chain: matmul -> bias -> gelu -> cast should be Family A."""
    builder = FusionTraceBuilder()
    A = builder.add_input("A", np.zeros((64, 32), dtype=np.float32))
    B = builder.add_input("B", np.zeros((32, 64), dtype=np.float32))
    bias = builder.add_input("bias", np.zeros((64, 64), dtype=np.float32))

    C = A @ B
    D = builder.add_elementwise_binary("add", C, bias)
    E = builder.add_elementwise_unary("gelu", D)
    graph = builder.finish(E)

    ops = [n["op"] for n in graph["nodes"]]
    assert ops == ["matmul", "add", "gelu"]
    assert len(graph["values"][3]["consumers"]) == 1
    assert len(graph["values"][4]["consumers"]) == 1
    print("  long_chain: PASS")


if __name__ == "__main__":
    print("=== Fusion Analysis Tests ===")
    test_family_a_classification()
    test_family_c_classification()
    test_escape_analysis()
    test_long_chain()
    print("\nAll fusion analysis tests passed! ✓")
