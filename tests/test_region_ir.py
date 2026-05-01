from __future__ import annotations

import pathlib
import sys
import copy

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matcore as mc
from matcore import frontend


class _DummyNative:
    calls: list[tuple[dict, str, tuple[object, ...]]] = []

    @staticmethod
    def compile_and_run(kernel_ir, target, *args):
        _DummyNative.calls.append((kernel_ir, target, args))


def _sample_inputs():
    blocks = np.zeros((3, 2, 4, 5), dtype=np.float32)
    partial = np.zeros((2, 4, 5), dtype=np.float32)
    query = np.zeros((5,), dtype=np.float32)
    return blocks, partial, query


def _native_or_skip():
    try:
        return frontend._get_native_module()
    except RuntimeError as exc:
        pytest.skip(f"native extension unavailable for this interpreter: {exc}")


def _base_region_kernel() -> dict:
    return {
        "kernel_name": "region_verifier_test",
        "version": "region_v1",
        "params": ["blocks", "partial", "query"],
        "values": [
            {
                "id": 0,
                "symbol": "blocks",
                "dtype": "float32",
                "shape": [3, 2, 4, 5],
                "kind": "input",
                "storage_hint": "auto",
                "producer": -1,
                "consumers": [0],
            },
            {
                "id": 1,
                "symbol": "partial",
                "dtype": "float32",
                "shape": [2, 4, 5],
                "kind": "input",
                "storage_hint": "auto",
                "producer": -1,
                "consumers": [0],
            },
            {
                "id": 2,
                "symbol": "query",
                "dtype": "float32",
                "shape": [5],
                "kind": "input",
                "storage_hint": "auto",
                "producer": -1,
                "consumers": [0],
            },
            {
                "id": 3,
                "symbol": "out",
                "dtype": "float32",
                "shape": [2, 4, 5],
                "kind": "output",
                "storage_hint": "auto",
                "producer": 0,
                "consumers": [],
            },
        ],
        "nodes": [
            {
                "id": 0,
                "op": "block_attn_res",
                "inputs": [0, 1, 2],
                "outputs": [3],
                "attrs": {
                    "block_count": 2,
                    "has_partial": True,
                    "eps": 1.0e-5,
                },
            }
        ],
        "input_values": [0, 1, 2],
        "output_values": [3],
        "topo_order": [0],
    }


def _runtime_tensors(query: np.ndarray | None = None):
    blocks, partial, default_query = _sample_inputs()
    out = np.empty_like(partial)
    return blocks, partial, query if query is not None else default_query, out


def _expect_native_error(kernel: dict, needle: str, *args: object) -> None:
    native = _native_or_skip()
    try:
        native.compile_and_run(kernel, "nvidia-dgpu:sm_89", *args)
    except RuntimeError as exc:
        message = str(exc)
        assert needle in message
    else:
        raise AssertionError(f"Expected native RegionV1 error containing {needle!r}")


def test_direct_block_attn_res_emits_region_v1_kernel() -> None:
    original_get_native_module = frontend._get_native_module
    _DummyNative.calls.clear()
    frontend._get_native_module = lambda: _DummyNative()
    try:
        blocks, partial, query = _sample_inputs()
        out = mc.block_attn_res(
            blocks,
            partial,
            query,
            block_count=2,
            has_partial=True,
            eps=1.0e-5,
            target="nvidia-dgpu:sm_89",
        )
    finally:
        frontend._get_native_module = original_get_native_module

    assert out.shape == partial.shape
    assert out.dtype == np.float32
    assert len(_DummyNative.calls) == 1
    kernel_ir, target, args = _DummyNative.calls[0]
    assert target == "nvidia-dgpu:sm_89"
    assert len(args) == 4
    assert kernel_ir["version"] == "region_v1"
    assert kernel_ir["params"] == ["blocks", "partial", "query"]
    assert kernel_ir["input_values"] == [0, 1, 2]
    assert len(kernel_ir["output_values"]) == 1
    assert kernel_ir["nodes"][0]["op"] == "block_attn_res"
    assert kernel_ir["nodes"][0]["attrs"] == {
        "block_count": 2,
        "has_partial": True,
        "eps": 1.0e-5,
    }


def test_jit_block_attn_res_traces_tensor_inputs_and_scalar_attrs() -> None:
    original_get_native_module = frontend._get_native_module
    _DummyNative.calls.clear()
    frontend._get_native_module = lambda: _DummyNative()
    try:

        @mc.jit
        def depth_residual(blocks, partial, query, block_count, has_partial=True):
            return mc.block_attn_res(
                blocks,
                partial,
                query,
                block_count=block_count,
                has_partial=has_partial,
                eps=1.0e-4,
            )

        blocks, partial, query = _sample_inputs()
        out = depth_residual(
            blocks,
            partial,
            query,
            1,
            has_partial=False,
            target="nvidia-dgpu:sm_89",
        )
    finally:
        frontend._get_native_module = original_get_native_module

    assert out.shape == partial.shape
    assert len(_DummyNative.calls) == 1
    kernel_ir, _, args = _DummyNative.calls[0]
    assert len(args) == 4
    assert kernel_ir["kernel_name"] == "depth_residual_region"
    assert kernel_ir["params"] == ["blocks", "partial", "query"]
    assert kernel_ir["nodes"][0]["attrs"] == {
        "block_count": 1,
        "has_partial": False,
        "eps": 1.0e-4,
    }


def test_block_attn_res_rejects_non_nvidia_target() -> None:
    blocks, partial, query = _sample_inputs()
    try:
        mc.block_attn_res(
            blocks,
            partial,
            query,
            block_count=1,
            target="x86-auto",
        )
    except ValueError as exc:
        assert "nvidia-dgpu" in str(exc)
    else:
        raise AssertionError("Expected RegionV1 to reject non-NVIDIA targets")


def test_native_region_verifier_rejects_bad_block_count() -> None:
    kernel = _base_region_kernel()
    kernel["nodes"][0]["attrs"]["block_count"] = 4
    _expect_native_error(
        kernel,
        "RegionV1 verifier: block_attn_res block_count",
        *_runtime_tensors(),
    )


def test_native_region_verifier_rejects_bad_topo_order() -> None:
    kernel = _base_region_kernel()
    kernel["topo_order"] = [99]
    _expect_native_error(
        kernel,
        "RegionV1 verifier: topo_order references missing node id 99",
        *_runtime_tensors(),
    )


def test_native_region_verifier_rejects_runtime_shape_mismatch() -> None:
    kernel = _base_region_kernel()
    bad_query = np.zeros((6,), dtype=np.float32)
    _expect_native_error(
        kernel,
        "RegionV1 verifier: runtime input 2 runtime shape",
        *_runtime_tensors(query=bad_query),
    )


def test_native_region_verifier_rejects_multiple_outputs() -> None:
    kernel = _base_region_kernel()
    extra_output = copy.deepcopy(kernel["values"][3])
    extra_output["id"] = 4
    extra_output["symbol"] = "out2"
    kernel["values"].append(extra_output)
    kernel["output_values"] = [3, 4]
    _expect_native_error(
        kernel,
        "RegionV1 verifier: region_v1 currently supports exactly one output value",
        *_runtime_tensors(),
    )


def test_native_region_accepts_multi_op_but_emitter_reports_lowering_gap() -> None:
    kernel = _base_region_kernel()
    first_output = copy.deepcopy(kernel["values"][3])
    first_output["kind"] = "intermediate"
    first_output["symbol"] = "mid"
    first_output["producer"] = 0
    final_output = copy.deepcopy(kernel["values"][3])
    final_output["id"] = 4
    final_output["symbol"] = "out"
    final_output["producer"] = 1
    kernel["values"][3] = first_output
    kernel["values"].append(final_output)
    kernel["nodes"][0]["outputs"] = [3]
    kernel["nodes"].append(
        {
            "id": 1,
            "op": "block_attn_res",
            "inputs": [0, 3, 2],
            "outputs": [4],
            "attrs": {
                "block_count": 2,
                "has_partial": True,
                "eps": 1.0e-5,
            },
        }
    )
    kernel["values"][0]["consumers"] = [0, 1]
    kernel["values"][1]["consumers"] = [0]
    kernel["values"][2]["consumers"] = [0, 1]
    kernel["values"][3]["consumers"] = [1]
    kernel["output_values"] = [4]
    kernel["topo_order"] = [0, 1]

    _expect_native_error(
        kernel,
        "RegionMlirEmitter: verified multi-op regions are accepted",
        *_runtime_tensors(),
    )
