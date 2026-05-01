from __future__ import annotations

import pathlib
import sys

import numpy as np

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
