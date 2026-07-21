from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matcore as mc


def _reference_block_attn_res(
    blocks: np.ndarray,
    partial: np.ndarray,
    query: np.ndarray,
    *,
    block_count: int,
    has_partial: bool,
    eps: float,
) -> np.ndarray:
    sources: list[np.ndarray] = []
    if block_count:
        sources.append(blocks[:block_count])
    if has_partial:
        sources.append(partial[None, ...])
    if not sources:
        return np.zeros_like(partial)

    packed = np.concatenate(sources, axis=0)
    inv_rms = 1.0 / np.sqrt(np.mean(packed * packed, axis=-1, keepdims=True) + eps)
    scores = np.sum((packed * inv_rms) * query.reshape(1, 1, 1, -1), axis=-1)
    scores = scores - np.max(scores, axis=0, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=0, keepdims=True)
    return np.sum(weights[..., None] * packed, axis=0).astype(np.float32)


def _is_missing_cuda_runtime(exc: Exception) -> bool:
    message = str(exc).lower()
    needles = [
        "failed to import native module",
        "could not load cuda",
        "cuda driver",
        "cuda runtime",
        "libcuda",
        "no cuda",
        "no nvidia",
    ]
    return any(needle in message for needle in needles)


def test_block_attn_res_host_correctness() -> None:
    rng = np.random.default_rng(20260501)
    blocks = rng.normal(size=(3, 2, 3, 7)).astype(np.float32)
    partial = rng.normal(size=(2, 3, 7)).astype(np.float32)
    query = rng.normal(size=(7,)).astype(np.float32)
    eps = 1.0e-5

    try:
        out = mc.block_attn_res(
            blocks,
            partial,
            query,
            block_count=2,
            has_partial=True,
            eps=eps,
            target="nvidia-dgpu:sm_89",
        )
    except Exception as exc:
        if _is_missing_cuda_runtime(exc):
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        raise

    expected = _reference_block_attn_res(
        blocks,
        partial,
        query,
        block_count=2,
        has_partial=True,
        eps=eps,
    )
    np.testing.assert_allclose(out, expected, rtol=1.0e-4, atol=1.0e-4)


def test_jit_block_attn_res_host_correctness() -> None:
    rng = np.random.default_rng(20260502)
    blocks = rng.normal(size=(2, 1, 2, 4)).astype(np.float32)
    partial = rng.normal(size=(1, 2, 4)).astype(np.float32)
    query = rng.normal(size=(4,)).astype(np.float32)
    eps = 1.0e-5

    @mc.jit
    def depth_residual(blocks, partial, query, block_count, has_partial=True):
        return mc.block_attn_res(
            blocks,
            partial,
            query,
            block_count=block_count,
            has_partial=has_partial,
            eps=eps,
        )

    try:
        out = depth_residual(
            blocks,
            partial,
            query,
            1,
            has_partial=True,
            target="nvidia-dgpu:sm_89",
        )
    except Exception as exc:
        if _is_missing_cuda_runtime(exc):
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        raise

    expected = _reference_block_attn_res(
        blocks,
        partial,
        query,
        block_count=1,
        has_partial=True,
        eps=eps,
    )
    np.testing.assert_allclose(out, expected, rtol=1.0e-4, atol=1.0e-4)
