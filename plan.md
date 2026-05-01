# Runtime-JIT Region IR + Block Attention Residual Kernel

## Summary

- Add `@mc.jit` as a structured tensor-program JIT path separate from
  `@mc.fused`.
- Add `region_v1` IR and native parsing/cache/lowering support.
- Add a first intrinsic, `mc.block_attn_res`, using packed block history and a
  direct NVIDIA GPU emitter.
- Keep current Family A/B/C fusion behavior unchanged.

Status: implemented and smoke-tested on the CPython 3.12 native extension.
The project `.venv` is present and works for native MatCore execution; pytest is
not installed in that env.

## Public API

- `@mc.jit` traces region programs and runtime-JIT compiles them.
- `mc.block_attn_res(blocks, partial, query, *, block_count, has_partial=True,
  eps=1e-6)` is the first v1 intrinsic.
- V1 input ABI:
  - `blocks`: `[MAX_BLOCKS, B, T, D]`
  - `partial`: `[B, T, D]`
  - `query`: `[D]`
  - output: `[B, T, D]`
- V1 specializes `block_count`, `has_partial`, and `eps` at compile time.
- V1 supports NVIDIA GPU and `float32` only.

## Key Changes

- Extend `KernelIR` with `kRegionV1` and `RegionIR`.
- Add Python `RegionTraceBuilder`, `RegionTensor`, `@mc.jit`, and
  tracer-aware `mc.block_attn_res`.
- Add native `region_v1` parser and cache-key hashing.
- Add a `RegionMlirEmitter` route in `MlirEngine::BuildAndLower`.
- Add rank 1/3/4 generic memref descriptor support in the executor.
- Add tests for IR capture, validation, correctness, and existing fusion
  regression coverage.

## Tests

- Passed: `ninja -C /home/hamza-usta/MatcoreDSL/build-review`
- New tests:
  - `tests/test_region_ir.py`
  - `tests/test_block_attn_res.py`
- Regression tests:
  - Passed: `tests/test_frontend_contract.py`
  - Passed: `tests/test_devtensor_fused.py`
  - Passed: `tests/test_fusion_contracts.py`
  - Pending broader sweep: `tests/test_family_a.py`
  - Pending broader sweep: `tests/test_family_b.py`
  - Pending broader sweep: `tests/test_family_c.py`

## Follow-Up Tasks

- Add restricted `mc.static_range` and structured `if` capture.
- Add `float16`/`bfloat16` support with `float32` accumulation.
- Add runtime scalar ABI for `block_count` and `has_partial`.
- Add high-level `mc.BlockBuffer` syntax over the packed ABI.
- Optimize the BlockAttnRes emitter with parallel reductions and better D
  tiling.
