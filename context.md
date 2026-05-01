# MatcoreDSL Context Notepad

## Current Objective

Implement runtime-JIT structured tensor programs by adding `region_v1` IR,
`@mc.jit`, and a first packed Block Attention Residual intrinsic:
`mc.block_attn_res(blocks, partial, query, block_count=..., has_partial=...)`.

This is the architectural bridge from hard-coded Family A/B/C fusion toward a
programmable tensor-algorithm compiler. The first target is Attention
Residuals / Block AttnRes style depth attention over packed block history.

## Accepted Decisions

- Keep existing `@mc.fused` and `KernelGraphIR` unchanged.
- Add a separate `@mc.jit` path for structured region programs.
- Add `KernelIRVersion::kRegionV1` and `RegionIR` beside `KernelGraphIR`.
- V1 only supports a first-class `BlockAttnResOp`.
- V1 ABI uses packed tensor history:
  - `blocks`: `[MAX_BLOCKS, B, T, D]`
  - `partial`: `[B, T, D]`
  - `query`: `[D]`
  - output: `[B, T, D]`
- `block_count`, `has_partial`, and `eps` are compile-time-specialized attrs.
- V1 dtype is `float32` only.
- V1 target is NVIDIA GPU only.

## Key Code Anchors

- Python frontend and exports:
  - `matcore/frontend.py`
  - `matcore/__init__.py`
- IR types:
  - `include/matcore/kernel_ir.h`
- Native parsing:
  - `src/bindings.cpp`
- Cache key:
  - `src/cache_manager.cpp`
- MLIR dispatch:
  - `src/mlir_engine.cpp`
- New region emitter:
  - `include/matcore/region_emitter.h`
  - `src/region_emitter.cpp`
  - `src/region_emitter_block_attn_res.cpp`
- Runtime invocation rank support:
  - `src/executor.cpp`
- Build source list:
  - `CMakeLists.txt`

## Implementation State

- RegionV1 is implemented as an additive path; existing `graph_v2` and
  `@mc.fused` stay unchanged.
- Python exposes `mc.jit` / `@mc.jit` and direct `mc.block_attn_res(...)`.
- Native parser/cache/lowering understand `version == "region_v1"`.
- `RegionMlirEmitter` supports one `block_attn_res` op and emits one explicit
  NVIDIA `gpu.launch`.
- The lowering pipeline routes `matcore.kernel_type = "region_*"` through the
  explicit GPU-launch lowering path instead of matmul mapping.
- Executor generic packed invocation now supports rank 1-4 memref descriptors,
  which covers `query[D]`, `partial/out[B,T,D]`, and
  `blocks[MAX_BLOCKS,B,T,D]`.

## Verification Checklist

- Passed: `ninja -C /home/hamza-usta/MatcoreDSL/build-review`
- Passed: `/usr/bin/python3 -m pytest tests/test_region_ir.py tests/test_block_attn_res.py -q`
  (covers direct intrinsic and `@mc.jit` native smoke)
- Passed: `/usr/bin/python3 -m pytest tests/test_region_ir.py tests/test_frontend_contract.py -q`
- Passed: `/usr/bin/python3 -m pytest tests/test_devtensor_fused.py -q`
- Passed: `/usr/bin/python3 -m pytest tests/test_fusion_contracts.py -q`
- Project-local Python environments exist at `.venv` and `.venv_gladiator`.
  Both are Python 3.12.3. `.venv` imports local MatCore and native BlockAttnRes
  smoke passed with max error `0.0`.
- Neither `.venv` nor `.venv_gladiator` currently has `pytest`; use
  `/usr/bin/python3 -m pytest ...` for pytest suites unless installing pytest
  into the project env.
- `tests/test_family_a.py`
- `tests/test_family_b.py`
- `tests/test_family_c.py`

## Notes

- Do not retry the failed Family C score-padding or naive row-subtile
  experiments as part of this work.
- Keep RegionV1 additive; no existing `graph_v2` schema migration in v1.
- BlockAttnRes v1 is correctness-first: float32, compile-time `block_count`,
  compile-time `has_partial`, packed layout, and one CTA per `(B,T)` row.
- Next architecture step is generalizing RegionV1 beyond one intrinsic:
  multiple region ops, control-flow nodes, symbolic scalar attrs, and a real
  region verifier before lowering.
