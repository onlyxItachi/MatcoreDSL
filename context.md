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
- Region verifier:
  - `include/matcore/region_verifier.h`
  - `src/region_verifier.cpp`
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
- `ValidateRegionIR` now runs before MLIR emission and rejects malformed value
  references, bad topo order, invalid output count, dtype/rank/shape mismatch,
  invalid BlockAttnRes attrs, and runtime tensor descriptor mismatch.
- `RegionMlirEmitter` supports one `block_attn_res` op and emits one explicit
  NVIDIA `gpu.launch`.
- BlockAttnRes score setup is now CTA-parallel: RMS and dot scores use
  thread-strided `D` loops plus shared-memory tree reductions. The old
  correctness-first path computed scores in `tid == 0` and caused barrier
  stalls / lane underutilization.
- `RegionMlirEmitter` now consumes topo-ordered nodes and reports a clear
  multi-op lowering gap after verifier acceptance instead of assuming
  `region.nodes.front()`.
- The lowering pipeline routes `matcore.kernel_type = "region_*"` through the
  explicit GPU-launch lowering path instead of matmul mapping.
- Executor generic packed invocation now supports rank 1-4 memref descriptors,
  which covers `query[D]`, `partial/out[B,T,D]`, and
  `blocks[MAX_BLOCKS,B,T,D]`.

## Verification Checklist

- Passed: `ninja -C /home/hamza-usta/MatcoreDSL/build-review`
- Passed: `/usr/bin/python3 -m pytest tests/test_region_ir.py tests/test_block_attn_res.py -q`
  (covers direct intrinsic, `@mc.jit` native smoke, invalid RegionIR verifier
  cases, and the multi-op lowering gap)
- Passed: `/usr/bin/python3 -m pytest tests/test_frontend_contract.py tests/test_fusion_contracts.py -q`
- Passed: `/usr/bin/python3 -m pytest tests/test_devtensor_fused.py -q`
- Nsight Compute on `[16,8,1024,128]`, `block_count=12`, `D=128`:
  old kernel duration ~5.51 ms, active threads/warp ~3.3, barrier stall ~73.8%;
  optimized kernel duration ~1.17 ms, active threads/warp ~31.7,
  compute/memory throughput ~87.5%.
- Large one-forward benchmark after reduction rewrite:
  `[8,4,512,128]` 0.284 ms, `[8,8,1024,128]` 0.681 ms,
  `[16,8,1024,128]` 1.597 ms, `[8,4,1024,256]` 0.409 ms,
  `[16,8,2048,128]` 2.432 ms. Correctness stayed below ~1.5e-6 max error.
- Project-local Python environments exist at `.venv` and `.venv_gladiator`.
  `.venv_gladiator` was repaired to `torch==2.11.0+cu130`, CUDA 13.0,
  NumPy 2.4.4 for attention-residual benchmarks.
- Project envs do not currently have `pytest`; use `/usr/bin/python3 -m pytest`
  for pytest suites unless installing pytest into the project env.
- `tests/test_family_a.py`
- `tests/test_family_b.py`
- `tests/test_family_c.py`

## Notes

- Do not retry the failed Family C score-padding or naive row-subtile
  experiments as part of this work.
- Keep RegionV1 additive; no existing `graph_v2` schema migration in v1.
- BlockAttnRes v1 is still not fully optimized, but the first NCU-guided
  reduction rewrite removed the worst `tid == 0` bottleneck.
- Next architecture step is generalizing RegionV1 beyond one intrinsic:
  true multi-op lowering, control-flow nodes, and symbolic/runtime scalar attrs.
