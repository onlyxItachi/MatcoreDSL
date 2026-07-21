# Codex Memory — 2026-04-23

## Current baseline

- Active repo: `/home/hamza-usta/MatcoreDSL`
- Toolchain currently pinned in `CMakeLists.txt` to MLIR/LLVM `18.1.3` and `C++20`.
- LLVM `22.1.3` and `C++23` are available on the machine, but not adopted yet. No migration has been started because the current blocker is functional fusion coverage, not an obvious LLVM 18 limitation.
- Latest update in this memory file: `2026-04-26`.

## Completed work before this note

### Frontend/runtime contract repair

- `@mc.kernel` now rejects bare helper names and expects namespace-qualified helpers.
- `get_config()` now reflects env-overlaid effective config.
- `mc.to_device()` validates target usage, preserves logical dtypes, and rejects quantized wrappers that previously lost metadata.
- `@mc.fused` now preserves logical dtypes better, accepts logical tensor views, and aligns traced/output dtype handling with runtime support.
- Native graph dtype parsing was aligned with the advertised dtype surface.

### Graph-mode and plan fixes

- Graph-mode plans now compile as a distinct variant.
- Split-K is disabled for graph-mode NVIDIA lowering to avoid capture-time workspace allocation during CUDA graph capture.
- `create_plan` / `execute_plan` now freeze and validate quantization metadata so replay with different scale/zero-point is rejected.
- Quantized `DeviceTensor` inputs are explicitly rejected in both Python and native binding paths.

## Benchmarking state

### Benchmark harness

- Added wrapper: `tests/run_complete_benchmarks.py`
- Main report directory:
  - `benchmark_reports/complete_20260423_182039/`
  - includes `summary.md`, `summary.json`, per-script stdout/stderr logs, and `notes.md`

### Important benchmark findings

- NVIDIA plain matmul / graph-mode is much healthier than fused paths.
- Large-size graph replay is near PyTorch parity around `1024x1024`.
- Power-of-two sweep is competitive or better at some mid/large sizes (`2048`, `4096`), then trails slightly again at larger sizes.
- Fused benchmark families A/B/C are still far behind PyTorch in the current benchmark scripts.

### Known benchmark issues and root causes

- Initial `benchmark_device_resident.py` failure in the wrapper was a harness `PYTHONPATH` issue; manual rerun succeeded after correcting that.
- AMD iGPU benchmarking reproduces a real backend lowering failure:
  - illegal residual `builtin.unrealized_conversion_cast`
  - seen in `benchmark_gladiator.py --arena amd-igpu`
  - also reproduced in `benchmark_massive.py --targets ... amd-igpu`

## Current fusion direction

Family A binary epilogues are now implemented on the current LLVM 18 toolchain.

### What was added

- `src/fusion_emitter.cpp` now generates Family A entry functions with:
  - graph input order preserved
  - output tensor appended last
- Family A epilogue metadata now uses richer per-op dictionaries via:
  - `matcore.fusion_epilogue_ops`
- The epilogue pass now supports:
  - unary epilogues
  - binary epilogues whose boundary operand is a graph input
  - correct operand ordering for non-commutative ops like `sub`
- The epilogue pass now also has an in-launch fast path for the non-tagged
  thread-mapped Family A route:
  - scans output tile views rooted at the final output tensor
  - clones matching sibling views for boundary tensors
  - applies the epilogue on the last K-iteration inside the first launch
  - avoids emitting a second epilogue launch for the common `f32` tiled path
- Backward compatibility remains for older unary metadata:
  - `matcore.fusion_epilogue_kinds`

### Current constraints

- Still staying on LLVM `18.1.3` and `C++20`.
- Binary Family A epilogues currently require:
  - exact-shape rank-2 boundary tensors
  - boundary dtype matching the final output dtype
- This patch does not yet address:
  - Family B binary glue nodes
  - Family C softmax/attention expansion
  - broader broadcasting semantics

### New validation coverage

- `tests/test_family_a.py` now includes end-to-end checks for:
  - `matmul + bias`
  - `relu(matmul + bias)`
  - `bias - matmul`
- Existing tracer and fusion-analysis tests still pass after the emitter change.
- Additional quick probes after the in-launch fast path:
  - `gemm_add_relu 256x128x256`: `3.558 ms`, `max_err=0.000013`
  - `bias_minus_gemm 256x128x256`: `2.467 ms`, `max_err=0.000013`
  - This is a modest improvement over the earlier fallback-only probe
    (`3.700 ms` and `2.557 ms` respectively).
- Follow-up fix for the small single-tile Family A route:
  - `gemm_exp(32x16x32)` now rewrites the epilogue inside the primary launch
    with no separate epilogue launch.
  - Focused probe: `max_err=0.000000119`
  - Larger binary-chain regression probe:
    `gemm_add_relu 256x128x256`: `2.345 ms`, `max_err=0.000011`
- Family B binary glue is now implemented for exact-shape rank-2 graph-input
  boundary tensors:
  - `(A @ B + bias) @ W`
  - `(bias - (A @ B)) @ W`
  - operand order is preserved for non-commutative glue ops
- Fusion pipeline stabilization follow-up:
  - emission now validates the supported surface before lowering, so
    unsupported `elementwise -> matmul` and generic tile-chain plans are
    rejected explicitly instead of being misrouted
  - Family C now binds `Q/K/V` by graph value id instead of assuming argument
    order `(Q, K, V)`
  - Family C now matches frontend semantics for `mc.softmax(Q @ K.T) @ V`
    by using unscaled scores; scaled-dot-product attention must be represented
    explicitly in the graph later
  - Family A epilogue lowering records route metadata via
    `matcore.fusion_epilogue_strategy` and `matcore.fusion_launch_count`
  - new tests cover Family C swapped arguments, rectangular scores, different
    value width, unscaled softmax semantics, and explicit unsupported-pattern
    rejection
- Family C tiled-v1 performance follow-up:
  - the emitter now caches each `Q @ K.T` chunk score in workgroup memory and
    reuses it for online-softmax accumulation instead of recomputing the dot
    product
  - Family C value tiling is clamped upward to `Dtile=64` when the output width
    permits it, while preserving one launch and unscaled softmax semantics
  - analysis shared-memory accounting now includes both `accum_tile` and
    `score_tile`, and scoring gives Family C a deterministic wider-value-tile
    preference
  - the disk-cache version was bumped to avoid reusing stale kernels generated
    by the previous Family C emitter
  - native compilation stats now expose `fusion_launch_count`,
    `family_c_strategy`, and `family_c_dtile` for route assertions
  - benchmark deltas:
    - `tests/bench_attention.py`: `64x64x32` improved from `0.893 ms` to
      `0.504 ms`, `128x128x64` from `2.875 ms` to `1.473 ms`
    - `tests/bench_fusion_fair.py`: Family C `256x256x128` improved from
      `7.641 ms` to `3.385 ms`

### Fusion emitter structure refactor — 2026-04-26

The fusion emitter was split by responsibility without changing the supported
surface or migrating toolchains. The intent was to make the fusion pipeline
easier to continue in parallel: common validation/helpers are isolated, each
fusion family has a separate emitter unit, and the epilogue pass is no longer
embedded inside the dispatcher file.

New/changed files:

- `src/fusion_emitter.cpp`
  - now only validates the plan and dispatches to `emitFamilyA/B/C`
- `src/fusion_emitter_internal.h`
  - internal shared declarations for emitter helpers and epilogue metadata
- `src/fusion_emitter_common.cpp`
  - common dtype/elementwise/cast helpers and emitter-surface validation
- `src/fusion_emitter_family_a.cpp`
  - Family A matmul plus epilogue lowering
- `src/fusion_emitter_family_b.cpp`
  - Family B matmul, glue, matmul lowering
- `src/fusion_emitter_family_c.cpp`
  - Family C attention/online-softmax lowering
- `src/fusion_epilogue_pass.cpp`
  - `CreateFusionEpiloguePass()` implementation and epilogue rewrite pass
- `CMakeLists.txt`
  - adds the new translation units to `MATCORE_NATIVE_SOURCES`

Validation after the split:

- `ninja -C /home/hamza-usta/MatcoreDSL/build-review`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_family_a.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_family_b.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_family_c.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_fusion_contracts.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_tracer.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_fusion_analysis.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_frontend_contract.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_graph_mode.py`

Notes from validation:

- All listed tests passed after the split.
- `tests/test_graph_mode.py` still prints the known
  `Could not compile mgpuMemset32: Symbols not found: [ _mlir_mgpuMemset32 ]`
  warning, but the test completes successfully.
- JIT-linked GPU tests still print linker `DT_TEXTREL` warnings from cached
  kernel objects; those warnings predate this refactor and did not block tests.
- CMake/ninja regenerated tracked `build-review/*` files because the source
  list changed.
- The project remains on LLVM/MLIR `18.1.3` and C++20 after this work.

### Family C block-cooperative attention follow-up — 2026-04-26

A GPT-5.4-mini xhigh co-pilot profiled the existing Family C path and wrote:

- `benchmark_reports/family_c_attention_profile_20260426.md`

Baseline evidence:

- Nsight Compute CLI `2026.1.0.0` is available.
- GPU: RTX 4060 Laptop.
- Baseline `128x128x64` Family C kernel was about `1.46 ms` in NCU.
- Baseline NCU showed only about `1.02%` SM throughput, about `5.18%`
  memory throughput, `8` CTAs of `32` threads, and about `2%` achieved
  occupancy.
- The conclusion was clear: the kernel was under-occupied and serialized inside
  the scalar online-softmax path; LLVM version was not the limiting factor.

Implemented follow-up:

- `src/fusion_emitter_family_c.cpp`
  - score generation is now distributed over the `Br x Bc` score tile
  - online-softmax row state (`m`, `l`, correction scale) lives in workgroup
    memory
  - V accumulation and final stores are distributed over the `Br x Dtile`
    output tile
  - route label is now `score_cached_block_coop_dtile64`
- `src/fusion_analysis.cpp`
  - shared-memory accounting now includes Family C row state buffers
- `src/cache_manager.h`
  - disk cache version bumped to
    `matcore-phase4-cache-v6-family-c-block-coop`
- `tests/test_family_c.py`
  - route assertion accepts the new block-cooperative Family C strategy
- `tests/ncu_profile_family_c.py`
  - focused Family C NCU harness using device-resident inputs and CUDA
    profiler start/stop

Post-change results:

- `tests/bench_attention.py`
  - `128x128x64` improved from roughly `1.48-1.73 ms` to about
    `0.83-0.90 ms` across observed runs
- Focused `tests/bench_fusion_fair.py` Family C:
  - `128x128x64`: `1.416 ms` -> `0.765 ms`
  - `256x256x128`: `3.905 ms` -> `2.076 ms`
- Post-change NCU for `128x128x64`:
  - duration about `597.6 us`
  - registers/thread down from `38` to `31`
  - static shared memory/block about `5.31 KB`
  - achieved occupancy still about `2.08%`

Validation after this change:

- `ninja -C /home/hamza-usta/MatcoreDSL/build-review`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_family_c.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_fusion_analysis.py`
- `/home/hamza-usta/MatcoreDSL/.venv/bin/python tests/test_fusion_contracts.py`

Remaining Family C direction:

- The cooperative rewrite is worth keeping as the new scalar baseline.
- It does not solve the fundamental launch geometry problem: `128x128x64` still
  launches only `8` CTAs.
- The next major performance step should be a real FlashAttention/MMA-style
  Family C path or another decomposition that increases row/CTA parallelism
  without excessive QK recomputation.

### Residuals

- The broader backend limitations remain unchanged:
  - Family C is now score-cached and wider-tiled, but still not an
    MMA/FlashAttention performance path
  - AMD lowering still reproduces the `builtin.unrealized_conversion_cast`
    failure on the benchmark path

## Existing dirty-worktree context

These files were already modified in the worktree before continuing fusion:

- `include/matcore/mlir_engine.h`
- `include/matcore/plan.h`
- `matcore/__init__.py`
- `matcore/config.py`
- `matcore/device_tensor.py`
- `matcore/frontend.py`
- `src/bindings.cpp`
- `src/cache_manager.cpp`
- `src/cache_manager.h`
- `src/fusion_analysis.cpp`
- `src/jit_runner.cpp`
- `src/lowering_pipeline.cpp`
- `src/mlir_engine.cpp`
- `tests/bench_pow2_extended.py`
- `tests/test_graph_mode.py`
- `tests/test_tracer.py`
- generated `build-review/*`

New files already added during this review cycle:

- `tests/run_complete_benchmarks.py`
- `tests/test_frontend_contract.py`
- `tests/test_fusion_contracts.py`
- `tests/test_plan_quantization.py`
- `benchmark_reports/complete_20260423_182039/*`
- `src/fusion_emitter_internal.h`
- `src/fusion_emitter_common.cpp`
- `src/fusion_emitter_family_a.cpp`
- `src/fusion_emitter_family_b.cpp`
- `src/fusion_emitter_family_c.cpp`
- `src/fusion_epilogue_pass.cpp`
- `tests/ncu_profile_family_c.py`
- `benchmark_reports/family_c_attention_profile_20260426.md`

## Immediate next step after this note

- Treat the fusion emitter split as the new working structure; avoid rebuilding
  large logic inside `src/fusion_emitter.cpp`.
- Family C should now continue from the block-cooperative scalar baseline toward
  FlashAttention/MMA-style lowering; AMD lowering remains the other major
  backend gap.
- Benchmark the Family A and Family B binary paths separately from the older
  unary suites so future regressions are visible.
- If LLVM `22.1.3` migration is attempted later, do it as a separate toolchain
  compatibility pass after the current LLVM 18 fusion behavior is preserved.

## 2026-04-29 Family C Restore And Rejected Experiments

Objective: continue Family C optimization while keeping the fusion pipeline
correct and benchmarkable.

Experiments attempted:

- Score-cache padding changed the score tile from `Br x Bc` to `Br x (Bc + 1)`.
  It passed correctness but regressed the largest tracked benchmark:
  `128x128x64` measured about `1.212 ms` versus `0.966 ms` after restoration.
- Naive row-subtile geometry was attempted to increase CTA count for underfilled
  shapes. It introduced a native SIGSEGV in the first Family C smoke test and
  was reverted.

Restored state:

- Family C is back on the block-cooperative scalar route:
  `score_cached_block_coop_dtile64`.
- The cache namespace is now `matcore-phase4-cache-v9-family-c-block-coop` to
  avoid stale kernels from the failed experiments.
- Geometry-only stats fields (`family_c_br`, `family_c_bc`,
  `family_c_grid_m`, `family_c_grid_d`) were removed.
- Route observability stats remain useful and are kept:
  `fusion_launch_count`, `family_c_strategy`, and `family_c_dtile`.

Validation after restoration:

- `ninja -C /home/hamza-usta/MatcoreDSL/build-review`
- `tests/test_family_c.py`
- `tests/test_fusion_analysis.py`
- `tests/test_fusion_contracts.py`
- `tests/test_family_a.py`
- `tests/test_family_b.py`

Benchmark snapshot from `tests/bench_attention.py`:

- `16x16x8`: `0.170 ms`
- `32x32x16`: `0.187 ms`
- `64x64x32`: `0.363 ms`
- `128x128x64`: `0.966 ms`

NCU snapshot for `128x128x64` after restoration:

- grid/block: `8 x 32`
- registers/thread: `31`
- static shared memory/block: `5.31 KB`
- waves/SM: `0.02`
- memory busy: `2.54%`
- L1/TEX hit rate: `97.12%`
- L2 hit rate: `79.74%`
- local/shared spilling: `0`

Conclusion:

- The current bottleneck is still work decomposition and launch underfill, not a
  simple memory-layout issue.
- Do not retry score-cache padding as the next step.
- Do not retry naive row-subtiling without isolating it behind a feature flag
  and adding IR/kernel debug checks.
- Next performance work should be a safer split-row or FlashAttention/MMA-style
  Family C path that increases CTA parallelism without corrupting operand
  bindings or duplicating too much QK work.
