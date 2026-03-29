# MatCore Hardware Blueprints

## Purpose

This file captures hardware-facing matmul blueprints that MatCore can lower toward.
It is intentionally stricter than a feature wishlist: each blueprint should describe
the memory movement, fragment loading, tensor-core instruction family, and writeback
shape that the compiler must preserve.

The immediate baseline is NVIDIA `sm_89` FP16 matmul, because that is the cleanest
warp-level path already reflected in the repo's NVIDIA notes:

- `vector.contract -> nvgpu.mma.sync -> nvvm.mma.sync`
- static warp-level `mmaShape = [16, 8, 16]`
- shared-memory staging chosen to feed `ldmatrix`
- vectorized epilogue/writeback rather than scalar stores

Everything below should be read as a lowering contract, not just a hardware summary.

## NVIDIA `sm_89` FP16 Baseline

### One-Line Flow

```text
global memory
  -> vectorized cp.async global->shared staging
  -> shared-memory tile layout padded/swizzled for ldmatrix
  -> warp-level ldmatrix fragment loads
  -> mma.sync.m16n8k16 with FP16 inputs and FP32 accumulators
  -> register epilogue
  -> vectorized global writeback
```

### Baseline Kernel Contract

- Architecture: NVIDIA Ada / `sm_89`
- Execution width: one warp is 32 threads and remains a compile-time constant
- Tensor-core family: warp-level `mma.sync`, not warpgroup `wgmma`
- Canonical MMA shape: `m16n8k16`
- Input element type: `f16`
- Preferred accumulator type: `f32`
- Output type: `f16` or `f32`
- Shared-memory loader: `ldmatrix` with `.b16` fragments
- Global/shared movement primitive: `cp.async` on `sm_80+`, used here as the `sm_89` staging baseline

### Stage 1: Vectorized Global -> Shared Async Copy

The baseline CTA should stage operand tiles from global memory into shared memory
with vectorized `cp.async` transactions instead of scalar loads followed by scalar
shared stores.

Required properties:

- Treat 16-byte per-copy chunks as the default staging width when alignment permits.
- Organize global tiles so each warp issues coalesced async copies over contiguous
  segments of A and B.
- Use multistage buffering so one K-slice can be copied while the previous slice is
  being consumed by `ldmatrix` and `mma.sync`.
- Close each staging batch with `cp.async.commit_group`, and wait with
  `cp.async.wait_group` before the corresponding shared tile becomes visible to the
  compute stage.
- Keep the async copy policy architecture-gated to `sm_80+`; `sm_89` inherits that
  contract directly.

Compiler implication:

- MatCore should model this stage as an explicit async shared-memory promotion step,
  not as a generic "load some values and hope later passes rediscover tensor-core
  legality."

### Stage 2: Shared-Memory Layout For `ldmatrix`

The async-copy stage is only half the design. The shared-memory tile must be laid
out so warp lanes can collectively feed `ldmatrix` without bank-conflict-heavy or
shape-unsafe address arithmetic.

Required properties:

- Stage K in quanta of 16 FP16 elements to match `m16n8k16`.
- Store A and B in shared memory with row strides and padding chosen for 8x8
  `ldmatrix` row addressing.
- Preserve a layout that lets each lane contribute row start addresses exactly as
  `ldmatrix` expects.
- If B is staged in a row-major shared layout for better global coalescing, load it
  with the transposed `ldmatrix` form so the tensor-core op still sees the required
  `row.col` contract.

Practical rule:

- Shared memory is not a generic cache here. It is a fragment staging area whose
  layout is dictated by the next instruction, `ldmatrix`.

### Stage 3: Shared -> Register Fragment Load With `ldmatrix`

For the FP16 baseline, fragment loads should use `ldmatrix` as the warp-cooperative
bridge from shared memory into the per-lane register fragments consumed by
`mma.sync`.

Relevant fragment contract from the local LLVM 18 NVGPU surface:

- A-side thread fragment shape: `vector<4x2xf16>`
- B-side thread fragment shape: `vector<2x2xf16>`
- Accumulator fragment shape: `vector<2x2xf32>`

Required properties:

- Use `nvgpu.ldmatrix` / `nvvm.ldmatrix` as the lowering target for FP16 shared
  fragments.
- Prefer the `numTiles = 4` path for A-side fragments when it matches the selected
  warp tile decomposition.
- Use the transposed form for B whenever the shared tile is staged in the opposite
  logical orientation from the `mma.sync.row.col` contract.
- Keep `ldmatrix` legality static: warp shape, tile count, transpose flag, and
  source element type must all be known before NVVM lowering.

### Stage 4: Register -> Tensor Core Compute With `mma.sync`

Once fragments are in registers, the warp issues warp-synchronous tensor-core MMA.

Baseline instruction family:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

Required properties:

- Keep the MMA family static per kernel. `m16n8k16` is not a runtime choice.
- Prefer FP32 accumulation even when the output tensor is FP16.
- Unroll or pipeline the K loop in units of the MMA K-depth, which is 16 for this
  baseline.
- Treat the register fragment map as part of the ABI between the `ldmatrix` stage
  and the `mma.sync` stage.

### Stage 5: Register Epilogue And Vectorized Writeback

The tensor core produces register-resident accumulators. The baseline epilogue
should preserve vector width through the store path instead of spilling into scalar
element stores.

Required properties:

- Keep accumulator fragments in registers until the final output conversion step.
- For FP32 output, prefer a 16-byte store shape per lane when the per-lane fragment
  ownership makes that possible.
- For FP16 output, convert and pack in registers before issuing vectorized stores.
- Write contiguous output strips so the store path is coalesced across the warp.
- Keep edge handling predicate-based at the store boundary; do not break the main
  interior tile into scalar code just to accommodate tails.

### What Must Stay Static In The FP16 Baseline

- Warp size: `32`
- MMA shape: `16x8x16`
- Per-warp fragment ownership
- Shared-memory operand layout
- Whether B uses transposed `ldmatrix`
- Async-copy stage depth
- Vectorized writeback width

### What Can Stay Dynamic

- Grid dimensions derived from runtime matrix sizes
- Macro tile count in M and N
- Tail predicates for partial tiles
- Runtime stride values, as long as they preserve the alignment and layout
  preconditions of the chosen blueprint

## NVIDIA INT8 Blueprint Requirements

INT8 should be treated as "same outer skeleton, different fragment contract," not as
"FP16 with a type substitution."

### What Carries Over From FP16

- CTA-level vectorized global -> shared staging should still use async copy where legal.
- Shared memory remains the operand exchange point between global memory and the warp.
- Warp-level tensor-core execution still uses `mma.sync`, not `wgmma`, on `sm_89`.
- Vectorized writeback still matters, because INT8 tensor-core outputs are naturally
  register-packed `s32` accumulators.

### What Changes For INT8

- Canonical MMA family becomes `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.
- The K-depth doubles from 16 to 32.
- Output type is currently fixed to `int32` in MatCore.
- Zero-point and scale handling must be treated as an explicit epilogue or pre/post
  correction policy, not as an accidental side effect of the MMA itself.

### Critical `sm_89` Constraint

The local LLVM 18 NVVM surface used by this repo exposes `ldmatrix` only for `.b16`
fragments. That means the FP16 baseline's `ldmatrix` stage does not directly carry
over to Ada INT8.

Practical consequence:

- Do not assume an Ada INT8 blueprint can reuse `nvgpu.ldmatrix` unchanged.
- INT8 needs a dedicated shared -> register fragment loader and packer for `s8`
  inputs on `sm_89`.
- The loader must produce the `.b32` register grouping expected by the integer
  `mma.sync` forms.

### INT8 Blueprint Checklist

- Define the exact shared-memory packing for `s8` tiles.
- Define the non-`ldmatrix` shared -> register fragment load path for `sm_89`.
- Keep K-slicing at 32 elements per tensor-core step.
- Preserve `int32` accumulators through the epilogue.
- Decide whether requantization is outside the kernel, a fused epilogue, or a
  separate blueprint.
- Keep vectorized global writeback for the `int32` result tile.

## NVIDIA FP8 Blueprint Requirements

FP8 is future work for MatCore and should not be described as a simple extension of
the `sm_89` FP16 baseline.

### Current Project Gate

The repo currently enforces all of the following:

- FP8 is NVIDIA-only.
- FP8 requires native NVIDIA FP8 tensor-core support.
- MatCore currently gates that support to `sm_90+`.
- MatCore does not yet implement the dedicated native NVIDIA FP8 lowering path.

That means there is no current "FP8 on `sm_89`" MatCore blueprint.

### Why FP8 Needs Its Own Blueprint

- The current repo policy ties FP8 to `sm_90+` `wgmma`, not to the `sm_89`
  warp-level `mma.sync` baseline.
- `wgmma` is descriptor-based and warpgroup-based, so its memory flow is different
  from the `ldmatrix`-fed warp-level baseline.
- FP8 introduces format policy, scale policy, and accumulator policy questions that
  do not exist in the plain FP16 path.

### Minimum Requirements For A Real MatCore FP8 Path

- Architecture gate: require `sm_90+`.
- Lowering family: `nvgpu.warpgroup.*` / `nvvm.wgmma.*`, not the current `sm_89`
  `nvgpu.mma.sync` baseline.
- Shared-memory contract: descriptor-friendly layout suitable for warpgroup MMA.
- Data-format contract: define exactly which FP8 formats are accepted first.
  Today that should start with NVIDIA `e4m3fn`; `e5m2` can be added later.
- Type-policy contract: never alias NVIDIA FP8 formats with AMD FNUZ FP8 formats.
- Scale-policy contract: define whether FP8 tensors carry explicit scales, block
  scales, or a separate quantization descriptor. Raw bytes without scale semantics
  are not a complete FP8 blueprint.
- Accumulator/output contract: keep `f32` accumulation as the safe baseline until a
  narrower accumulator is proven correct and worthwhile.
- Epilogue contract: define conversion, saturation, and vectorized writeback rules
  explicitly.

### Short Rule

FP8 is not "reuse the FP16 pipeline and swap the dtype." It is a separate lowering
route with separate architecture gates, memory descriptors, and type semantics.

## AMD Flow Placeholder

This section is intentionally a placeholder. The repo has AMD routing notes, but not
yet a single hardware blueprint at the same level of precision as the NVIDIA `sm_89`
baseline above.

Needed before this section is considered complete:

- Choose the first-class AMD baseline family:
  `amdgpu.mfma` on CDNA or `amdgpu.wmma` on RDNA.
- Define the global -> LDS staging policy.
- Define the LDS -> register fragment load policy.
- Define the exact matrix-core instruction family per chosen `gfx` target.
- Define FP16/BF16 and INT8 accumulator/output contracts.
- Define the AMD FP8 type policy explicitly, including `FNUZ` encoding differences.

## NPU Flow Placeholder

This section is intentionally a placeholder. The repo currently treats NPU paths as
external-toolchain integrations rather than as an in-tree LLVM 18 lowering route.

Needed before this section is considered complete:

- Name the target NPU family and toolchain explicitly.
- Define the memory hierarchy:
  host DRAM -> DMA -> on-chip SRAM / scratchpad -> tensor engine -> writeback.
- Define the matmul primitive granularity and accumulator precision.
- Define whether tiling is compiler-owned, runtime-owned, or delegated to a vendor
  graph compiler.
- Define the quantization and low-precision metadata contract up front.
- Define how MatCore hands off to the external compiler/runtime boundary.

## Bottom Line

The NVIDIA `sm_89` FP16 path is the first complete MatCore hardware blueprint:
vectorized async global -> shared copy, `ldmatrix` fragment load, warp-level
`mma.sync`, and vectorized writeback.

INT8 should reuse the same outer kernel skeleton but must introduce a distinct
shared -> register fragment loader on `sm_89`.

FP8 should remain a separate future route gated to `sm_90+` WGMMA until MatCore has
an explicit descriptor-based FP8 lowering path.
