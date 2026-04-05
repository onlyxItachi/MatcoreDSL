# FP8 WGMMA in MLIR 18.1.3 on NVIDIA Hopper (sm_90+) 

This note summarizes what MLIR 18.1.3 can and cannot do for Hopper FP8 matmul using warp-group MMA (`wgmma`).

## Bottom line

- **MLIR 18.1.3 has NVGPU/NVVM pieces for Hopper WGMMA and FP8 types.**
- **It does not have a stock upstream `linalg.matmul -> nvgpu.warpgroup.mma` rewrite for FP8.**
- The practical upstream path in 18.1.3 is:
  - use Transform dialect for **TMA copies / async grouping / pipelining**,
  - then **manually emit** or **custom-rewrite** `nvgpu.warpgroup.*` ops,
  - then run `-convert-nvgpu-to-nvvm`.
- In practice, **FP8 WGMMA in MLIR 18.1.3 is only viable with `f32` accumulation / store**, not `f16` accumulation, due to verifier limitations.

---

## 1. `nvgpu.warpgroup.mma` support in MLIR 18.1.3

### What the op is

`nvgpu.warpgroup.mma` is the Hopper-facing high-level op that lowers to one or more `nvvm.wgmma.mma_async` instructions, wrapped with:

- `nvvm.wgmma.fence.aligned`
- `nvvm.wgmma.commit.group.sync.aligned`
- `nvvm.wgmma.wait.group.sync.aligned`

It consumes:

- `!nvgpu.warpgroup.descriptor<tensor = memref<... , 3>>` for A and B
- `!nvgpu.warpgroup.accumulator<fragmented = vector<MxNxt>>` for C/D

### Type combinations accepted by the 18.1.3 verifier

From `mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp`, the verifier nominally accepts:

- `f32 += f16 * f16`
- `f16 += f16 * f16`
- `f32 += tf32 * tf32`
- `f32 += bf16 * bf16`
- `f16 += bf16 * bf16`
- `f32 += fp8 * fp8`
- `f16 += fp8 * fp8`
- `s32 += i1 * i1`
- integer paths also exist, but comments/code are inconsistent in 18.1.3 and are not the focus here

For FP8, the verifier accepts builtin MLIR types:

- `f8E4M3FN`
- `f8E5M2`

and does **not** require A and B to be the same FP8 flavor. Mixed `e4m3/e5m2` passes the datatype check.

### Actual FP8 limitation in 18.1.3

A later verifier check effectively blocks non-`f32` FP8 accumulation:

- if result/accumulator is not `f32`, then only `f16` and `bf16` multiplicands are allowed
- therefore **`f16 += fp8 * fp8` is nominally listed, but rejected in practice**

So for FP8 WGMMA, the practical supported form is:

- **`f32 += f8E4M3FN/f8E5M2 * f8E4M3FN/f8E5M2`**

### Shape rules that matter

The high-level op infers WGMMA shape from descriptor/result types. In 18.1.3 lowering:

- `M = 64` per hardware WGMMA instruction
- `N` must be in the legal Hopper set
- `K` is chosen by multiplicand type:
  - `tf32` -> `k8`
  - `f16`/`bf16` -> `k16`
  - `f8E4M3FN`/`f8E5M2` -> **`k32`**
  - `i1` -> `k256`

For FP8/BF16/F16/TF32, legal `N` values are:

- `8, 16, 24, ..., 256` (step 8)

### Layout restriction

18.1.3 currently only supports:

- **A non-transposed / row-major**
- **B transposed / column-major**

The verifier rejects `transposeA` without `transposeB`.

### Store limitation

`nvgpu.warpgroup.mma.store` in 18.1.3 only accepts:

- **`f32` accumulator/result fragments**

So even if other accumulator types appear nominally supported earlier, the stock store op is `f32`-only.

---

## 2. `float8_e4m3fn` support in NVGPU

In MLIR IR, the builtin type is spelled:

- **`f8E4M3FN`**

not `float8_e4m3fn`.

Important distinction:

- NVGPU does **not** define its own FP8 scalar type
- it consumes MLIR builtin float8 types

In 18.1.3:

- the NVGPU verifier recognizes `Float8E4M3FN` / `Float8E5M2`
- `-convert-nvgpu-to-nvvm` maps them to NVVM WGMMA types:
  - `f8E4M3FN` -> `#nvvm.wgmma_type<e4m3>`
  - `f8E5M2` -> `#nvvm.wgmma_type<e5m2>`

So the scalar type plumbing exists in upstream MLIR 18.1.3 for Hopper WGMMA lowering.

---

## 3. How to lower FP8 matmul in MLIR 18.1.3

### What upstream does **not** provide

There is **no stock upstream Transform op or canonical pass** in 18.1.3 that rewrites:

- `linalg.matmul` -> `nvgpu.warpgroup.mma`

The only built-in matmul rewrite is:

- `transform.nvgpu.rewrite_matmul_as_mma_sync`

which targets **`nvgpu.mma.sync`** (warp-level MMA), not Hopper WGMMA.

### Practical lowering strategy

For FP8 on Hopper in MLIR 18.1.3, the practical route is:

1. **bufferize** the matmul
2. **map tiles to GPU shared memory**
3. use **TMA** for global->shared copies where possible
4. **manually emit** or **custom-rewrite** to:
   - `nvgpu.tma.create.descriptor`
   - `nvgpu.tma.async.load`
   - `nvgpu.warpgroup.generate.descriptor`
   - `nvgpu.warpgroup.mma.init.accumulator`
   - `nvgpu.warpgroup.mma`
   - `nvgpu.warpgroup.mma.store`
5. run **`-convert-nvgpu-to-nvvm`**
6. finish the usual GPU/NVVM/LLVM lowering pipeline

### Minimal op pattern for FP8 Hopper WGMMA

Use:

- A/B tiles in **shared memory** (`memref<..., 3>` / workgroup memory)
- a **tensor-map descriptor** with:
  - `swizzle = swizzle_128b`
  - `interleave = none`
- `nvgpu.warpgroup.generate.descriptor` for A and B
- `nvgpu.warpgroup.mma.init.accumulator -> !nvgpu.warpgroup.accumulator<fragmented = vector<MxNxf32>>`
- `nvgpu.warpgroup.mma ... {transposeB}`
- `nvgpu.warpgroup.mma.store`

For FP8, use **`vector<...xf32>` accumulators/results**, not `vector<...xf16>`.

### Passes to run after WGMMA is in the IR

The tested Hopper/TMA lowering flow in 18.1.3 is roughly:

```text
-transform-interpreter
-convert-nvgpu-to-nvvm
-gpu-kernel-outlining
-convert-scf-to-cf
-convert-nvvm-to-llvm
-convert-vector-to-llvm
-convert-math-to-llvm
-lower-affine
-convert-index-to-llvm=index-bitwidth=32
-convert-arith-to-llvm
-finalize-memref-to-llvm
-convert-func-to-llvm
```

For actual Hopper codegen, attach an NVVM target like:

```text
--nvvm-attach-target="module=<gpu-module> features=+ptx80 chip=sm_90 O=3"
```

Then lower GPU modules to NVVM/LLVM as usual.

---

## 4. Transform dialect sequences that are useful in 18.1.3

### Available NVGPU Transform ops in 18.1.3

Upstream 18.1.3 includes:

- `transform.nvgpu.rewrite_matmul_as_mma_sync`
- `transform.nvgpu.rewrite_copy_as_tma`
- `transform.nvgpu.create_async_groups`
- `transform.nvgpu.pipeline_shared_memory_copies`
- `transform.apply_conversion_patterns.nvgpu.nvgpu_to_nvvm`

### Missing piece

There is **no** upstream:

- `transform.nvgpu.rewrite_matmul_as_warpgroup_mma`
- `transform.nvgpu.rewrite_matmul_as_wgmma`

So a practical Hopper FP8 sequence is:

1. use Transform dialect to find/promote/stage copies
2. rewrite `linalg.copy` to TMA
3. pipeline shared-memory traffic / async groups
4. apply your **custom** rewrite from tiled `linalg.matmul` (or a custom matmul op) to `nvgpu.warpgroup.*`
5. apply NVGPU->NVVM conversion patterns

### Example shape of a practical transform sequence

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %copies = transform.structured.match ops{["linalg.copy"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.rewrite_copy_as_tma %copies
      : (!transform.any_op) -> ()

    %loops = transform.structured.match ops{["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.pipeline_shared_memory_copies failures(propagate) %loops
      { depth = 2, peel_epilogue }
      : (!transform.any_op) -> !transform.any_op

    // Custom step needed here in MLIR 18.1.3:
    // rewrite tiled FP8 matmul -> nvgpu.warpgroup.*

    transform.yield
  }
}
```

If you want a purely upstream path with no custom rewrite, you are limited to **`mma.sync`**, not Hopper WGMMA.

---

## 5. Tile sizes and layouts that are practical on Hopper

### WGMMA tile family

For Hopper WGMMA in MLIR 18.1.3:

- hardware instruction tile is always **`m64`**
- for FP8, the instruction family is **`m64nNk32`** with legal `N` in `8..256` by 8

That means a macro-tile like:

- `128x128x64`

is implemented as multiple WGMMA instructions over `64x128x32` subproblems.

### TMA/shared-memory tiles seen in upstream tests

The 18.1.3 TMA transform/integration tests use shared-memory tiles such as:

- `64x8`
- `8x128`

Those are good reference shapes for A/B staging when feeding Hopper tensor-core code.

### Practical FP8 recommendation

For an FP8 Hopper kernel in MLIR 18.1.3, a conservative practical recipe is:

- stage A/B through TMA into shared memory
- use **`swizzle_128b`**
- use **`interleave = none`**
- choose macro-tiles that decompose into **`m64nNk32`** WGMMA units
- keep B in the transposed/shared-memory layout expected by WGMMA
- use **`f32` accumulators** end-to-end

`nvgpu.warpgroup.generate.descriptor` verifies exactly those layout restrictions in 18.1.3.

---

## 6. Known limitations / bugs in MLIR 18.x FP8 + NVGPU support

### 1. No stock `linalg.matmul -> nvgpu.warpgroup.mma`

This is the biggest practical gap. Upstream 18.1.3 has the low-level pieces, but not the end-to-end rewrite you want for FP8 Hopper GEMM.

### 2. FP8 verifier/lowering mismatch around `f16` accumulation

The verifier nominally allows `f16 += fp8 * fp8`, but a later limitation rejects it. Treat this as:

- **FP8 WGMMA in 18.1.3 => use `f32` accumulation only**

### 3. `warpgroup.mma.store` is `f32`-only

This further pushes practical implementations toward `f32` accumulators/results.

### 4. Descriptor generation is restrictive

`nvgpu.warpgroup.generate.descriptor` currently requires:

- shared-memory memrefs
- `swizzle_128b`
- `interleave = none`

If your TMA/shared-memory layout differs, the stock op rejects it.

### 5. Dynamic shared memory is still awkward in 18.1.3

MLIR issue `#72513` exists because dynamic shared-memory support was still rough around the 18.x timeframe. In practice, older Hopper examples often use awkward workarounds such as zero-sized shared-memory globals.

### 6. NVVM ops lack strong PTX/SM gating

LLVM issue `#69448` is relevant: many NVVM ops do not encode PTX/SM availability, so some invalid target combinations fail late during NVPTX/LLVM lowering instead of at the MLIR level.

### 7. Hopper toolchain caveat: PTXAS may serialize WGMMA unexpectedly

Independent of MLIR, Hopper toolchains have reported `ptxas` warnings about serialized `wgmma.mma_async` pipelines. If generated code looks correct but underperforms, inspect emitted PTX/SASS and `ptxas` warnings before assuming the MLIR IR is wrong.

---

## Recommended implementation plan for MatcoreDSL

If MatcoreDSL wants FP8 Hopper GEMM on MLIR 18.1.3, the most realistic plan is:

1. **Tile and bufferize matmul yourself**
2. use Transform dialect for **TMA copy rewrite** and **shared-memory pipelining**
3. add a **custom rewrite** from your tiled FP8 matmul form to `nvgpu.warpgroup.*`
4. constrain the generated kernel to:
   - FP8 inputs: `f8E4M3FN` / `f8E5M2`
   - accumulator/result: `f32`
   - WGMMA instruction tiles: `m64nNk32`
   - descriptors: `swizzle_128b`, `interleave=none`
   - layout: A row-major, B transposed
5. lower with `-convert-nvgpu-to-nvvm` and an `sm_90` target

If you need a purely upstream, no-custom path in MLIR 18.1.3, use:

- `transform.nvgpu.rewrite_matmul_as_mma_sync`

but that is **not** FP8 Hopper WGMMA.

---

## Primary sources

- MLIR 18.1.3 NVGPU op definitions:  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td
- MLIR 18.1.3 NVGPU verifier logic:  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/lib/Dialect/NVGPU/IR/NVGPUDialect.cpp
- MLIR 18.1.3 NVGPU->NVVM lowering:  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/lib/Conversion/NVGPUToNVVM/NVGPUToNVVM.cpp
- MLIR 18.1.3 NVGPU transform ops:  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/include/mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.td
- MLIR 18.1.3 NVGPU tests:  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/test/Dialect/NVGPU/transform-matmul-to-nvvm.mlir  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/test/Dialect/NVGPU/tmaload-transform.mlir  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/test/Dialect/NVGPU/transform-pipeline-shared.mlir  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/test/Dialect/NVGPU/invalid.mlir  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/test/Conversion/NVGPUToNVVM/nvgpu-to-nvvm.mlir
- MLIR builtin float8 type names:  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/include/mlir/IR/BuiltinTypes.td
- NVVM WGMMA type/shape table:  
  https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-18.1.3/mlir/include/mlir/Dialect/LLVMIR/NVVMOps.td
- LLVM issue on NVGPU/NVVM Hopper gaps:  
  https://github.com/llvm/llvm-project/issues/69448
- Hopper TMA / WGMMA external context:  
  https://pytorch.org/blog/hopper-tma-unit/  
  https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/  
  https://forums.developer.nvidia.com/t/ptxas-mysterious-warning-for-wgmma-mma-async-instruction-serialization/340610
