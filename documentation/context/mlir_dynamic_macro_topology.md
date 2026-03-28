# MLIR 18 Dynamic Macro-Topology Blueprint

## Problem Statement

MatCore currently relies on `transform.gpu.map_forall_to_blocks ... generate_gpu_launch grid_dims = [...]` in [`/home/hamza-usta/MatcoreDSL/src/gpu_mapping.cpp`](/home/hamza-usta/MatcoreDSL/src/gpu_mapping.cpp). That requires host-side C++ to compute macro-grid dimensions early and inject them as static integers into the transform IR.

That design has two failures:

- it couples compile-time host code to runtime matrix sizes
- it can cause large static loop structures and compile-time IR blowups when the macro-topology is baked too early

The fix is a custom C++ pass that builds `gpu.launch` with runtime `memref.dim`-derived grid sizes.

## Exact MLIR 18 Failure Mechanism

The blocker is not hypothetical. MLIR 18 documents the limitation directly.

From the local header:

- [`/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td:247`](#/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td:247)

Relevant behavior of `transform.gpu.map_forall_to_blocks`:

- dynamic `scf.forall` trip counts are currently not supported
- dynamic block dim sizes are currently not supported
- only bufferized `scf.forall` is supported

That matches the observed failure:

- when the top-level `scf.forall` bounds are computed from `memref.dim`
- and MatCore attempts to use `transform.gpu.map_forall_to_blocks`
- the transform rejects the op because it requires a statically sized, normalized `forall`

There is a second related constraint in the same local file:

- [`/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td:128`](#/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td:128)

`transform.gpu.map_nested_forall_to_threads` also takes static `block_dims` attributes. That is fine for MatCore, because block dimensions are micro-constants and should remain static. The dynamic problem is the outer macro-grid.

## Required Architectural Split

### Keep transform dialect for micro-topology

Use transform dialect only where static configuration is appropriate:

- block tile sizes
- thread tile sizes
- workgroup/shared-memory promotion
- tensor-core rewrite preconditions
- thread distribution within a block

### Replace transform dialect for macro-topology

Do not use `transform.gpu.map_forall_to_blocks` to derive the launch grid.

Instead:

1. materialize the tiled block-level `scf.forall`
2. run a custom C++ pass over the bufferized IR
3. extract runtime dimensions using `memref.dim` or `tensor.dim`
4. compute `grid_x`, `grid_y`, `grid_z` inside IR using `arith` / `affine` / `index` ops
5. create `gpu.launch` directly with those dynamic values
6. splice the original block-level loop body into the launch body

That bypasses the static limitation cleanly.

## Where The Custom Pass Should Sit

For the current NVIDIA path in [`/home/hamza-usta/MatcoreDSL/src/lowering_pipeline.cpp`](/home/hamza-usta/MatcoreDSL/src/lowering_pipeline.cpp), the pass should sit after tiling/promotion has materialized the top-level block structure, but before NVGPU rewrite and before the NVVM pipeline.

Recommended stage order:

1. match and tile `linalg.matmul`
2. promote A/B tiles to workgroup memory
3. materialize loop structure needed for block-level work
4. run `DynamicMacroGridMappingPass`
5. keep block dimensions static via thread mapping / launch config passes
6. apply tensor-core rewrite (`nvgpu.mma.sync`) when legal
7. lower vector/NVGPU/GPU ops to NVVM

The pass replaces only the macro-grid launch creation step. It should not own tensor-core rewrite logic.

## Custom Pass Blueprint

### Pass Contract

Input IR assumptions:

- bufferized memref-based matmul payload
- top-level block-distributed loop nest exists or can be recognized
- block tile sizes are already encoded as static constants

Output IR guarantees:

- `gpu.launch` exists
- `gpu.launch` grid size operands are runtime SSA values derived from `memref.dim`
- block size operands remain static SSA constants
- host C++ never injects `grid_dims = [...]`

### Pass Responsibilities

1. Find the entry `func.func`.
2. Find the top-level matmul or tiled loop carrier.
3. Read runtime sizes from the input/output memrefs:
   - `m = memref.dim %lhs, 0`
   - `k = memref.dim %lhs, 1`
   - `n = memref.dim %rhs, 1`
4. Compute macro-grid counts using index arithmetic:
   - `grid_y = ceil_div(m, block_tile_m)`
   - `grid_x = ceil_div(n, block_tile_n)`
   - `grid_z = 1`
5. Create `gpu.launch grid(%bx, %by, %bz) blocks(%tx, %ty, %tz)` with:
   - dynamic grid SSA values
   - static block size constants
6. Move or clone the tiled block-level payload into the launch body.
7. Replace loop IV usage with block/thread IDs where appropriate.
8. Erase the old top-level block loop if fully replaced.

## C++-Style Pseudocode

```cpp
struct DynamicMacroGridMappingPass
    : public PassWrapper<DynamicMacroGridMappingPass,
                         OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect,
                    mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    auto matmul = findTargetMatmulOrTiledCarrier(func);
    if (!matmul)
      return signalPassFailure();

    OpBuilder b(func.getContext());
    b.setInsertionPoint(matmul);
    Location loc = matmul->getLoc();

    Value lhs = func.getArgument(0);
    Value rhs = func.getArgument(1);

    Value m = b.create<memref::DimOp>(loc, lhs, 0);
    Value n = b.create<memref::DimOp>(loc, rhs, 1);

    Value blockTileM = b.create<arith::ConstantIndexOp>(loc, kBlockTileM);
    Value blockTileN = b.create<arith::ConstantIndexOp>(loc, kBlockTileN);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);

    Value gridY = buildCeilDivIndex(b, loc, m, blockTileM);
    Value gridX = buildCeilDivIndex(b, loc, n, blockTileN);
    Value gridZ = one;

    Value blockX = b.create<arith::ConstantIndexOp>(loc, kBlockThreadsX);
    Value blockY = b.create<arith::ConstantIndexOp>(loc, kBlockThreadsY);
    Value blockZ = b.create<arith::ConstantIndexOp>(loc, kBlockThreadsZ);

    auto launch = b.create<gpu::LaunchOp>(loc,
                                          gridX, gridY, gridZ,
                                          blockX, blockY, blockZ);

    Block &launchBody = launch.getBody().front();
    rehomeOrCloneMacroTiledPayloadIntoLaunch(matmul, launchBody, b);
    rewriteMacroIndicesToGpuIds(launch, launchBody, kBlockTileM, kBlockTileN);

    eraseOldTopLevelMacroLoopIfReplaced(matmul);
  }
};
```

Helper for ceil-div on index values:

```cpp
Value buildCeilDivIndex(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value rhsMinusOne = b.create<arith::SubIOp>(loc, rhs, one);
  Value numerator = b.create<arith::AddIOp>(loc, lhs, rhsMinusOne);
  return b.create<arith::DivUIOp>(loc, numerator, rhs);
}
```

## Concrete Refactor Requirements For MatCore

### 1. Header purge

Remove host-side macro-grid fields from both:

- [`/home/hamza-usta/MatcoreDSL/include/matcore/gpu_mapping.h`](/home/hamza-usta/MatcoreDSL/include/matcore/gpu_mapping.h)
- [`/home/hamza-usta/MatcoreDSL/include/matcore/gpu_tiling.h`](/home/hamza-usta/MatcoreDSL/include/matcore/gpu_tiling.h)

Fields to delete:

- `grid_x`
- `grid_y`
- `grid_z`

### 2. Transform IR purge

In [`/home/hamza-usta/MatcoreDSL/src/gpu_mapping.cpp`](/home/hamza-usta/MatcoreDSL/src/gpu_mapping.cpp):

- stop emitting `grid_dims = [...]`
- stop computing `ceilDiv(m, block_tile_*)` in host C++
- keep only static micro-constants and tensor-core legality checks

### 3. New pass registration

Register the new pass in the modular NVIDIA path and run it in place of the old macro-grid transform stage.

### 4. Preserve static thread geometry

It is valid to keep:

- `block_threads_x`
- `block_threads_y`
- `block_threads_z`
- warp size `32`
- MMA/WGMMA tile constants

Those are micro-topology, not macro-topology.

## Why This Avoids LLVM IR Explosion

The static-grid path encourages the compiler pipeline to see a fully materialized macro launch structure too early. By moving grid derivation into SSA values based on `memref.dim`, the compiler lowers one parametric kernel body instead of specializing the launch geometry from concrete host-side integers during JIT IR construction.

That keeps:

- the kernel shape stable
- block micro-architecture static
- grid extent dynamic

This is the correct balance for a JIT compiler.

## Source Notes

Local MLIR / LLVM 18 sources:

- [`/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td:128`](#/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td:128)
- [`/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td:247`](#/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td:247)
- [`/usr/lib/llvm-18/include/mlir/Dialect/GPU/IR/GPUOps.h.inc`](#/usr/lib/llvm-18/include/mlir/Dialect/GPU/IR/GPUOps.h.inc)
- [`/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/Utils.h`](#/usr/lib/llvm-18/include/mlir/Dialect/GPU/TransformOps/Utils.h)

Project files analyzed:

- [`/home/hamza-usta/MatcoreDSL/src/gpu_mapping.cpp`](/home/hamza-usta/MatcoreDSL/src/gpu_mapping.cpp)
- [`/home/hamza-usta/MatcoreDSL/src/lowering_pipeline.cpp`](/home/hamza-usta/MatcoreDSL/src/lowering_pipeline.cpp)
- [`/home/hamza-usta/MatcoreDSL/src/gpu_nvvm_lowering.cpp`](/home/hamza-usta/MatcoreDSL/src/gpu_nvvm_lowering.cpp)
