# Dynamic Matmul Padding Strategy

## Phase 4.4 Goal

MatCore must stop falling back away from NVIDIA Tensor Core MMA just because the
logical matrix sizes are not already aligned to the hardware micro-tile.

For Tensor Core matmul, the effective contract is:

- `M` must be padded to a multiple of `16`
- `N` must be padded to a multiple of `8`
- `K` must be padded to a multiple of `16`

The compiler should therefore rewrite:

- `lhs: [M x K]`
- `rhs: [K x N]`
- `out: [M x N]`

into padded buffers:

- `lhs_pad: [Mp x Kp]`
- `rhs_pad: [Kp x Np]`
- `out_pad: [Mp x Np]`

where:

- `Mp = ceilDiv(M, 16) * 16`
- `Np = ceilDiv(N, 8) * 8`
- `Kp = ceilDiv(K, 16) * 16`

Padding is high-side only and uses zeros.

## Current MatCore Constraint

MatCore currently builds `linalg.matmul` directly on memrefs in
[`src/mlir_engine.cpp`](/home/hamza-usta/MatcoreDSL/src/mlir_engine.cpp), not on
tensor IR. That matters because it changes the practical implementation choice.

In pure MLIR terms, `tensor.pad` is the canonical tensor-level padding op, and
it supports mixed static/dynamic high padding in local LLVM 18
[`TensorOps.td`](/usr/lib/llvm-18/include/mlir/Dialect/Tensor/IR/TensorOps.td).
However, `tensor.pad` is not destination-style, as documented in local
[`DestinationStyleOpInterface.td`](/usr/lib/llvm-18/include/mlir/Interfaces/DestinationStyleOpInterface.td),
and using it cleanly here would mean carrying tensor IR farther through the
pipeline and bufferizing later.

For the current MatCore codebase, the practical strategy is:

- keep the pass memref-based
- implement the same padding semantics with `memref.alloc`, `memref.subview`,
  `linalg.fill`, and `linalg.copy`
- run this pass before NVIDIA mapping and MMA transform application

This fits the current pipeline with the fewest invasive changes.

## Recommended Pass Contract

Create a new pass, e.g. `DynamicMatmulPaddingPass`, as an
`OperationPass<mlir::func::FuncOp>`.

The pass should:

1. Find each `linalg.matmul` in the function.
2. Read the logical `M`, `N`, `K` from the matmul operand/output memrefs.
3. Compute `Mp`, `Np`, `Kp`.
4. If already aligned, do nothing.
5. Otherwise:
   - allocate padded memrefs
   - zero-fill them
   - copy valid input regions into padded buffers
   - run a replacement `linalg.matmul` on the padded buffers
   - copy only the valid `[0:M, 0:N]` result region back to the original output
6. Erase the original unpadded `linalg.matmul`.

This guarantees:

- no out-of-bounds writes to the original output
- no residual odd-sized MMA input reaching the NVIDIA transform pipeline
- exact logical output shape preservation

## Why This Must Run Early

Padding must happen before NVIDIA mapping and before the transform-dialect MMA
rewrite.

In the current pipeline, the important order in
[`src/lowering_pipeline.cpp`](/home/hamza-usta/MatcoreDSL/src/lowering_pipeline.cpp)
is:

1. `selectNvidiaMappingForModule(...)`
2. `ApplyNvidiaMmaTransformToModule(...)`
3. dynamic macro-topology pass
4. MMA preparation / MMA rewrite / launch config / loop materialization

The new padding pass should run before step 1, so that:

- the mapper sees MMA-compatible sizes
- the transform payload does not need to reason about remainder tiles
- `rewrite_to_mma_sync` can be forced based on dtype support instead of shape

## Interaction With Existing GPU Passes

`PromoteGpuWorkgroupAllocationsPass` in
[`src/gpu_tiling.cpp`](/home/hamza-usta/MatcoreDSL/src/gpu_tiling.cpp) only
works on workgroup allocations inside `gpu.launch`.

The new padded allocations are top-level function-local buffers that exist
before GPU outlining/mapping. That is okay:

- the padding pass should create normal host/default-memory memrefs
- tiling and promotion should then operate on the padded matmul in the usual way
- only the internal workgroup staging buffers should be promoted later

So the padding pass should not try to create workgroup allocations itself.

## Tensor-Level Research Notes

Tensor-level padding is still the canonical MLIR idea and is useful context:

- `tensor.pad` computes result shape as `low + dim + high`
- `tensor.extract_slice` and `tensor.insert_slice` represent the valid region
- after bufferization, those lower into allocation/subview/copy style behavior

If MatCore later moves to a tensor-first frontend IR, a tensor-level padding
pass would become the cleaner long-term design.

For Phase 4.4 in the current codebase, memref-level padding is the better fit.

## Memref-Level Rewrite Pattern

Assume:

- `lhs: memref<MxKxT>`
- `rhs: memref<KxNxT>`
- `out: memref<MxNxT>`

Rewrite to:

1. Compute padded sizes `Mp`, `Np`, `Kp`.
2. Allocate:
   - `lhsPad : memref<Mp x Kp x T>`
   - `rhsPad : memref<Kp x Np x T>`
   - `outPad : memref<Mp x Np x T>`
3. Fill `lhsPad`, `rhsPad`, and `outPad` with zero.
4. Create subviews:
   - `lhsPadValid = memref.subview lhsPad[0, 0][M, K][1, 1]`
   - `rhsPadValid = memref.subview rhsPad[0, 0][K, N][1, 1]`
   - `outPadValid = memref.subview outPad[0, 0][M, N][1, 1]`
5. Copy:
   - `linalg.copy lhs -> lhsPadValid`
   - `linalg.copy rhs -> rhsPadValid`
6. Replace the original matmul with:
   - `linalg.matmul ins(lhsPad, rhsPad) outs(outPad)`
7. Copy the valid result back:
   - `linalg.copy outPadValid -> out`
8. Erase the original matmul.

The original `linalg.fill` on `out` from `mlir_engine.cpp` can be left in place
as redundant-but-safe, or removed if trivially matched just before the original
matmul.

## Safe Write-Back Rule

This is the non-negotiable safety rule:

- never write the full padded result directly into the original output memref

Instead:

- compute into `outPad`
- extract a valid top-left subview of size `[M, N]`
- copy only that valid subview into `out`

That avoids out-of-bounds writes by construction.

## Shape Computation Strategy

The pass should support mixed static/dynamic memref shapes.

Use:

- static integers when the memref type already knows the dimension
- `memref.dim` when the dimension is dynamic

Helpful local APIs:

- [`mlir::memref::getMixedSize`](/usr/lib/llvm-18/include/mlir/Dialect/MemRef/IR/MemRef.h)
- [`mlir::memref::getMixedSizes`](/usr/lib/llvm-18/include/mlir/Dialect/MemRef/IR/MemRef.h)
- [`memref.subview`](/usr/lib/llvm-18/include/mlir/Dialect/MemRef/IR/MemRefOps.td)

Padding computation can be expressed with `arith` index ops:

```c++
Value ceilDivMul(OpBuilder &b, Location loc, Value dim, int64_t tile) {
  Value cTile = b.create<arith::ConstantIndexOp>(loc, tile);
  Value cOne = b.create<arith::ConstantIndexOp>(loc, 1);
  Value dimPlusTileMinusOne =
      b.create<arith::AddIOp>(loc, dim,
          b.create<arith::SubIOp>(loc, cTile, cOne));
  Value q = b.create<arith::DivUIOp>(loc, dimPlusTileMinusOne, cTile);
  return b.create<arith::MulIOp>(loc, q, cTile);
}
```

For static dims, the pass can fold to constants.

## Pseudo-C++ Skeleton

```c++
struct DynamicMatmulPaddingPass
    : public PassWrapper<DynamicMatmulPaddingPass,
                         OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<linalg::MatmulOp, 4> matmuls;
    func.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

    OpBuilder b(&getContext());
    for (linalg::MatmulOp op : matmuls) {
      Value lhs = op.getInputs()[0];
      Value rhs = op.getInputs()[1];
      Value out = op.getOutputs()[0];

      auto lhsType = dyn_cast<MemRefType>(lhs.getType());
      auto rhsType = dyn_cast<MemRefType>(rhs.getType());
      auto outType = dyn_cast<MemRefType>(out.getType());
      if (!lhsType || !rhsType || !outType || !lhsType.hasRank() ||
          lhsType.getRank() != 2 || !rhsType.hasRank() || rhsType.getRank() != 2 ||
          !outType.hasRank() || outType.getRank() != 2) {
        continue;
      }

      b.setInsertionPoint(op);
      Location loc = op.getLoc();

      Value m = getDimValue(b, loc, out, outType, 0);
      Value n = getDimValue(b, loc, out, outType, 1);
      Value k = getDimValue(b, loc, lhs, lhsType, 1);

      Value mp = ceilDivMul(b, loc, m, 16);
      Value np = ceilDivMul(b, loc, n, 8);
      Value kp = ceilDivMul(b, loc, k, 16);

      if (isAlreadyAligned(lhsType, rhsType, outType))
        continue;

      MemRefType lhsPadType = makePaddedType(lhsType, /*rows=*/mp, /*cols=*/kp);
      MemRefType rhsPadType = makePaddedType(rhsType, /*rows=*/kp, /*cols=*/np);
      MemRefType outPadType = makePaddedType(outType, /*rows=*/mp, /*cols=*/np);

      Value lhsPad = createAllocForType(b, loc, lhsPadType, {mp, kp});
      Value rhsPad = createAllocForType(b, loc, rhsPadType, {kp, np});
      Value outPad = createAllocForType(b, loc, outPadType, {mp, np});

      Value zero = buildZeroConstant(b, loc, lhsType.getElementType());
      b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{lhsPad});
      b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{rhsPad});
      b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{outPad});

      Value lhsPadValid = createSubview(b, loc, lhsPad, {0, 0}, {m, k});
      Value rhsPadValid = createSubview(b, loc, rhsPad, {0, 0}, {k, n});
      Value outPadValid = createSubview(b, loc, outPad, {0, 0}, {m, n});

      b.create<linalg::CopyOp>(loc, lhs, lhsPadValid);
      b.create<linalg::CopyOp>(loc, rhs, rhsPadValid);
      b.create<linalg::MatmulOp>(loc, ValueRange{lhsPad, rhsPad},
                                 ValueRange{outPad});
      b.create<linalg::CopyOp>(loc, outPadValid, out);

      op.erase();
    }
  }
};
```

## Practical MatCore Integration

Stage 2 should:

1. Add the new pass declaration/creation in the NVIDIA GPU support layer.
2. Run the pass in the NVIDIA branch of
   [`src/lowering_pipeline.cpp`](/home/hamza-usta/MatcoreDSL/src/lowering_pipeline.cpp)
   before `selectNvidiaMappingForModule(...)`.
3. Remove the dynamic-shape rejection from:
   - [`src/gpu_mapping.cpp`](/home/hamza-usta/MatcoreDSL/src/gpu_mapping.cpp)
   - [`src/gpu_tiling.cpp`](/home/hamza-usta/MatcoreDSL/src/gpu_tiling.cpp)
4. Force `rewrite_to_mma_sync` from dtype legality, not shape divisibility.

After this, every FP16 Tensor Core-compatible matmul should be rewritten into an
MMA-friendly padded form before the NVIDIA transform sequence sees it.

## Main Traps

- Padding only inputs but not the destination breaks destination-style matmul.
- Copying full padded output into the original output risks OOB writes.
- Running the padding pass after MMA preparation is too late.
- Leaving remainder shapes to the transform pipeline defeats the purpose of the
  phase.
- If the pass creates workgroup allocations directly, it will interfere with the
  existing promotion path.

## Bottom Line

For Phase 4.4 in the current MatCore codebase, the correct implementation
direction is:

- a memref-based `DynamicMatmulPaddingPass`
- inserted before NVIDIA mapping/transform
- padding all non-aligned matmuls to `16x8x16`
- computing into padded buffers
- copying only the valid `[M, N]` region back
- removing the current dynamic-shape MMA fallback gate
