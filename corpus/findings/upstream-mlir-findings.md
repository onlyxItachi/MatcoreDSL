# Upstream MLIR Transformation & Dialect Findings

## 1. Upstream MLIR Lowering Pathway

The primary structured lowering pipeline for matrix multiplication in upstream MLIR across LLVM 20, 21, and 22 follows this canonical progression:

```text
linalg.matmul (on tensors with Destination Passing Style)
    │
    ▼ (Tile to thread blocks / workers)
scf.forall / scf.for loop nests
    │
    ▼ (One-Shot Bufferization)
memref.subview allocations + bufferization.materialize_in_destination
    │
    ▼ (Vectorization)
vector.transfer_read -> vector.contract -> vector.transfer_write
    │
    ▼ (Hardware Matrix Mapping)
Target Dialect:
  ├── CPU: vector.unroll -> LLVM Dialect (vector.fma / llvm.intr.fma)
  ├── NVIDIA: nvgpu.mma.sync / nvvm.mma.sync
  └── AMD: amdgpu.mfma / rocdl.mfma
    │
    ▼ (Terminal Lowering)
LLVM Dialect -> LLVM IR
```

---

## 2. Key Archaeological Observations

### A. One-Shot Bufferization Behavior
* Bufferization is most optimal when executed **after** high-level tiling but **before** fine-grained vectorization.
* Using explicit Destination-Passing Style (DPS) with `bufferization.materialize_in_destination` guarantees in-place output modification without inserting defensive allocations (`memref.alloc`) or buffer copies (`memref.copy`).

### B. Vector Contraction & Unrolling
* `vector.contract` is the universal structured representation for tensor contractions.
* When lowering to CPU SIMD, `vector.unroll` splits large $M \times N \times K$ vector contractions into exact $6 \times 16$ or $12 \times 32$ hardware register tiles, directly mapping to vector FMA hardware instructions.
