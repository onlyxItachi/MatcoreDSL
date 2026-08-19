# Semantic Loss Atlas

## 1. The Principle of Semantic Preservation

> **Fundamental Invariant:** *Preserve high-level semantic information until the very last optimization or lowering phase that can profit from it. A lower representation may consume a fact only after encoding its meaning structurally downstream or completing every decision that requires it.*

---

## 2. Transition Atlas Across Lowering Stages

| Semantic Field | Stage 1: `.mdsl` $\rightarrow$ Matcore IR v1 | Stage 2: v1 $\rightarrow$ `mdsl` MLIR | Stage 3: `mdsl` $\rightarrow$ `linalg` / `scf` | Stage 4: `linalg` $\rightarrow$ `vector` / `memref` | Stage 5: `vector` $\rightarrow$ LLVM IR | Stage 6: LLVM IR $\rightarrow$ Machine Code |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Operation Identity** | `PRESERVED` (explicit GEMM) | `PRESERVED` (`mdsl.gemm`) | `ENCODED_STRUCTURALLY` (`linalg.matmul`) | `ENCODED_STRUCTURALLY` (`vector.contract`) | `CONSUMED` (FMA / load / store loops) | `MACHINE` (ISA instruction selection) |
| **Shape & Rank** | `PRESERVED` ($M, N, K$, rank 2) | `PRESERVED` (tensor/memref dims) | `PRESERVED` (affine maps) | `ENCODED_STRUCTURALLY` (vector sizes) | `WEAKENED` (flat pointer offsets) | `LOST_ACCIDENTALLY` (register bytes) |
| **Dtype & Accum Dtype** | `PRESERVED` (`f32`, `f32`) | `PRESERVED` | `PRESERVED` | `PRESERVED` | `PRESERVED` (`float`, `<8 x float>`) | `ENCODED_STRUCTURALLY` (AVX registers) |
| **Layout & Strides** | `PRESERVED` (row-major, $lda, ldb, ldc$) | `PRESERVED` (strided memref) | `PRESERVED` (indexing maps) | `CONSUMED` (vector load permutations) | `CONSUMED` (GEP stride arithmetic) | `MACHINE` (base + index * scale) |
| **Alignment Precondition** | `PRESERVED` (precondition flag) | `PRESERVED` (`#mdsl.precondition`) | `PRESERVED` | `ENCODED_STRUCTURALLY` (`vector.transfer_read` align) | `PRESERVED` (`align 32` metadata) | `CONSUMED` (`vmovups` vs. `vmovaps`) |
| **Memory Space** | `PRESERVED` (host / global / shared) | `PRESERVED` (attribute) | `PRESERVED` (`#gpu.address_space`) | `PRESERVED` | `ENCODED_STRUCTURALLY` (LLVM addrspace) | `MACHINE` (LDS/Shared vs Global) |
| **Aliasing & Effects** | `PRESERVED` (`out(C)` write-only) | `PRESERVED` (`destination_style`) | `ENCODED_STRUCTURALLY` (DPS ties) | `CONSUMED` (bufferization aliasing) | `WEAKENED` (`noalias` parameter attrs) | `LOST` (registers only) |
| **Numerical Policy** | `PRESERVED` (`explicit-gemm-f32-v1`) | `PRESERVED` (`#mdsl.policy`) | `CONSUMED` (reassociation allowed) | `ENCODED_STRUCTURALLY` (FMA contractions) | `WEAKENED` (`fast-math` flags) | `MACHINE` (vfmadd213ps) |
| **Source Provenance** | `PRESERVED` (File, Line, Col) | `PRESERVED` (MLIR `Location`) | `PRESERVED` (`FusedLoc` / `FileLineColLoc`) | `WEAKENED` (often collapses to root loc) | `WEAKENED` (`!dbg` metadata) | `CONSUMED` (DWARF / PDB tables) |

---

## 3. Critical Failure Points: Premature Semantic Loss

1. **Flattening High-Dimensional Layouts to 1D Pointer Arithmetic Too Early**:
   * *Problem*: If an optimizer flattens `memref<MxNxK>` into a raw `ptr + (i*stride)` before vectorization, the vectorizer cannot infer contiguous 2D tile slices and falls back to scalar loads or non-contiguous gathers.
   * *Mitigation*: Keep structured indexing maps (`affine_map`) through the Linalg and Vector abstraction levels.
2. **Dropping Alignment Contracts Before Vector Transfer Lowering**:
   * *Problem*: Unannotated pointer decays cause LLVM CodeGen to emit unaligned vector moves (`vmovups`), losing throughput on architectures with aligned memory pipelines.
   * *Mitigation*: Matcore must propagate alignment preconditions as verified metadata down to LLVM IR `align 32` attributes.
3. **Loss of Non-Aliasing Proofs at Bufferization**:
   * *Problem*: When converting Destination-Passing Style (DPS) tensor ops to MemRefs, naive bufferizers insert redundant defensive memory copies (`memref.copy`) if they cannot prove non-overlapping buffers.
   * *Mitigation*: Matcore enforces explicit `out(C)` write-only contracts and asserts non-aliasing preconditions at the semantic boundary.
