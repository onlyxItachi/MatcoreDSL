# Core Lowering Invariants & Invariant Analysis

**Confidence Level**: `STRONGLY_SUPPORTED`  
**Evidence Surface**: Multi-version comparison (LLVM 20.1.8, 21.1.8, 22.1.8); 15 diagnostic cases; 120 archaeology artifacts; Optimization remarks.

---

## 1. Lowering Invariant 1: Structured Contraction Lowering Ladder

### OBSERVED:
- In upstream MLIR, structured tensor contractions map across a canonical multi-stage sequence:
  $$\text{Semantic Op (`mdsl.gemm`)} \longrightarrow \text{Linalg (`linalg.matmul`)} \longrightarrow \text{Vector (`vector.contract`)} \longrightarrow \text{Target Matrix / FMA} \longrightarrow \text{LLVM IR}$$
- Lowerings bypassing structured stages (e.g. naive AST-to-scalar-pointer loops) fail to produce optimal vector unrolling or matrix hardware instructions.

### INFERRED:
- Preserving structured multidimensional indexing maps until vector contraction is necessary for the vectorizer to infer contiguous multidimensional slices.

### ARCHITECTURAL IMPLICATION:
- Matcore MLIR bridges directly to `linalg.matmul` / `vector.contract` rather than emitting low-level scalar loop nests.

---

## 2. Lowering Invariant 2: Destination-Passing Style (DPS)

### OBSERVED:
- `linalg.matmul` and upstream tensor operations bind results to explicit output operands (`outs(%C)`).
- When destination writes are disjoint and memory spaces match, One-Shot Bufferization achieves zero-allocation in-place bufferization.

### INFERRED:
- DPS provides the structural foundation for in-place mutation without intermediate defensive heap allocations.

### ARCHITECTURAL IMPLICATION:
- MatcoreDSL's `matcore::mdsl::out(C)` design maps directly to MLIR DPS semantics.

---

## 3. Lowering Invariant 3: Separation of Preconditions from Facts

### OBSERVED:
- Optimization passes in LLVM and MLIR treat alignment and no-alias annotations as contracts requiring static proof or dominating pre-mutation runtime checks. Unproven assertions are conservatively ignored by backend code generators.

### ARCHITECTURAL IMPLICATION:
- MDSLC must enforce alignment and non-aliasing as preconditions: static verification must precede optimization, or a fail-closed runtime guard must dominate the compute region before destination mutation.
