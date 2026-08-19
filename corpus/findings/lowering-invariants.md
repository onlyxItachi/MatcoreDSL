# Core Lowering Invariants (LLVM 20 ↔ 21 ↔ 22)

## 1. What Remains Substantially Invariant Across LLVM Generations

Through cross-version compiler archaeology across LLVM 20, 21, and 22, the following core lowering invariants have been established as foundational:

---

### Invariant 1: Multi-Stage Contraction Lowering
Tensor contractions follow an immutable multi-stage lowering ladder:
$$\text{High-level Op (GEMM)} \longrightarrow \text{Structured Linalg} \longrightarrow \text{Vector Contraction} \longrightarrow \text{Target Matrix Intrinsic / FMA} \longrightarrow \text{LLVM IR}$$
* Every attempt to bypass intermediate stages (e.g. going directly from AST to flat LLVM loops) loses optimization opportunities (vector unrolling, register tiling, hardware tensor cores).

---

### Invariant 2: Destination-Passing Style (DPS) as the Bufferization Foundation
* Tying operation outputs to explicit destination values (`outs(...)`) in MLIR is the single most durable pattern across LLVM 20–22.
* This validates Matcore's `matcore::mdsl::out(C)` explicit destination mutation design as architecturally permanent.

---

### Invariant 3: Separation of Preconditions from Facts
* In LLVM IR and MLIR, optimization passes cannot legally assume non-aliasing or pointer alignment without:
  1. Static proof (e.g., `noalias` parameters, `__builtin_assume_aligned`).
  2. Dominating runtime checks/guards.
* Matcore must strictly maintain this boundary: assertions in the DSL are contracts that require verification or pre-mutation guards before CodeGen consumes them.

---

### Invariant 4: Hardware Matrix Instruction Boundaries
* For both NVIDIA (`mma.sync`) and AMD (`amdgpu.mfma`), matrix operations are strictly cooperative at the sub-warp/warp/wavefront level.
* Compiler representations must preserve warp-collective semantics and cannot scalarize individual lane stores without destroying performance.
