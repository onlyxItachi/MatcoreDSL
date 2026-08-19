# Matcore Architectural Boundary Findings

## 1. The Tripartite Compiler Responsibility Model

Formalized in [ADR-0009](../../docs/adr/0009-mdslc-semantic-compiler-foundation.md), MatcoreDSL enforces a strict separation of concerns across three structural layers:

```text
+-----------------------------------------------------------------------------------------------+
| WHAT    : Semantic Intent & Provenance (Matcore Domain)                                       |
|           - Explicit authenticated capture (`matcore::mdsl::gemm`, `matcore::mdsl::out(C)`)  |
|           - Conservative C++ pattern recognition & legality proofs                           |
|           - Matcore IR v1 DTO + Matcore MLIR semantic dialect (`mdsl.gemm`, `mdsl.map`)       |
|           - Typed numerical policy (`explicit-gemm-f32-v1`)                                   |
+-----------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+-----------------------------------------------------------------------------------------------+
| HOW     : Scheduling, Transformation & Structuring (Upstream MLIR & Matcore Planner)          |
|           - Matcore Deterministic Planner (tile size, workspace allocation, variant selection)|
|           - Upstream structured substrates: Linalg (`linalg.matmul`), SCF (`scf.forall`),     |
|             Tensor/MemRef, Vector (`vector.contract`), One-Shot Bufferize                     |
|           - Fusion of elementwise epilogues (`mdsl.map` -> point loops / vector ops)          |
+-----------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+-----------------------------------------------------------------------------------------------+
| MACHINE : Target-Specific Lowering & Code Generation (LLVM & Hardware Runtimes)               |
|           - CPU: AVX2/FMA, AVX-512, AMX microkernels, parallel thread pools, OpenBLAS         |
|           - GPU: NVGPU/NVVM/PTX (`mma.sync`, `ldmatrix`), AMDGPU/ROCDL (`mfma`, LDS)          |
|           - C ABI dispatch (`matcore_runtime_gemm_f32_v0`), object/executable emission        |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Explicit Ownership Boundaries

### What Matcore / MDSLC Owns:
1. **Frontend Integration & Authentication**: In-process Clang LibTooling verification of `<matcore/mdsl.h>` declarations, source file snapshotting, and source location tracking.
2. **Idiom Recognition & Legality Proofs**: Recognizing ordinary C++ compute loops and verifying dependence, aliasing, numerical, and barrier legality before raising. (Recognition $\neq$ Permission).
3. **Semantic Representation**: The internal `mdsl` MLIR dialect encoding exact operation identity, shapes, layouts, destination overwrite semantics, memory spaces, and numerical policies.
4. **Matcore $\rightarrow$ Upstream MLIR Bridge**: Verified, lossless translation from Matcore IR v1 / `mdsl` dialect into structured Linalg/Vector/MemRef substrates.
5. **Planning & Profitability Selection**: Deterministic variant selection (reference, tiled, vectorized, packed, parallel, provider) based on hardware capability records and caller-owned workspace budgets.
6. **Legality Precondition Enforcement**: Treating alignment and no-alias claims as preconditions that mandate static proof or a dominating runtime guard prior to output mutation.

### What Matcore / MDSLC Must NOT Rewrite (Delegated Upstream):
1. **C++ Parsing and Type Checking**: Handled completely by Clang Frontend / Sema.
2. **Generic SSA / Region / Use-Def Infrastructure**: Handled by MLIR Core.
3. **Generic Loop Tiling & Loop Distribution**: Handled by upstream `scf` and `linalg` transform passes.
4. **Generic Vectorization & Contraction Lowering**: Handled by `vector` dialect and `vector.contract` lowerings.
5. **Instruction Selection, Register Allocation & Machine Scheduling**: Handled by LLVM CodeGen / Target Backends.
6. **Object & Assembly Emission**: Handled by LLVM MC layer and native platform linkers (`lld-link`, `ld.lld`).

---

## 3. Destination Mutation and Numerical Invariants

* **Destination Overwrite Semantics**: `mdsl.gemm` produces the post-overwrite semantic value of its explicit write-only destination `out(C)`. It represents an in-place modification of existing storage, not a newly allocated tensor. Bufferization must alias the result to destination storage, preserving the observable mutation.
* **Precondition vs. Fact Rule**: A declaration of 32-byte alignment or `noalias` is a *contract precondition*. Downstream optimizers may consume it as a fact *only* after static proof or after a dominating fail-closed guard executes before destination mutation.
* **The `explicit-gemm-f32-v1` Policy**:
  * Precision: F32 accumulation.
  * Reassociation: Permitted only within the K-reduction dimension.
  * Contraction: FMA formation permitted.
  * Non-Finite Values: NaNs and infinities preserved without payload or ordering guarantees.
  * Signed Zero: Relaxed.
  * Approximate Math: Strictly forbidden (no reciprocal approximations).
