# Confidence Recalibration Changelog (Iteration 5)

This changelog records the calibration of scientific claims resulting from the adversarial audit and evidence expansion in the MatcoreDSL Windows Lowering Corpus investigation.

---

## 1. Claim Calibration Matrix

| Claim ID | Focus Area | Previous Confidence | Recalibrated Confidence | Audit Justification & Empirical Cause |
| :--- | :--- | :--- | :--- | :--- |
| **CLAIM-001** | Multi-Stage Lowering Ladder | `ARCHITECTURAL_INVARIANT` / `very_high` | **STRONGLY_SUPPORTED** / `high` | Multi-stage MLIR lowering is the standard upstream structured pipeline, but direct provider runtime ABIs (`matcore_runtime_gemm_f32_v0`) bypass it. Absolute invariant label was too broad. |
| **CLAIM-002** | DPS Bufferization Allocation Freedom | `ARCHITECTURAL_INVARIANT` / `very_high` | **STRONGLY_SUPPORTED** / `high` | One-Shot Bufferization avoids copies only when operand indexing maps are disjoint and permit in-place aliasing. Non-identity/overlapping maps still trigger defensive allocations. |
| **CLAIM-003** | Separation of Preconditions from Facts | `ARCHITECTURAL_INVARIANT` / `very_high` | **STRONGLY_SUPPORTED** / `high` | Verified across LLVM IR and MLIR: alias and alignment flags cannot be consumed without static proof or dominating pre-mutation guards. |
| **CLAIM-004** | GPU Hardware Collective Boundaries | `ARCHITECTURAL_INVARIANT` / `very_high` | **SUPPORTED** / `medium` | Target code generation to PTX and AMD GCN is verified; however, physical execution on this host is not possible due to GPU hardware unavailability (`GAP-0001`, `GAP-0002`). |
| **CLAIM-005** | `linalg.matmul` Invariance (20 ↔ 22) | `ARCHITECTURAL_INVARIANT` / `very_high` | **STRONGLY_SUPPORTED** / `high` | Syntax and DPS representation are identical across LLVM 20, 21, and 22. |
| **CLAIM-006** | `vector.contract` Unrolling Refinement | `PROVISIONAL_INVARIANT` / `medium` | **SUPPORTED** / `medium` | Observed in upstream MLIR transformation passes; bounded by lack of standalone `mlir-opt.exe` on Windows (`GAP-0004`). |
| **CLAIM-007** | Clang LibTooling AST Matcher Stability | `ARCHITECTURAL_INVARIANT` / `very_high` | **STRONGLY_SUPPORTED** / `high` | Header path and callback signature shifts in LibTooling are API-level only; downstream assembly for GEMM kernels is functionally invariant across versions. |
| **CLAIM-008** | Restrict Annotation Alias Check Elimination | `ARCHITECTURAL_INVARIANT` / `very_high` | **STRONGLY_SUPPORTED** / `high` | Assembly inspection confirms `__restrict__` eliminates pointer overlap check branches emitted in unannotated GEMM. Loop remainder branches still exist. |
| **CLAIM-009** | 32-Byte Alignment Vector Instruction Selection | `ARCHITECTURAL_INVARIANT` / `very_high` | **STRONGLY_SUPPORTED** / `high` | Verified in target-aware compilation: aligned pointers generate `vmovaps`, while unaligned pointers generate `vmovups`. |
| **CLAIM-010** | AVX2 Target YMM Register Utilization | `STRONGLY_SUPPORTED` / `high` (CONTRADICTED in v1) | **STRONGLY_SUPPORTED** / `high` (in v2) | In baseline v1, generic IR retargeting in `llc` produced 0 YMM registers. Corrected in v2 by invoking Clang with `-mavx2 -mfma`, producing verified 26 YMM registers. |
| **CLAIM-011** | AVX-512 Target ZMM Register Utilization | `STRONGLY_SUPPORTED` / `high` (CONTRADICTED in v1) | **STRONGLY_SUPPORTED** / `high` (in v2) | In baseline v1, generic IR retargeting in `llc` produced 0 ZMM registers. Corrected in v2 by invoking Clang with `-mavx512f...`, producing verified 14–26 ZMM registers. |
| **CLAIM-012** | Naive GEMM Inner Loop Reduction Blocker | `STRONGLY_SUPPORTED` / `high` (OVERSTATED in v1) | **STRONGLY_SUPPORTED** / `high` (CORRECTED) | Previous finding misattributed failure to vectorization solely to aliasing. Compiler remarks prove IEEE 754 floating-point reassociation legality is the primary blocker for inner loop reduction. Outer loop vectorized with VF=8. |
| **CLAIM-013** | Bufferization Phase Ordering | `STRONGLY_SUPPORTED` / `high` | **SUPPORTED** / `medium` | Ordering bufferization after tiling and before vectorization avoids memref copies in standard pipelines. |
| **CLAIM-014** | NVIDIA Ada `sm_89` Shared Memory Swizzling | `STRONGLY_SUPPORTED` / `high` | **SUPPORTED** / `medium` | Supported by PTX code generation and CUTLASS/CuTe reference architectures; host execution runtime unavailable. |
| **CLAIM-015** | AMD Matrix Core Wavefront Collective Execution | `STRONGLY_SUPPORTED` / `high` | **SUPPORTED** / `medium` | Supported by target IR and ISA specifications; host execution runtime unavailable. |
| **CLAIM-016** | "Pass-by-Pass" Label | `STRONGLY_SUPPORTED` / `high` (OVERSTATED in v1) | **STRONGLY_SUPPORTED** / `high` (CORRECTED) | Corpus v1 was an endpoint snapshot suite. Corpus v2 adds true optimization pass checkpoints (`raw`, `sroa`, `opt`, `remarks`, `target-aware`). |
| **CLAIM-017** | "Schema Verified" Label | `STRONGLY_SUPPORTED` / `high` (OVERSTATED in v1) | **PENDING VALIDATION** in Iteration 6 | Conflation of SHA-256 hash checks with formal JSON schema validation corrected. Formal schema validator implemented in Iteration 6. |
