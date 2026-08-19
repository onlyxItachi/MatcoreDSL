# Cross-Version Differential Analysis (LLVM/MLIR 20 ↔ 21 ↔ 22)

## 1. Executive Summary & Change Classification

This document tracks changes across **LLVM/MLIR 20.1.8, 21.1.8 (MDSLC Baseline), and 22.1.8**, isolating superficial API churn from true semantic and lowering alterations.

### Classification Categories:
1. **ARCH-INVARIANT**: Stable compiler behavior, identical lowering semantics, and invariant code generation.
2. **API-CHURN**: Renamed C++ helper functions, changed header paths, or modified constructor signatures that require adapter code but carry zero semantic difference.
3. **DIALECT-EVOLUTION**: Structural enhancements to MLIR operations, attributes, or pass registrations.
4. **CODEGEN-DELTA**: Measurable changes in instruction selection, register allocation, or loop vectorization heuristics.

---

## 2. Comprehensive Version Matrix

| Subsystem / Feature | LLVM 20.1.8 | LLVM 21.1.8 (MDSLC Baseline) | LLVM 22.1.8 | Classification | Architectural Impact on MatcoreDSL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`linalg.matmul` Representation** | Standard DPS with `ins(%A, %B)` `outs(%C)` | Standard DPS, enhanced type inference | Standard DPS, stricter rank validation | `ARCH-INVARIANT` | None. Matcore v1 bridge maps directly to `linalg.matmul` across all versions. |
| **`vector.contract` Lowering** | Lowered via `ConvertVectorToLLVM` | Enhanced FMA reduction order | Optimized unrolled contraction patterns | `CODEGEN-DELTA` | LLVM 21/22 produce tighter FMA loop unrolling on AVX2/FMA targets. |
| **One-Shot Bufferization** | `OneShotBufferizePass` default | `bufferization.materialize_in_destination` stabilized | Stricter aliasing analysis on memory spaces | `DIALECT-EVOLUTION` | LLVM 21+ allows clean destination passing without redundant `memref.copy`. |
| **Clang LibTooling AST API** | `clang::ast_matchers` C++20 | Updated `PPCallbacks` & `MacroArgs` | Minor header refactor in AST matchers | `API-CHURN` | Matcore frontend requires only isolated version adapters in `native_frontend.cpp`. |
| **NVIDIA NVGPU / NVVM** | `nvgpu.mma.sync` `m16n8k16` | Added `cp.async.bulk` & TMA abstractions | Expanded Hopper `wgmma` / SM90 primitives | `DIALECT-EVOLUTION` | sm_89 baseline (`mma.sync` + `ldmatrix`) is identical across 20, 21, 22. |
| **AMDGPU / ROCDL** | `amdgpu.mfma` matrix core ops | Enhanced MFMA scheduling patterns | Additional CDNA3 / RDNA3.5 matrix shapes | `DIALECT-EVOLUTION` | Wave-level matrix contracts remain consistent. |
| **CPU Loop Vectorization** | SlpVectorizer + LoopVectorize | Improved gather/scatter cost model | Fine-grained AVX-512 unroll-and-jam | `CODEGEN-DELTA` | Vectorization heuristics improve slightly on irregular loop bounds in 22. |

---

## 3. Stable Lowering Invariants (What Never Breaks)

1. **Structured Contraction Abstraction**: The sequence `linalg.matmul` $\rightarrow$ `vector.contract` $\rightarrow$ target intrinsic $\rightarrow$ LLVM IR remains the universal lowering backbone across all three releases.
2. **Destination Passing Style (DPS)**: Tying results to explicit write-only destinations `outs(...)` is universally preserved, confirming Matcore's `matcore::mdsl::out(C)` design as future-proof.
3. **Precondition Lowering**: In all three versions, alignment information must be explicitly conveyed via LLVM attributes (`align 32`) or assume builtins; otherwise, CodeGen conservatively defaults to unaligned vector memory access.
