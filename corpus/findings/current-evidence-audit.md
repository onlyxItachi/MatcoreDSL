# Current Evidence Audit (Iteration 0)

**Date**: 2026-08-19  
**Audit Purpose**: Rigorous examination of generated artifacts, execution scripts, schemas, and findings to separate verified facts from aspirational or overstated claims.

---

## 1. Physical Artifact Inventory

### A. Raw Data Plane (`C:\Users\hamza\MDSLC-Corpus\windows-x64\`)
- **Total Files**: 90 files across 15 case subdirectories (`20.1.8`, `21.1.8`, `22.1.8`).
- **File Distribution**:
  - 15 $\times$ `descriptor.json`
  - 15 $\times$ `O0.ll` (Frontend unoptimized LLVM IR)
  - 15 $\times$ `O2.ll` (Mid-tier optimized LLVM IR)
  - 15 $\times$ `O3.ll` (Generic `-O3` optimized LLVM IR)
  - 15 $\times$ `O3_avx2.s` (Target-retargeted assembly via `llc -mattr=+avx2,+fma`)
  - 15 $\times$ `O3_avx512.s` (Target-retargeted assembly via `llc -mattr=+avx512...`)

### B. Control Plane (`corpus/`)
- **Schemas**: 4 schemas (`corpus-entry`, `environment`, `semantic-loss`, `schedule`).
- **Inputs**: 5 CPU C/C++ kernels, 4 MLIR files, 2 GPU files.
- **Manifests**: `windows-corpus-v1.json` (15 cases registered).
- **Fingerprints**: `llvm-20.1.8`, `llvm-21.1.8`, `llvm-22.1.8` JSON ledgers.
- **Scripts**: 4 PowerShell scripts (`probe-toolchains.ps1`, `generate-corpus.ps1`, `verify-corpus.ps1`, `export-descriptors.ps1`).

---

## 2. Discrepancy & Reality Analysis (What Was Actually Generated vs Claimed)

| Category | Initial Claim in Report / Findings | Physical Reality in Artifacts | Scientific Assessment |
| :--- | :--- | :--- | :--- |
| **Pipeline Granularity** | "Pass-by-pass snapshots (AST $\rightarrow$ LLVM IR O0/O2/O3 $\rightarrow$ ASM)" | Only 3 optimization tier snapshots (`O0`, `O2`, `O3`) were emitted. No intermediate pass checkpoints (SROA, LoopSimplify, Vectorize) were saved. | **OVERSTATED**: The baseline corpus contains *endpoint/tier snapshots*, not a fine-grained pass-by-pass pipeline trace. |
| **Target-Aware Vectorization** | "AVX2 uses 256-bit `ymm` registers; AVX-512 uses 512-bit `zmm` registers" | Inspection of all 30 generated `.s` files shows `YMM = 0` and `ZMM = 0`. All assembly files use strictly 128-bit `XMM` registers with VEX/EVEX prefixes. | **CONTRADICTED**: Generic `O3.ll` was compiled first without target flags, freezing 128-bit SSE vectorization before `llc` saw AVX2/AVX-512 flags. |
| **Vectorization Root Cause** | "Aliasing prevents vectorization in naive GEMM" | Optimization remarks (`-Rpass-analysis=loop-vectorize`) reveal the outer $j$ loop was vectorized, but the inner $k$ reduction loop was blocked by floating-point non-associativity (strict IEEE 754 precision). | **INCOMPLETE / MISINTERPRETED**: Floating-point reassociation legality is the primary blocker for inner $k$ reduction vectorization. |
| **Schema Validation** | "Corpus 100% verified against schema" | `verify-corpus.ps1` checked only file existence and SHA-256 hashes against the manifest. No JSON schema validation was executed. | **OVERSTATED**: Hash integrity was conflated with schema validity. |
| **AMDGPU Execution** | Documented in `amd-findings.md` | AMDGPU target is not compiled into the prebuilt Windows LLVM binaries (`llc --version` does not list `amdgpu`). | **CORRECTLY FLAGGED AS GAP**: Matches `GAP-0001` in evidence gaps. |
