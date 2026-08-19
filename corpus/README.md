# MatcoreDSL Windows Lowering Corpus (LLVM 20 ↔ 21 ↔ 22)

## 1. Overview and Scientific Purpose

This corpus is a scientific, evidence-backed reference artifact capturing compiler lowering behavior, transformation invariants, and semantic degradation across **LLVM/MLIR 20.1.8, 21.1.8 (MDSLC production baseline), and 22.1.8** on Windows x64.

### Core Questions Addressed
1. **Semantic Survival**: Which semantic fields (shapes, strides, aliasing, alignment, domain, numerical policies) survive each lowering boundary?
2. **Decision Points**: At which stage is each optimization, loop tiling, vectorization, and ISA instruction selection made?
3. **Matcore Architectural Boundaries**:
   - **WHAT**: Authenticated source capture + Matcore Semantic MLIR (`mdsl` dialect).
   - **HOW**: Legality verification, scheduling, Linalg/Tensor/Vector structural transformations.
   - **MACHINE**: Target specific lowering (AVX2, AVX-512, AMX, NVPTX/MMA, AMDGPU/MFMA, C ABI).
4. **Version Stability**: Which lowering properties remain invariant across LLVM 20, 21, and 22 versus superficial API churn?

---

## 2. Directory Layout

```text
corpus/
  ├── README.md               # This document
  ├── schema/                 # Formal JSON schemas for entries, environments, semantic loss, and schedules
  ├── inputs/                 # Diagnostic input kernels (CPU C/C++, structured MLIR, GPU CUDA/HIP)
  ├── recipes/                # Automated compilation and lowering pipeline recipes
  ├── environments/           # Toolchain environment descriptors for Windows x64
  ├── manifests/              # Machine-readable corpus manifest indexing all generated cases
  ├── fingerprints/           # SHA-256 hashes and structural fingerprints of representative artifacts
  ├── findings/               # Scientific findings, Semantic Loss Atlas, and Schedule Knowledge Base
  ├── scripts/windows/        # PowerShell automation for bootstrapping, probing, and verification
  └── gold/                   # Curated, compact regression fixtures
```

---

## 3. Data Plane Separation

* **Control Plane (Committed in Git)**: All files under `corpus/` (schemas, recipes, inputs, findings, manifests, fingerprints).
* **Data Plane (External to Git)**: Raw pass-by-pass dumps, LLVM IR, assembly, and object binaries are generated outside the repository under `C:\Users\hamza\MDSLC-Corpus\windows-x64\` (configurable via scripts).
