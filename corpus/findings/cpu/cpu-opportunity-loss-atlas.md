# CPU Lowering Opportunity Loss Atlas

**Investigation Scope**: Comprehensive audit of optimization opportunities destroyed by premature lowering or early library dispatch  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Stage-by-Stage Opportunity Loss Matrix

| Lowering Transition | Information Destroyed | Optimization Opportunity Lost | Irreversible Consequence | Prevention Mechanism in Matcore |
| :--- | :--- | :--- | :--- | :--- |
| **1. Compound AST $\rightarrow$ Classical CBLAS Call** | High-level computational graph of adjacent elementwise post-ops (Bias, ReLU). | **Epilogue Register Fusion**: Cannot fuse post-ops into the register writeback stage. | Forces full matrix $C$ write to RAM followed by separate read/write pass. | **Preserve Compound Operations**: Lower via MLIR `vector.contract` + elementwise writeback. |
| **2. `mdsl` MLIR $\rightarrow$ Pure LLVM Scalar IR** | Multidimensional tensor shapes, layout affine maps, explicit `out(C)` mutability. | **One-Shot Bufferization & Macro-Tiling**: Cannot synthesize memory packing buffers or prove zero-aliasing. | Reverts to generic scalar loop nests vulnerable to TLB/cache thrashing. | **Bridge through Structured MLIR**: Lower `mdsl` $\rightarrow$ `linalg.matmul` $\rightarrow$ `vector.contract`. |
| **3. Target-Agnostic Clang IR $\rightarrow$ Backend Retargeting** | Target ISA knowledge (`-mavx2`, `-mavx512f`) absent at frontend optimization time. | **Vector Width Optimization ($VF=8/16$)**: Middle-end vectorizer freezes to generic 128-bit SSE defaults. | `llc -mattr=+avx2` emits 128-bit VEX-encoded instructions instead of 256-bit YMM ops. | **Mandatory Target Awareness**: Pass target subtarget flags (`-mavx2`, `-mavx512f`) to Clang/MLIR. |
| **4. `mdsl.gemm` $\rightarrow$ LLVM IR without `reassoc`** | Semantic authorization of floating-point reordering. | **SIMD Vector Reduction**: Middle-end vectorizer is blocked by strict IEEE 754 precision. | Emits sequential scalar `vfmadd231ss` loop (0 YMM ops). | **Preserve Reassociation Policy**: Lower `explicit-gemm-f32-v1` to `fastmath<reassoc>` attributes in MLIR. |
