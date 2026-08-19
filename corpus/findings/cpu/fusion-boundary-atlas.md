# Fusion Boundary Atlas: Across Compilers, JIT, and Numerical Libraries

**Investigation Scope**: Multi-provider analysis across OpenBLAS, BLIS, Eigen, LIBXSMM, and MLIR/LLVM  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Fusion Taxonomy Across Execution Models

| Fusion Category | Concrete Transformation | Classical BLAS (OpenBLAS) | Modern JIT (LIBXSMM / oneDNN) | Compiler-Cooperative (Eigen) | MDSLC / MLIR Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mathematical** | $\alpha \cdot (A \times B)$ Scaling | Fused into writeback (`ymm0`) or packing | In-register FMA scale | Template expression fold | In-register FMA fold |
| **Reduction** | $\beta \cdot C$ Initialization | Separated `BETA_OPERATION` | Staged in register load | Expression rewrite | Destination-Passing Style |
| **Post-Op / Activation** | Bias Addition + ReLU / GELU | **FORBIDDEN by C ABI** (requires separate pass) | **Fused in JIT Microkernel** (in-register before store) | Fused via lazy expression templates | **Fused in `vector.contract` writeback** |
| **Data Layout** | Transpose + Contraction | Handled during packing (`GEMM_ITCOPY`) | Specialized JIT stride loaders | Evaluated into temporary matrix | Transpose permutation in `affine_map` |
| **Graph-Level** | LayerNorm + GEMM + Residual | Broken into discrete function calls | Fused across fused-operator graph | Handled via sequential operator calls | Fused via MLIR bufferization & elementwise fusion |

---

## 2. Why Fixed C ABIs Block Post-Op Fusion

Standard BLAS interfaces (`cblas_sgemm`) expose a fixed C signature:
```c
void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, ... float alpha, const float *A, ... float beta, float *C, ...);
```
Because the signature cannot accept activation functions or auxiliary bias vectors, the application must:
1. Execute `cblas_sgemm` $\rightarrow$ writes full matrix $C$ to RAM.
2. Execute elementwise kernel (e.g. `bias_relu(C)`) $\rightarrow$ reads $C$ from RAM, applies bias+ReLU, writes $C$ back to RAM.

**The Matcore / MLIR Architectural Advantage**:
By fusing post-ops into the register writeback stage before the matrix tile is written to memory, the intermediate round-trip to RAM is **100% eliminated**, saving $2 \times M \times N \times 4$ bytes of memory bus bandwidth per layer!
