# Fusion Boundary Atlas: Across Compilers, JIT, and Numerical Libraries

**Investigation Scope**: Multi-provider analysis across OpenBLAS, BLIS, Eigen, LIBXSMM, and MLIR/LLVM  
**Audited Benchmark**: Real host micro-benchmark measuring GEMM + separate vs fused ReLU  
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

## 2. Program-Level Store/Reload Pass Elimination vs Physical DRAM Traffic

Standard BLAS interfaces (`cblas_sgemm`) expose a fixed C signature:
```c
void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, ... float alpha, const float *A, ... float beta, float *C, ...);
```
Because the signature cannot accept activation functions or auxiliary bias vectors, the application must execute a separate activation pass over matrix $C$.

**Clarification of Memory Traffic Savings**:
- **Logical Program Dataflow**: Fusing post-ops into the register writeback stage eliminates an entire intermediate store and subsequent load pass at the program level.
- **Physical DRAM Traffic vs Cache Residency**:
  - For **cache-resident working sets ($N \le 128$)**: Matrix $C$ remains in L1/L2 CPU cache. The separate pass hits L1/L2 cache, resulting in zero physical DRAM traffic in both cases.
  - For **cache-exceeding matrices ($N \ge 512$)**: Fusing the epilogue eliminates an entire round-trip of matrix $C$ through the physical DRAM memory bus.
  - For **elementwise chains (Bias + LayerNorm + ReLU)**: Fusing delivers substantial speedups on memory-bandwidth-bound inference layers.
