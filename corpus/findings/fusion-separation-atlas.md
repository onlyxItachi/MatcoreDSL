# Fusion / Separation Atlas: Compiler vs. BLAS

**Investigation Scope**: OpenBLAS v0.3.29, Upstream MLIR, and Clang/LLVM 20–22  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Taxonomic Classification Matrix

| Operation Category | Specific Transformation | OpenBLAS Strategy | Generic Compiler (LLVM) Strategy | Structural Cause & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **A. Semantic / Operation** | $\alpha \cdot (A \times B)$ Scaling | **In-Register Store Fusion** (in `ymm0` at writeback) or **Packing Pre-Scale** | Emits separate scalar multiply or folds into FMA | Keeps inner $K$ accumulation loop pure ($C_{tile} += A \times B$) without extra runtime scalar multiplications. |
| **A. Semantic / Operation** | $\beta \cdot C$ Initialization | **Separated First-Slice Pass** (`BETA_OPERATION`) | Fuses into main expression | Slicing $K$ requires $\beta$ applied only once at $k=0$; fusing $\beta$ into the inner loop requires reloading/scaling $C$ on every step. |
| **B. Data-Movement** | Memory Packing ($A \rightarrow \text{SA}, B \rightarrow \text{SB}$) | **Strictly Separated** (`ICOPY` / `OCOPY` routines) | Fused (in-place streaming) | Packed panels are reused across multiple loop iterations (packed $B$ reused for all $M_C$ tiles). In-loop packing causes redundant copying. |
| **B. Data-Movement** | Register Loading & Broadcast | **Fused in Microkernel** (`vmovups` + `vbroadcastss`) | Fused in Vectorizer | Maximizes instruction-level parallelism (ILP) and overlaps load latency with FMA pipeline execution. |
| **B. Data-Movement** | Cache Hierarchy Blocking | **Separated Explicit Outer Loops** ($J_C \rightarrow P_C \rightarrow I_C$) | Relies on Polyhedral / Loop Tiling passes | BLAS explicitly aligns tile sizes ($M_C, K_C, N_C$) to physical cache capacities (L1, L2, L3). Generic compilers struggle to infer optimal blocking without hardware models. |
| **C. Graph / Post-Op** | Bias Addition & Activation (ReLU, GELU) | **Separated / Unavailable** (Standard CBLAS C ABI boundary) | **Fused in MLIR / Epilogue** | CBLAS C ABI is fixed. Compilers and DSLs (like Matcore) with fused epilogue support fuse activations into register writeback, avoiding round-trips to RAM. |
| **D. Operation Reuse** | TRMM / TRSM Trailing Updates | **Reused GEMM Microkernel** (`STRMMKERNEL = sgemm_kernel_8x4...`) | Distinct loop syntheses | BLAS amortizes microkernel optimization across level-3 operations; triangular solves reduce to diagonal solve + GEMM panel updates. |
| **D. Shape Dispatch** | Rank-1 / Vector Cases ($M=1$ or $N=1$) | **Dispatched to GEMV** (`sgemv_n` / `sgemv_t`) | Uniform generic loop lowering | When $M=1$ or $N=1$, Goto packing overhead exceeds compute cost; GEMV streams vectors directly without workspace allocation. |
