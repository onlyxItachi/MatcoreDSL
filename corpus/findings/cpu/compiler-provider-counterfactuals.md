# Compiler vs. Provider Execution Counterfactuals

**Investigation Scope**: Comprehensive comparison across 6 distinct CPU execution strategies on identical mathematical GEMM tasks.  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Multi-Strategy Comparative Matrix

| Execution Strategy | Semantic Requirements | Cache Blocking Ownership | Packing Strategy | Epilogue Fusion Capability | Best Suited Matrix Regimes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Naive C $\rightarrow$ LLVM** | Strict standard C99 | None (Scalar loop) | None | Weak (separate loop) | Toy examples; sub-optimal for all matrix workloads. |
| **B. Enriched C $\rightarrow$ LLVM** | `__restrict__`, `reassoc`, `align` | Relies on LLVM loop tiling | None (In-place streaming) | Fused if in same translation unit | Medium square matrices ($M,N,K \le 64$) with unit strides. |
| **C. Tiled C $\rightarrow$ LLVM** | Explicit multi-loop blocking | Source-explicit 32x32x64 | None | Fused across micro-tiles | Medium matrices fitting in L2/L3 cache. |
| **D. Structured MLIR $\rightarrow$ LLVM** | `linalg.matmul` / `vector.contract` | MLIR `scf.forall` / `linalg.tile` | Can represent packing via `memref.copy` | **Optimal (Fused into writeback)** | High-level DSL pipelines; fused AI operations (GEMM+ReLU). |
| **E. Static BLAS (OpenBLAS/BLIS)** | CBLAS standard C ABI | Hand-tuned 5-loop hierarchy | **Contiguous Panels (SA, SB)** | **Blocked by fixed C ABI** | **Large matrices ($M,N,K \ge 128$)** requiring peak GFLOPS. |
| **F. JIT Provider (LIBXSMM/oneDNN)** | Dynamic shape & dtype descriptors | Bypassed (L1/L2 resident) | **Zero-Pack (Direct JIT streaming)**| **Optimal (JIT In-Register Fusion)**| **Small to medium matrices ($M,N,K \le 128$)** in deep learning. |

---

## 2. Key Synthesis: The Regime Boundaries

```text
               Small Shapes (M,N,K <= 32)      Medium (32 < N <= 128)          Large (N > 128)
           ┌───────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
Top        │ 1. JIT Provider (LIBXSMM)     │ 1. Structured MLIR (Fused)    │ 1. Static BLAS (OpenBLAS/BLIS)│
Performers │ 2. Structured MLIR (Vector)   │ 2. JIT Provider (LIBXSMM)     │ 2. Structured MLIR + Packing  │
           │ 3. Enriched C / Eigen         │ 3. Static BLAS (OpenBLAS)     │ 3. Tiled C                    │
           └───────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
Why:       Packing overhead dominates;      Post-op fusion saves RAM traffic; TLB & cache miss penalty dominates;
           direct JIT streaming wins.       microkernel ILP is decisive.     amortized packing is mandatory.
```
