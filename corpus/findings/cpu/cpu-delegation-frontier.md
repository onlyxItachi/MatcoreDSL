# CPU Delegation Frontier & Boundary Tradeoff Atlas

**Investigation Scope**: Comprehensive architectural tradeoff analysis across 4 candidate compiler/provider delegation boundaries  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Candidate Delegation Frontier Matrix

| Candidate Delegation Frontier | Required Semantic Contract | Opportunities Gained | Opportunities Lost if Dispatched / Lowered Too Early | Implementation Complexity | Portability & Stability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Boundary 1: Matcore $\rightarrow$ Structured MLIR (`linalg` / `vector`)** | Rich affine maps, destination passing (`out(C)`), FP reassociation flags. | **Optimal Epilogue Fusion**; automatic buffer reuse; cross-target vector unrolling. | If lowered prematurely to LLVM IR, multidimensional affine loop structure and DPS alias guarantees are lost. | Moderate (uses upstream MLIR dialects). | **Highest**: Portable across x86, ARM, RISC-V, and GPU backends. |
| **Boundary 2: Matcore $\rightarrow$ Pure LLVM IR** | `noalias`, `assume_aligned`, `reassoc` flags on scalar loop nests. | Direct compilation via standard Clang/LLC. | **Loses Macro-Kernel Cache Sizing & Memory Packing**: LLVM cannot synthesize global packing buffers. | Low (emits C / LLVM IR directly). | High portability; low peak throughput on large strided matrices. |
| **Boundary 3: Matcore $\rightarrow$ Static BLAS (OpenBLAS / BLIS)** | CBLAS standard C ABI (`cblas_sgemm`). | **Maximum Large Matrix Throughput**: Reuses mature hand-tuned assembly microkernels and 5-loop engine. | **DESTROYS POST-OP FUSION**: Fixed C ABI mandates writing matrix $C$ to RAM before applying activations. | **Lowest**: Pure C ABI call (`matcore_runtime_gemm_f32_v0`). | High (depends on external shared library linking). |
| **Boundary 4: Matcore $\rightarrow$ JIT Provider (LIBXSMM)** | Runtime dimension descriptors ($M, N, K$, strides). | **Zero-Pack Microkernel Streaming**: Instant dynamic code gen for small/medium shapes ($N \le 128$). | Introduces runtime JIT compilation latency and executable memory page overhead. | High (requires linking JIT runtime). | Target-specific (x86/ARM JIT backends). |

---

## 2. The Hybrid Multi-Frontier Strategy for MatcoreDSL

The evidence demonstrates that **no single delegation boundary is universally optimal across all matrix shapes and operational contexts**:

```text
                                 Incoming Matcore eDSL Expression
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [Compound Operation with Epilogue]                             [Pure Standard GEMM]
        (e.g., GEMM + Bias + ReLU)                                     (No Post-Ops)
                 │                                                             │
                 ▼                                                             ▼
    Lower via Structured MLIR                                    Evaluate Matrix Shape / Target:
    (`mdsl` -> `linalg` -> `vector.contract`)                                  │
                 │                                              ┌──────────────┴──────────────┐
                 ▼                                              ▼                             ▼
    In-Register Fused Epilogue Writeback               [Small Shapes: N <= 64]      [Large Shapes: N > 64]
    (Saves 50% Memory Bus Bandwidth)                   Emit Structured MLIR Tile   Dispatch to Authenticated
                                                       (Zero Packing Overhead)      CBLAS / OpenBLAS Provider
```
