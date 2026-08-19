# Current Generalization State & Baseline Audit

**Investigation Baseline**: LLVM 21.1.8 (MDSLC Baseline), OpenBLAS v0.3.29, BLIS master (`061c2eb`), Eigen 3.4.0 (`3147391`), LIBXSMM master (`ead2376`)  
**Scope**: Transition from initial Haswell/AVX2 SGEMM anchor to the modern multi-architecture, multi-provider CPU compute decision architecture.  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Generalization Audit Matrix of Prior Findings

| Finding / Prior Claim | Current Scope | Underlying General Principle | Specificity Artifact to Remove / Broaden |
| :--- | :--- | :--- | :--- |
| **1. Register Pressure Saturation Ceiling ($16 \times 6$)** | Haswell / AVX2 (16 YMM registers) | **Universal Register File Law**: $\text{Accumulators} + \text{Temporaries} \le N_{\text{arch\_regs}}$. Exceeding this budget causes stack spills. | The specific geometry $16 \times 6$ is fixed-SIMD 16-register specific. Must generalize to 32-register SIMD (AVX-512, NEON), scalable vectors (SVE, RVV), and tile registers (AMX). |
| **2. Reduction Vectorization Blocker** | C99 Row-Major IJK GEMM | **Floating-Point Semantics Law**: Strict IEEE 754 precision forbids altering addition order without `reassoc` or loop interchange. | Blocker applies to all reduction loops across all ISAs (SSE, AVX, NEON, SVE, RVV), not merely GEMM on x86. |
| **3. Goto 5-Loop Cache Blocking Hierarchy** | OpenBLAS Level-3 GEMM Driver | **Memory Hierarchy Alignment Law**: Large multidimensional contractions require cache-sliced macro-tiles ($N_C \in L3, K_C \in L2, M_C \in L1$) to prevent TLB thrashing. | Goto structure is one design point (OpenBLAS). BLIS uses a formal 5-loop control tree; LIBXSMM bypasses it for small shapes; Eigen uses template blocking. |
| **4. Contiguous Data Packing (SA, SB)** | OpenBLAS `ICOPY` / `OCOPY` | **Stride Elimination & Amortization Law**: Transforming arbitrary memory strides into unit-stride micro-panels amortizes copy overhead when reuse factor $R \gg 1$. | For small matrices ($M,N,K \le 32$) or high cache-hit rates, packing overhead exceeds compute benefit (OpenBLAS and LIBXSMM use direct no-pack paths). |
| **5. Fused Epilogue Advantage** | GEMM + ReLU / Bias / Sin | **Register Store Stage Fusion**: Fusing elementwise operations into the register writeback stage eliminates intermediate memory round-trips. | Classical CBLAS C ABI separates epilogues; modern JIT providers (oneDNN, LIBXSMM) and DSLs (Matcore) synthesize fused post-op epilogues. |
| **6. Frontend Target Awareness** | Clang `-mavx2` vs `llc -mattr=+avx2` | **Target-Aware IR Optimization Invariant**: Vectorization factor and unroll decisions freeze during middle-end optimization; backends cannot widen frozen vector IR. | Universal across all LLVM targets (x86, AArch64, RISC-V). |

---

## 2. Boundaries Established for the Generalization Expansion

1. **Host Is Not Target**: All non-host architectures (AArch64 NEON/SVE, RISC-V RVV, x86 AMX) are probed via target-aware Clang/LLC cross-compilation and marked `statically_compiled_only`.
2. **Multi-Provider Comparison**: We explicitly compare OpenBLAS (hand-tuned asm), BLIS (formal microkernel framework), Eigen (C++ expression/packet templates), and LIBXSMM (JIT microkernel generator).
3. **Multi-Operation Suite**: Expanding beyond SGEMM to DGEMM, GEMV, DOT reduction, AXPY streaming, 2D Transpose, fused epilogues, and small/skinny matrix dimensions.
