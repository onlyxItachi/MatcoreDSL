# The MDSLC CPU Execution Decision Function F

**Core Theoretical Deliverable**: The complete, evidence-grounded mathematical and algorithmic decision function $F$ governing high-performance CPU lowering in MatcoreDSL.

---

## 1. Formal Specification of Decision Function F

$$F(\text{op}, M, N, K, \text{dtype}, \text{layout}, \text{aliasing}, \text{reassoc}, \text{fusion}, \text{threads}, \text{target}) \longrightarrow \text{ExecutionStrategy}$$

```
                ┌────────────────────────────────────────┐
                │        MatcoreDSL Operation            │
                └───────────────────┬────────────────────┘
                                    │
                    Is Shape Small & Compile-Time Known?
                         (e.g., M,N,K <= 16 Static)
                                   / \
                                 YES  NO
                                 /     \
               ┌────────────────────┐   │
               │ Direct MLIR/LLVM   │   │
               │ In-Register Unroll │   │
               │ (5-7x Speedup)     │   │
               └────────────────────┘   │
                                        │
                         Is Epilogue Fused (e.g. ReLU/Bias)?
                                       / \
                                     YES  NO
                                     /     \
               ┌────────────────────────┐   │
               │ Lower to MLIR          │   │
               │ vector.contract with   │   │
               │ In-Register Store ReLU │   │
               └────────────────────────┘   │
                                            │
                          Is Operation Large GEMM (N >= 64)?
                                           / \
                                         YES  NO
                                         /     \
               ┌────────────────────────────┐   │
               │ Dispatch to Authenticated  │   │
               │ OpenBLAS / BLIS (CBLAS)    │   │
               │ with 5-Loop Cache Packing  │   │
               └────────────────────────────┘   │
                                                │
                              Is Operation Level 2 (GEMV/GEVM)?
                                               / \
                                             YES  NO
                                             /     \
             ┌──────────────────────────────────┐  ┌───────────────────┐
             │ Parallel Multi-Core Vector Path: │  │ LLVM Direct Vector│
             │ - GEVM: Broadcast Streaming      │  │ Reduction (DOT/GER│
             │ - Cap Threads at Bus Knee (<=8T) │  │ with Reassoc)     │
             └──────────────────────────────────┘  └───────────────────┘
```

---

## 2. Parameter-by-Parameter Decision Rules

### A. Shape Dimension Regimes ($M, N, K$)
1. **Tiny Static Shapes ($M, N, K \le 16$, Compile-Time Known)**:
   - **Decision**: Lower via template / compile-time static unroll directly in MLIR / LLVM.
   - **Empirical Grounding**: Static shape unrolling yields **5.7–7.16x speedup** over dynamic loop code by completely removing loop counters and branch overhead.
2. **Medium Cache-Resident Shapes ($16 < M, N, K \le 128$)**:
   - **Decision**: 2D microkernel tile ($M_R \times N_R = 16 \times 6$ on AVX2) with in-place streaming or single-pass panel packing.
3. **Large Memory-Exceeding Shapes ($M, N, K \ge 256$)**:
   - **Decision**: Dispatch to authenticated CBLAS / OpenBLAS / BLIS with a 5-loop cache hierarchy ($M_C \times K_C \times N_C$).

---

### B. Operation Class Adaptation
1. **GEMM (Level 3 Compute-Bound)**:
   - Scales linearly with core count (up to 12 physical cores and 24 SMT threads, achieving 189 GFLOP/s).
   - Requires $M_R \times N_R = 16 \times 6$ (75% vector register saturation) to avoid load stalls.
2. **GEMV vs GEVM (Level 2 Memory-Bound)**:
   - **GEVM ($y^T = x^T \cdot A$) is strictly faster than GEMV ($y = A \cdot x$) by 1.4–1.6x** because vector broadcasting avoids horizontal SIMD reduction overhead.
   - **Threading Cap**: Multi-threaded Level 2 operations must cap worker threads at the memory bus saturation knee ($\le 8$ threads on Zen 5) and **strictly forbid SMT** (which degrades throughput by -29%).
3. **GEVV-INNER (Dot Product)**:
   - Requires `reassoc` permission to vectorize in C99.
   - Vectorizes via 1D SIMD reduction lanes.

---

### C. The Minimum Ownership Boundary for MDSLC
MatcoreDSL must own exactly **Three Responsibilities**:
1. **Semantic Authorization Layer**: Attach Destination-Passing Style `out(C)`, IEEE `reassoc` reordering permissions, and `noalias` preconditions.
2. **Epilogue Fusion Engine**: Fuse elementwise post-ops (Bias, ReLU, GELU) into the register store stage of contractions, eliminating intermediate memory roundtrips.
3. **Dispatch & Hybrid Delegation Frontier**:
   - Small static shapes $\rightarrow$ MLIR in-register unrolling.
   - Large standard operations $\rightarrow$ Authenticated CBLAS / OpenBLAS.
   - Compound fused graphs $\rightarrow$ MLIR structured lowering pipeline.
