# Information-Unlock Map: What Unlocks What in LLVM

**Investigation Scope**: Controlled GEMM variant suite on LLVM 21.1.8 (AVX2/FMA)  
**Confidence Level**: `STRONGLY_SUPPORTED` (Empirical step-by-step counterfactual compilation)

---

## 1. The Step-by-Step Information Unlock Matrix

| Variant & Step | Fact Introduced | LLVM Optimization Decision | Assembly Consequence | Unlocked Capability |
| :--- | :--- | :--- | :--- | :--- |
| **01_naive** | None (Dynamic C99 IJK GEMM) | Scalar fallback loop; inner reduction rejected. | 0 YMM ops, scalar `vfmadd231ss` | Baseline sequential scalar loop |
| **02_restrict** | Disjoint memory spaces (`__restrict__`) | Eliminates `vector.memcheck` runtime pointer overlap checks. | 0 YMM ops; removes alias branch prologue | **Control-flow simplification** (eliminates alias guards) |
| **03_aligned** | 32-Byte pointer alignment (`assume_aligned`) | Selects aligned vector loads for legal paths. | Remains scalar in IJK reduction | **Aligned memory operations** (when vectorization is active) |
| **04_reassoc_pragma** | **Floating-Point Reassociation Permission** (`#pragma vectorize`) | **Vectorizes Inner K Reduction Loop** with $VF=8$. | **32 YMM vector instructions**, 15 `vmovaps`, 10 `vfmadd231ps` | **Horizontal SIMD vector reduction** |
| **05_loop_order_ikj** | **Loop Interchange ($I, J, K \rightarrow I, K, J$)** | **Vectorizes Inner Loop without Fast-Math** ($VF=8$). | **17 YMM vector instructions**, 6 `vfmadd231ps` | **Parallel column accumulation** (bypasses FP reduction blocker) |
| **06_constant_dims** | **Static Constant Trip Counts ($M=N=K=64$)** | Proves trip count divisible by $VF$; eliminates loop peeling. | **33 YMM vector instructions**, zero remainder branches | **Complete loop unrolling & zero-peeling vector body** |
| **07_tiled** | Cache-blocked 32x32x64 loop nest | Unrolls 2D micro-panels within cache boundaries. | 26 YMM vector instructions | **L1/L2 cache residency** |

---

## 2. Key Synthesis: The Hierarchy of Semantic Information

1. **The Primary Blocker is Floating-Point Semantics, NOT Aliasing**:
   - In standard row-major $I, J, K$ order, the inner $K$-loop is a scalar reduction. Strict IEEE 754 ordering forbids parallel vector reduction. Reassociation permission (`reassoc` or `#pragma`) is the single key that unlocks inner reduction vectorization.
2. **Loop Ordering Bypasses Semantic Limitations**:
   - Reordering the loop nest to $I, K, J$ converts the innermost loop into a parallel elementwise addition (`C[j] += a * B[j]`), making vectorization 100% legal under strict IEEE 754 math without needing reassociation!
3. **Static Shapes Eliminate Loop Peeling Overhead**:
   - Exposing static shapes ($M, N, K$) or guaranteed alignment multiples unlocks complete loop unrolling and eliminates scalar cleanup code.
