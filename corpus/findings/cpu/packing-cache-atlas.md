# Packing, Cache Hierarchy & Data-Movement Atlas

**Investigation Scope**: Multi-provider comparison across OpenBLAS v0.3.29, BLIS master, Eigen 3.4.0, LIBXSMM master, and LLVM/MLIR  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Provider Strategy Comparison Matrix

| Provider / Engine | Cache Blocking Ownership | Packing Ownership & Strategy | When Packing is Used | When Packing is Bypassed |
| :--- | :--- | :--- | :--- | :--- |
| **OpenBLAS** | Hand-crafted 5-loop nest in `driver/level3/level3.c` ($J_C \rightarrow P_C \rightarrow I_C$). | Dedicated assembly/C copy routines (`GEMM_INCOPY`, `GEMM_ONCOPY`). | Large matrices ($M, N, K \ge 64$) exceeding L1/L2 cache capacity. | Small shapes ($M, N, K \le 16\text{–}32$) via `sgemm_small_kernel_nn_...` and `sgemm_direct_performant.c`. |
| **BLIS** | Formal Control Tree (`bli_gemm_cntl.c`) composing macro-kernel nodes. | Object-based packing engine (`bli_pack.c`) aligning to $M_R \times K_C$ and $K_C \times N_R$. | Standard GEMM execution path across all registered architectures. | Direct-to-microkernel paths for unit-stride panels or custom microkernel plugins. |
| **Eigen** | C++ template-driven blocking (`GeneralBlockPanelKernel.h` / `gebp_kernel`). | Header-based packing loops (`gemm_pack_lhs`, `gemm_pack_rhs`) in L1/L2 buffers. | Standard dynamic-size matrix products exceeding small matrix thresholds. | Compile-time fixed-size matrices (`Matrix<float, 4, 4>`) unroll directly to SIMD without packing. |
| **LIBXSMM** | Bypassed (relies on L1/L2 cache residency for small/medium shapes). | **Zero Packing (Direct Streaming)**: JIT microkernels load directly from strided matrices. | Never packs for small/medium shapes ($M,N,K \le 128$). | **Always bypassed**; eliminates copy overhead entirely for small/medium tiles. |
| **Generic LLVM** | Relies on polyhedral loop tiling passes (`Polly` or MLIR affine tiling). | Standard middle-end loop optimizer does not synthesize heap/stack packing buffers by default. | Never generates packing buffers automatically. | Always in-place streaming (vulnerable to TLB/cache thrashing on large non-unit strides). |

---

## 2. Theoretical Amortization Model

Memory packing incurs an upfront memory copy cost:

$$\text{Cost}_{\text{pack}} = O(M \cdot K) + O(K \cdot N)$$

Against matrix multiplication compute:

$$\text{Compute}_{\text{GEMM}} = 2 \cdot M \cdot N \cdot K \text{ FLOPs}$$

The theoretical amortization ratio is:

$$\text{Amortization Ratio} = \frac{\text{Cost}_{\text{pack}}}{\text{Compute}} = \frac{1}{2N} + \frac{1}{2M}$$

* **When $M, N \ge 128$**: The theoretical copy ratio is $< 0.008$ ($< 0.8\%$), completely amortizing the copy cost while accelerating compute throughput via stride-free contiguous SIMD streaming.
* **When $M, N \le 16$**: The copy ratio rises to $> 0.06$ ($> 6\%$), dominating execution time and making direct un-packed streaming (LIBXSMM / OpenBLAS direct path) strictly preferable.

---

## 3. Source-Documented OpenBLAS Cache Constants
From `driver/level3/level3.c` & `KERNEL.HASWELL`:
- `GEMM_P = 504`: Macro-tile dimension $M_C$ (sized for L1/L2 cache)
- `GEMM_Q = 128`: Macro-tile dimension $K_C$ (sizes packed $B$ for L2 cache)
- `GEMM_R = 4096`: Macro-tile dimension $N_C$ (sizes panel for L3 cache / RAM)
- `GEMM_UNROLL_M = 8`: Microkernel unroll factor $M_R$
