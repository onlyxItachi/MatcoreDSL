# OpenBLAS SGEMM Execution & Source Anatomy Map

**Source Revision**: OpenBLAS v0.3.29 (`8795fc7985635de1ecf674b87e2008a15097ffab`)  
**Target Architectures Examined**: Haswell (AVX2+FMA), Skylake-X (AVX-512), Zen (AVX2), Neoverse/ARMv8 (AArch64)  
**Confidence Level**: `STRONGLY_SUPPORTED` (Direct source inspection and call-graph reconstruction)

---

## 1. Complete End-to-End Call Graph

```text
CBLAS API (`cblas_sgemm` in `interface/gemm.c`)
    │
    ▼ (Argument validation, Transpose flag resolution, NaN/Zero fast returns)
Level-3 Dispatcher (`driver/level3/gemm.c`)
    │
    ├── Small Shape Fast Path (if M*N*K <= Threshold -> `sgemm_small_kernel_nn_...`)
    └── Standard GotoBLAS 5-Loop Engine:
            │
            ├── Thread Partitioning (`driver/level3/gemm_thread_m.c` / `gemm_thread_n.c`)
            │       │ (Partitions M or N across worker threads with caller/thread workspace SA, SB)
            │       ▼
            └── Core Cache-Blocking Loop Nest (`driver/level3/level3.c`):
                    │
                    ├── [Loop 5: Outer N] Iterate j = 0..N step GEMM_R (fits in L3 / RAM)
                    │     │
                    │     ├── [Loop 4: Outer K] Iterate l = 0..K step GEMM_Q (fits in L2 / LLC)
                    │     │     │
                    │     │     ├── [Beta Scaling] Scale C by beta on first K-slice (`BETA_OPERATION`)
                    │     │     │
                    │     │     ├── [Pack B] Pack B panel into SB (`GEMM_ONCOPY` / `gemm_oncopy_haswell.S`)
                    │     │     │
                    │     │     └── [Loop 3: Outer M] Iterate i = 0..M step GEMM_P (fits in L1/L2)
                    │     │           │
                    │     │           ├── [Pack A] Pack A panel into SA (`GEMM_INCOPY` / `gemm_ncopy_8.c`)
                    │     │           │
                    │     │           └── [Loops 2 & 1: Microkernel] (`KERNEL_OPERATION`)
                    │     │                 │
                    │     │                 └── `sgemm_kernel_8x4_haswell_2.c` / `16x4.S`:
                    │     │                       - 2D Register Tile (e.g. 16x6 or 8x4)
                    │     │                       - Unrolled K-loop with FMA instructions
                    │     │                       - Sequential vector loads from prepacked SA, SB
                    │     │                       - In-register alpha multiplication + C writeback
```

---

## 2. Detailed Execution Phases

### Phase A: Interface & Normalization (`interface/gemm.c`)
* Validates matrix dimensions ($M, N, K \ge 0$), leading dimensions ($lda \ge M$, $ldb \ge K$, $ldc \ge M$).
* Evaluates quick returns: if $M=0$ or $N=0$, returns immediately; if $K=0$ or $\alpha=0$, dispatches purely to `BETA_OPERATION` (scaling $C \leftarrow \beta C$) and returns.

### Phase B: Multi-Threading & Partitioning (`driver/level3/gemm_thread_m.c`)
* Slices either $M$ (thread-M) or $N$ (thread-N) across available worker threads in the execution pool.
* Assigns distinct, non-overlapping destination sub-tiles of $C$ to each thread.
* Each thread receives dedicated, contiguous workspace pointers `sa` and `sb`.

### Phase C: Goto 5-Loop Cache Blocking Architecture (`driver/level3/level3.c`)
The classical GotoBLAS algorithm enforces strict memory hierarchy alignment:
1. **Loop 5 ($J_C$)**: Blocks $N$ by `GEMM_R` ($N_C \approx 2048–4096$), maximizing reuse of packed $A$ across large $N$ slices.
2. **Loop 4 ($P_C$)**: Blocks $K$ by `GEMM_Q` ($K_C \approx 256–512$), sizing packed $B$ to fit comfortably in the CPU L2/L3 cache.
3. **Pack Matrix B**: Invokes `GEMM_ONCOPY` to copy column/row-major $B$ into contiguous micro-panels ($K_C \times N_R$).
4. **Loop 3 ($I_C$)**: Blocks $M$ by `GEMM_P` ($M_C \approx 64–128$), sizing packed $A$ to fit in L1/L2 cache.
5. **Pack Matrix A**: Invokes `GEMM_INCOPY` to copy $A$ into contiguous micro-panels ($M_R \times K_C$).
6. **Microkernel Execution**: Calls `KERNEL_OPERATION` to compute the matrix product on cached, prepacked $A$ and $B$.

---

## 3. Microkernel Architecture: Haswell AVX2 Case Study

In `kernel/x86_64/sgemm_kernel_8x4_haswell_2.c`:
* **Register Tile Geometry**: $M_R = 16, N_R = 6$ or $M_R = 8, N_R = 4$.
* **Vector Register Layout (16 YMM Registers Total)**:
  * `ymm0`: Dedicated to scalar parameter $\alpha$.
  * `ymm1`, `ymm2`: Matrix $A$ vector loads (`vmovups (%0)` / `vmovsldup`).
  * `ymm3`: Matrix $B$ scalar broadcasts (`vbroadcastss` / `vbroadcastsd`).
  * `ymm4`–`ymm15`: **12 accumulator registers** holding the $2\text{D}$ register tile.
* **Spill / Reload Count**: **Zero spills** in the inner compute loop.
* **Prefetching**: Explicit software prefetch instructions (`prefetcht0 512(%0)`) pipeline subsequent buffer lines into L1 cache ahead of compute.
