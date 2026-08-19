# Semantic Information Unlock & FP64 Execution Atlas

**Investigation Focus**: Empirical evaluation of compiler semantic signals (`noalias`, alignment, static shapes, strides) and comprehensive Double-Precision (FP64) execution across Zen 5 hardware.  
**Audited Toolchain**: LLVM/Clang 21.1.8 with AVX2/FMA/AVX-512  
**Confidence Level**: `REPLICATED_OBSERVATION`

---

## 1. Compiler Semantic Unlock Findings

| Semantic Factor | Operation & Shape | Semantic Variant | Measured Time (ns) | Effective Throughput | Performance Consequence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Static Shape** | GEMM $4 \times 4$ | Compile-time constant | **2.51 ns** | **50.99 GFLOP/s** | **6.04x speedup** vs dynamic loop (15.14 ns) |
| **Static Shape** | GEMM $8 \times 8$ | Compile-time constant | **12.70 ns** | **80.62 GFLOP/s** | **7.16x speedup** vs dynamic loop (91.02 ns) |
| **Static Shape** | GEMM $16 \times 16$ | Compile-time constant | **88.85 ns** | **92.20 GFLOP/s** | **5.70x speedup** vs dynamic loop (506.13 ns) |
| **Pointer Aliasing**| GEMM $64 \times 64$ | `__restrict__` (NoAlias) | 12,596.75 ns | 41.62 GFLOP/s | 1.01x (Negligible runtime penalty on large shapes) |
| **32-Byte Alignment**| GEMM $256 \times 256$ | Aligned (`_mm256_load_ps`) | 434,742.00 ns | 77.18 GFLOP/s | 1.07% faster than unaligned loadu |
| **Stride** | Dot Product $N=16384$| Contiguous (Stride 1) | 12,685.48 ns | 10.33 GB/s | Contiguous stream saturates L1/L2 prefetcher |

---

## 2. 2D Register Microkernel Tile Search ($M_R \times N_R$ on 16 YMM Budget)

| Tile Geometry ($M_R \times N_R$) | Live YMM Accumulators | Fractional Budget | Microkernel Time (ns) | Peak Compute Throughput | Register Spill Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$8 \times 4$** | 4 YMM | 25% budget | 193.73 ns | 84.57 GFLOP/s | **0 Spills** |
| **$16 \times 4$** | 8 YMM | 50% budget | 235.19 ns | 139.33 GFLOP/s | **0 Spills** |
| **$16 \times 6$** | 12 YMM | **75% budget** | **302.23 ns** | **162.63 GFLOP/s (Peak)**| **0 Spills** |
| **$24 \times 4$** | 12 YMM | 75% budget | 615.70 ns | 79.83 GFLOP/s | 0 Spills (Load-port stall) |
| **$32 \times 4$** | 16 YMM | 100% budget | 1178.06 ns | 55.63 GFLOP/s | **Spill Cliff (-65% drop)** |

---

## 3. Double-Precision (FP64) Operation Suite

| Operation Class | Matrix / Vector Shape | Measured Time (ns) | Compute Throughput | Bandwidth Throughput | Hardware Limit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GEMM FP64** | $16 \times 16$ | 501.14 ns | 16.35 GFLOP/s | 12.26 GB/s | 4-wide vector limit |
| **GEMM FP64** | $64 \times 64$ | 19,317.90 ns | 27.14 GFLOP/s | 5.09 GB/s | 4-wide vector limit |
| **GEMM FP64** | $256 \times 256$ | 1,006,974.00 ns | 33.32 GFLOP/s | 1.56 GB/s | 4-wide vector limit |
| **GEMV FP64** | $64 \times 64$ | 344.65 ns | 23.77 GFLOP/s | 95.08 GB/s (L1/L2) | Cache bandwidth peak |
| **GEVM FP64** | $64 \times 64$ | 426.36 ns | 19.21 GFLOP/s | 76.85 GB/s (L1/L2) | Cache bandwidth peak |
| **GEMV FP64** | $1024 \times 1024$ | 183,020.40 ns | 11.46 GFLOP/s | 45.83 GB/s | Row reduction stalls |
| **GEVM FP64** | $1024 \times 1024$ | **128,808.10 ns** | **16.28 GFLOP/s** | **65.12 GB/s** | **1.42x faster than GEMV** |
