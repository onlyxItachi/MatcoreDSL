# 5-Operation Runtime Characterization Atlas

**Investigation Scope**: Multi-shape, multi-provider empirical execution sweep across the 5 fundamental CPU matrix/vector operation families:
1. `GEMM` ($C = A \times B + C$)
2. `GEMV` ($y = A \cdot x$)
3. `GEVM` ($y^T = x^T \cdot A$)
4. `GEVV-INNER` ($s = x^T \cdot y$)
5. `GEVV-OUTER` ($A += x \cdot y^T$)

**Audited Silicon**: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores, AVX2/FMA/AVX-512)  
**Toolchain**: LLVM/Clang 21.1.8, Eigen 3.4.0  
**Confidence Level**: `REPLICATED_OBSERVATION`

---

## 1. Measured Performance Across Operation Families

| Operation | Shape | Provider / Implementation | Median Time (ns) | Throughput (GFLOP/s) | Effective BW (GB/s) | Correctness Oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GEMM FP32** | $16 \times 16$ | 1D Tiled SIMD (C) | 500 ns | 16.38 GFLOP/s | 6.14 GB/s | **PASS (Error < 1e-5)** |
| **GEMM FP32** | $16 \times 16$ | Eigen3 (2D Microkernel) | **100 ns** | **81.92 GFLOP/s** | **30.72 GB/s** | **PASS (Error < 1e-5)** |
| **GEMM FP32** | $64 \times 64$ | 1D Tiled SIMD (C) | 14,600 ns | 35.91 GFLOP/s | 3.37 GB/s | **PASS (Error < 1e-5)** |
| **GEMM FP32** | $64 \times 64$ | Eigen3 (2D Microkernel) | **4,600 ns** | **113.98 GFLOP/s** | **10.69 GB/s** | **PASS (Error < 1e-5)** |
| **GEMM FP32** | $256 \times 256$ | 1D Tiled SIMD (C) | 853,400 ns | 39.32 GFLOP/s | 0.92 GB/s | **PASS (Error < 1e-5)** |
| **GEMM FP32** | $256 \times 256$ | Eigen3 (2D Microkernel) | **374,300 ns** | **89.65 GFLOP/s** | **2.10 GB/s** | **PASS (Error < 1e-5)** |
| **GEMM FP32** | $512 \times 512$ | 1D Tiled SIMD (C) | 6,856,300 ns | 39.15 GFLOP/s | 0.46 GB/s | **PASS (Error < 1e-5)** |
| **GEMM FP32** | $512 \times 512$ | Eigen3 (2D Microkernel) | **2,473,200 ns** | **108.54 GFLOP/s** | **1.27 GB/s** | **PASS (Error < 1e-5)** |
| **GEMV FP32** | $64 \times 64$ | SIMD Row Reduction | 200 ns | 40.96 GFLOP/s | 84.48 GB/s (L1/L2) | **PASS (Error < 1e-5)** |
| **GEVM FP32** | $64 \times 64$ | SIMD Col Accumulation | 200 ns | 40.96 GFLOP/s | 84.48 GB/s (L1/L2) | **PASS (Error < 1e-5)** |
| **GEMV FP32** | $16 \times 4096$ | SIMD Row Reduction | 9,700 ns | 13.51 GFLOP/s | 28.72 GB/s | **PASS (Error < 1e-5)** |
| **GEVM FP32** | $16 \times 4096$ | SIMD Col Accumulation | **6,100 ns** | **21.49 GFLOP/s** | **45.67 GB/s** | **PASS (Error < 1e-5)** |
| **GEMV FP32** | $4096 \times 4096$| SIMD Row Reduction | 2,808,300 ns | 11.95 GFLOP/s | 23.91 GB/s (DRAM) | **PASS (Error < 1e-5)** |
| **GEVM FP32** | $4096 \times 4096$| SIMD Col Accumulation | **2,125,300 ns** | **15.79 GFLOP/s** | **31.59 GB/s (DRAM)**| **PASS (Error < 1e-5)** |

---

## 2. Key Empirical Architectural Findings

1. **2D Register Microkernel Tile vs 1D Row Vectorization**:
   - 1D row vectorization (loading 1 scalar of $A$ and streaming 1 row of $B$ into $C$) saturates at **~39 GFLOP/s** on a single core due to load-port stalls and limited accumulator reuse.
   - Eigen3's 2D register accumulator tile ($M_R \times N_R = 8 \times 4$ / $16 \times 4$) with memory packing achieves **108–114 GFLOP/s ($>2.8\times$ faster)** by amortizing memory load bandwidth over $M_R \times N_R$ parallel FMAs.
2. **GEMV vs GEVM Memory-Access Asymmetry**:
   - For row-major storage:
     * **GEMV ($y = A \cdot x$)** is reduction-bound (each row computes a dot product requiring horizontal reduction across $K$).
     * **GEVM ($y^T = x^T \cdot A$)** is broadcast-accumulation-bound (each element $x_i$ broadcasts across an entire contiguous row of $A$ and accumulates into $y$).
     * On short-wide shapes ($16 \times 4096$), **GEVM is $1.6\times$ faster than GEMV** (6.10 $\mu$s vs 9.70 $\mu$s) because vector broadcasting completely eliminates horizontal reductions.
3. **Memory Bandwidth Bottleneck Transition**:
   - In GEMV/GEVM, memory bandwidth peaks at **80–84 GB/s** when data resides in L1/L2 cache ($N \le 256$), and throttles to **24–31 GB/s** at $4096 \times 4096$ (64 MB matrix) when operations become memory-bus bound by physical DRAM bandwidth.
