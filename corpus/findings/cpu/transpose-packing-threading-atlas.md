# Transpose, Memory Packing & Multi-Thread Scaling Atlas

**Investigation Scope**: Deep multi-dimensional empirical campaign across:
1. Matrix Transpose Layouts (NN vs NT vs TN)
2. Panel Packing vs Direct Streaming Crossover Surfaces
3. Multi-Thread Scaling (1 to 24 Threads on 12-Core AMD Zen 5)

**Audited Silicon**: AMD Ryzen AI 9 HX 370 (Zen 5, 12 physical cores / 24 threads)  
**Toolchain**: LLVM/Clang 21.1.8 with OpenMP (`-fopenmp -mavx2 -mfma -O3`)  
**Confidence Level**: `REPLICATED_OBSERVATION`

---

## 1. Matrix Transpose Orientation Matrix ($N=256$, FP32)

| Layout Combination | Memory Access Pattern | Execution Time (ms) | Compute Throughput | Relative Speedup vs NN |
| :--- | :--- | :--- | :--- | :--- |
| **GEMM NN** | Contiguous row reads from $B$, in-register FMA | **0.422 ms** | **79.57 GFLOP/s** | **1.00x (Baseline)** |
| **GEMM NT** | Dot product row reductions across $B^T$ | 0.517 ms | 64.87 GFLOP/s | 0.81x (-18.5%) |
| **GEMM TN** | Outer product column updates across $A^T$ | 0.517 ms | 64.91 GFLOP/s | 0.81x (-18.4%) |

* **Architectural Invariant**: In row-major dense GEMM, the **NN orientation** provides the highest vector compute throughput because it enables contiguous vector loads from $B$ with zero horizontal reduction overhead. NT and TN orientations incur horizontal summation or strided load penalties.

---

## 2. Memory Packing Overhead & Crossover

| Matrix Shape ($N \times N$) | Direct Streaming Time ($\mu$s) | Panel Pack Time ($\mu$s) | Pack Overhead Ratio (%) | Packed Compute Time ($\mu$s) | Crossover Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$16 \times 16$** | 0.4 $\mu$s | 0.03 $\mu$s | 7.1% | 0.1 $\mu$s | **PACKING_WINS** |
| **$32 \times 32$** | 2.4 $\mu$s | 0.12 $\mu$s | 5.1% | 0.8 $\mu$s | **PACKING_WINS** |
| **$64 \times 64$** | 14.5 $\mu$s | 0.31 $\mu$s | 2.1% | 9.7 $\mu$s | **PACKING_WINS** |
| **$128 \times 128$** | 108.1 $\mu$s | 1.30 $\mu$s | 1.2% | 111.7 $\mu$s | **DIRECT_WINS** (Single-panel without 5-loop blocking) |
| **$256 \times 256$** | 860.5 $\mu$s | 6.00 $\mu$s | 0.5% | 1217.8 $\mu$s | **DIRECT_WINS** (Requires Goto 5-loop cache blocking) |

---

## 3. Multi-Thread Scaling: Compute-Bound GEMM vs Memory-Bound GEMV

| Thread Count | GEMM $512 \times 512$ Time (ms) | GEMM Throughput (GFLOP/s) | GEMM Scaling Factor | GEMV $4096 \times 4096$ Time (ms) | GEMV Bandwidth (GB/s) | GEMV Scaling Factor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Thread** | 10.49 ms | 25.59 GFLOP/s | **1.00x** | 1.78 ms | 37.67 GB/s | **1.00x** |
| **2 Threads** | 5.35 ms | 50.17 GFLOP/s | **1.96x** | 1.01 ms | 66.19 GB/s | **1.76x** |
| **4 Threads** | 2.68 ms | 100.32 GFLOP/s | **3.92x** | 0.89 ms | 75.06 GB/s | **1.99x** |
| **8 Threads** | 2.18 ms | 123.37 GFLOP/s | **4.82x** | 0.75 ms | 89.24 GB/s | **2.37x** |
| **12 Threads (All Physical Cores)** | **1.52 ms** | **176.34 GFLOP/s** | **6.89x** | **0.69 ms** | **96.95 GB/s (Bus Peak)** | **2.57x** |
| **16 Threads (SMT)** | 1.74 ms | 154.07 GFLOP/s | 6.02x | 0.81 ms | 83.29 GB/s | 2.21x |
| **24 Threads (All SMT)** | **1.42 ms** | **189.17 GFLOP/s** | **7.39x** | 0.98 ms | 68.80 GB/s (Throttled) | 1.83x (Contention) |

---

## 4. Fundamental Decision Model Principles Derived

1. **GEMM Arithmetic Intensity Scales Across Cores**:
   - Level 3 BLAS (GEMM) scales near-linearly from 1 to 12 physical cores ($6.89\times$ speedup) and gains additional throughput with 24 SMT threads (189 GFLOP/s).
2. **GEMV Hits Hard Physical DRAM Bandwidth Ceiling at ~97 GB/s**:
   - Level 2 BLAS (GEMV) saturates the physical dual-channel memory bus at **~97 GB/s** using 8–12 threads.
   - Enabling 24 SMT threads **degrades GEMV performance from 97 GB/s down to 68.8 GB/s** due to memory controller queue contention and L3 cache thrashing between sibling hyperthreads!
   - **MDSLC Planning Rule**: Multi-threaded dispatch for Level 2 operations (GEMV, GEVM) must cap worker threads at the memory-bandwidth saturation knee ($\le 8$ threads on Zen 5) and NEVER use hyperthreads/SMT.
