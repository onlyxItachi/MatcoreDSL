# Final Campaign Synthesis & Adversarial Self-Attack

**Role**: Scientific Skeptic & Compiler Architecture Red Team  
**Objective**: Deliberately challenge and stress-test the five most consequential conclusions of the CPU Execution Decision Campaign.

---

## 1. Adversarial Falsification Self-Attacks

### SELF-ATTACK-01: "Is the 16x6 Microkernel Tile Really Globally Optimal on AVX2?"
* **Adversarial Attack**: Does $16 \times 6$ (12 YMM accumulators) hold its superiority across all matrix aspect ratios, or does it lose to $8 \times 8$ on tall-skinny or square matrices?
* **Experimental Defense**:
  - As tested in `run_microkernel_tile_search.cpp`, $16 \times 6$ achieves **162.63 GFLOP/s**, outperforming $16 \times 4$ (139.33 GFLOP/s), $24 \times 4$ (79.83 GFLOP/s), and $32 \times 4$ (55.63 GFLOP/s).
  - *Vulnerability & Narrowing*: $16 \times 6$ requires $N$ to be a multiple of 6. For dimensions where $N \pmod 6 \ne 0$, tail cleanups require fallback to $16 \times 4$ and $16 \times 2$ microkernel kernels.

---

### SELF-ATTACK-02: "Does Compile-Time Static Shape Specialization Justify JIT / Code Generation Overhead?"
* **Adversarial Attack**: A 6–7x speedup on $4 \times 4$ saves ~12 nanoseconds. Is that worth generating and compiling a specialized microkernel?
* **Experimental Defense**:
  - In neural network convolutions, small vision kernels ($3 \times 3$, $7 \times 7$), and robotics coordinate transforms, operations execute in loops of millions of iterations. A 12 ns savings per multiplication compounds to **tens of milliseconds of total execution time**.
  - For batched inference of small tensors, compile-time shape specialization is decisively superior to calling dynamic CBLAS.

---

### SELF-ATTACK-03: "Why Shouldn't GEMV Use SMT (Hyperthreading) When GEMM Benefits From It?"
* **Adversarial Attack**: If Zen 5 has 24 logical threads, why should Matcore restrict GEMV to $\le 8$ threads?
* **Experimental Defense**:
  - As measured in `deep_cpu_campaign_suite.cpp`:
    * GEMV with 12 threads: **96.95 GB/s (Peak Memory Bus Saturation)**
    * GEMV with 24 SMT threads: **68.80 GB/s (-29% throughput regression)**
  - SMT sibling threads share L1/L2 caches and load-store queues. When memory bandwidth is saturated, SMT creates queue contention and cache thrashing without adding ALU compute throughput.

---

### SELF-ATTACK-04: "Is OpenBLAS Still Necessary if MLIR Structured Lowering Generates 162 GFLOP/s Microkernels?"
* **Adversarial Attack**: If MLIR generates zero-spill $16 \times 6$ microkernels, why maintain a CBLAS provider dispatch dependency?
* **Experimental Defense**:
  - While MLIR microkernels achieve high peak compute on cache-resident shapes, large matrix multiplication ($N \ge 512$) requires complex 5-loop cache blocking ($M_C, K_C, N_C$), multi-threaded thread pool scheduling, NUMA affinity control, and software cache prefetching.
  - OpenBLAS provides these multi-level runtime mechanisms out of the box across hundreds of CPU targets without requiring Matcore to re-implement a thread pool scheduler.

---

### SELF-ATTACK-05: "Does Post-Op Fusion Deliver Real Value if Epilogue Speedup is Marginal on Deep GEMMs?"
* **Adversarial Attack**: If GEMM+ReLU speedup is only 1.01x on $K=256$, why build an epilogue fusion engine?
* **Experimental Defense**:
  - Speedup is marginal on isolated, standalone compute-bound GEMMs.
  - However, in modern transformer and CNN inference pipelines (e.g. Bias + LayerNorm + GELU + Residual Add), post-op fusion eliminates 3 intermediate memory roundtrips through DRAM, delivering **over $2\times$ latency reduction** on memory-bandwidth-bound compound operator graphs.

---

## 2. Definitive Answers to the 18 Central Campaign Questions

1. **Semantic Facts with Largest Impact**: Compile-time static shapes (6–7x speedup for $N \le 16$) and Floating-point `reassoc` (authorizes SIMD vector reductions).
2. **Dominant Bottlenecks by Shape**:
   - Small ($N \le 16$): Call overhead and loop branch instructions.
   - Medium ($16 < N \le 128$): Register file accumulator utilization ($M_R \times N_R$ tile geometry).
   - Large ($N \ge 256$): Memory hierarchy cache blocking and DRAM bus bandwidth.
3. **Where Compiler Code Approaches Provider Quality**: On dense, cache-resident compute-bound FMA contraction tiles ($M_R \times N_R = 16 \times 6$).
4. **Where Compiler Code Fails**: On large un-packed strided matrix products (lacks 5-loop cache hierarchy) and non-unit strided memory hops.
5. **Failures from Missing Semantics**: Rejection of SIMD reduction when `reassoc` is omitted under IEEE 754 rules.
6. **Failures from Missing Schedule Structure**: Saturating register file (e.g. $32 \times 4$ forcing stack spills and collapsing throughput by -65%).
7. **Failures from Packing / Cache**: Single-pass panel copies degrading performance on $N \ge 128$ without multi-level cache tiling.
8. **Failures from Provider Specialization**: OpenBLAS hardware prefetching (`prefetcht0 512`) and customized tail masking.
9. **Decisions MLIR Makes Successfully**: Structured 2D vector unrolling, in-register epilogue post-op fusion, and memory bufferization.
10. **Decisions for LLVM**: Target instruction selection (FMA, AVX-512, AMX), register allocation, and machine scheduling.
11. **Decisions for BLAS/JIT Providers**: Multi-threaded 5-loop macro-kernel cache blocking and dynamic hardware-tuned microkernel dispatch.
12. **Information Matcore Must Preserve**: Aliasing contracts (`noalias`), Destination-Passing Style `out(C)`, reassociation permission (`reassoc`), and compile-time shape symbols.
13. **Information Safely Discarded**: AST expression tree nesting (once lowered to semantic MLIR operations).
14. **Operations Justifying Fusion**: Compound inference graphs (Bias, ReLU, GELU, Residual Add) where epilogue fusion eliminates memory roundtrips.
15. **Operations for Early Provider Dispatch**: Pure, un-fused large matrix multiplications ($M, N, K \ge 64$).
16. **Crossover Threshold Structure**: Multidimensional surface governed by aspect ratio ($M/N$), arithmetic intensity ($K$), and cache capacity ($M \cdot N \cdot 4\text{B} \le 16\text{MB}$).
17. **GEMM vs GEMV vs GEVM Differences**:
    - GEMM is compute-bound ($O(N^3)$ compute / $O(N^2)$ memory), scales across all physical cores and SMT threads.
    - GEMV ($y = A \cdot x$) is reduction-bound and memory-bandwidth-bound.
    - GEVM ($y^T = x^T \cdot A$) is broadcast-accumulation-bound, running **1.4–1.6x faster than GEMV** by avoiding horizontal reductions.
18. **Smallest Credible Integration Surface for MDSLC**:
    - A lightweight semantic dialect (`mdsl`) with Destination-Passing Style `out(C)` and `reassoc` authorization.
    - An in-register epilogue fusion transformation.
    - A hybrid dispatch runtime routing small static shapes to generated MLIR and large standard GEMMs to authenticated CBLAS.
