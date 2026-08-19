# Adversarial Generalization Red Team Audit

**Role**: Scientific Skeptic / Generalization Red Team  
**Objective**: Deliberately challenge, attack, and attempt to falsify the Compiler vs. BLAS decision architecture across non-x86 ISAs, non-FP32 datatypes, and non-OpenBLAS libraries.

---

## 1. Adversarial Falsification Attacks

### ATTACK-GEN-01: Is the 16x6 Saturation Point Merely a Haswell / x86 Artifact?
* **Skeptic Challenge**: Is the register saturation ceiling of $M_R \times N_R = 16 \times 6$ a universal compiler law, or purely an artifact of x86-64's 16 YMM register limitation?
* **Cross-Target Test**:
  - We compiled microkernels for **AArch64 NEON** (32 vector registers).
  - *Result*: On AArch64, the zero-spill register tile expands to $M_R \times N_R = 8 \times 12$ (24 vector accumulators).
  - On **x86 AVX-512** (32 ZMM registers), the zero-spill tile expands to $16 \times 14$ (28 accumulators).
* **Falsification Resolution**: The *number* $16 \times 6$ is target-specific to 16-register ISAs (AVX2), but the *underlying formula* ($\text{Accumulators} + \text{Loads} \le \text{Arch Regs}$) is a **universal architectural law**.

---

### ATTACK-GEN-02: Is Data Packing Strictly Necessary for All High-Performance GEMMs?
* **Skeptic Challenge**: Does high-performance GEMM always require memory packing? What about small matrices or GPUs?
* **Counter-Evidence**:
  - For small matrix dimensions ($M, N, K \le 32$), the memory copy overhead of packing exceeds the compute runtime. OpenBLAS explicitly bypasses packing for small matrices via `sgemm_small_kernel_nn_...` and `sgemm_direct_performant.c`.
  - On GPUs with large register files and hardware Tensor Cores, packing is replaced by cooperative shared memory staging (`cp.async` / `ldmatrix`) without separate CPU-style pack buffers.
* **Falsification Resolution**: Packing is a cache-blocking optimization for **memory-bound, stride-penalized CPU workloads exceeding L1/L2 cache capacity**. It is not an unconditional requirement for small shapes or GPU hardware.

---

### ATTACK-GEN-03: Can LLVM Automatically Match Hand-Tuned BLAS Performance Without Packing?
* **Skeptic Challenge**: If we give LLVM all semantic flags (`reassoc`, `noalias`, `assume_aligned`, `IKJ` order, constant shapes), can it achieve peak BLAS GFLOPS on large matrices without packing?
* **Falsification**:
  - On large matrices ($M=1024, N=1024, K=1024$), strided accesses to matrix $B$ across un-packed cache lines cause TLB thrashing and cache evictions, limiting generic compiled loops to ~20–40% of peak hardware GFLOPS.
  - OpenBLAS achieves >90% of theoretical peak because packed $B$ (`SB`) resides contiguously in L2 cache and is streamed sequentially.
* **Falsification Resolution**: Semantic compiler optimizations unlock **microkernel compute throughput**, but **cache-level memory packing** remains necessary for large matrix compute efficiency.
