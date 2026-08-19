# CPU Native Lowering & Microkernel Findings

## 1. CPU Lowering Archaeology (x86-64 Architecture)

This document records code generation, loop vectorization, FMA contraction, and register allocation behavior observed when lowering canonical GEMM kernels through Clang/LLVM across optimization tiers (`-O0`, `-O2`, `-O3`, `-march=x86-64-v3`, `-march=x86-64-v4`).

---

## 2. Key Code Generation Observations

### A. Impact of Aliasing Annotations (`__restrict__` vs. Unannotated)
* **Unannotated Naive GEMM ([`gemm_f32_naive.c`](../inputs/cpu/gemm_f32_naive.c))**:
  * Clang/LLVM cannot prove at compile time that pointer $C$ does not overlap with pointers $A$ or $B$.
  * *Resulting Code*: At `-O3`, LLVM emits a runtime pointer overlap check (branching to a scalar fallback loop if pointers alias, and executing a vectorized loop only if runtime addresses are disjoint).
* **Restricted GEMM ([`gemm_f32_restrict.c`](../inputs/cpu/gemm_f32_restrict.c))**:
  * The `__restrict__` contract proves disjoint memory spaces.
  * *Resulting Code*: LLVM completely eliminates the runtime alias check prologue, unrolls the innermost $K$ reduction, and directly issues vector FMA chains.

### B. Impact of Alignment Preconditions (`__builtin_assume_aligned`)
* **Without Alignment Proof**: LLVM conservatively emits unaligned vector move instructions (`vmovups`).
* **With 32-Byte Alignment Proof**: LLVM emits aligned memory operations (`vmovaps`), which improves load/store execution port throughput and guarantees zero cache-line split penalties.

### C. Vector Instruction Selection (AVX2 vs. AVX-512)
* **AVX2 / FMA (`-march=x86-64-v3`)**:
  * Uses 256-bit `ymm` registers (8 x `float` per vector).
  * Emits `vbroadcastss` for matrix $A$ scalar broadcasting and `vfmadd213ps` / `vfmadd231ps` for fused multiply-accumulate across `ymm` accumulators.
* **AVX-512 (`-march=x86-64-v4`)**:
  * Uses 512-bit `zmm` registers (16 x `float` per vector).
  * Emits 512-bit FMA instructions (`vfmadd213ps %zmm`) and utilizes 32 architectural vector registers (ZMM0–ZMM31) to maintain larger active accumulation tiles in register space.
