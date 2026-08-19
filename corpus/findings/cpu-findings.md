# CPU Native Lowering & Microkernel Findings

**Confidence Class**: `STRONGLY_SUPPORTED`  
**Evidence Surface**: LLVM 20.1.8, 21.1.8, 22.1.8; 5 CPU diagnostic kernels; Optimization remarks; Target-aware AVX2/AVX-512 assembly.

---

## 1. Loop Vectorization & Inner Reduction Behavior

### OBSERVED:
- In unannotated naive GEMM ([`gemm_f32_naive.c`](../inputs/cpu/gemm_f32_naive.c)) and restricted GEMM ([`gemm_f32_restrict.c`](../inputs/cpu/gemm_f32_restrict.c)), Clang's LoopVectorizer vectorizes the **outer parallel column loop $j$** (vectorization width 8, unroll factor 4 on AVX2; vectorization width 16 on AVX-512).
- The **inner reduction loop $k$** is **not vectorized** across scalar accumulators. Compiler optimization remarks explicitly report:
  `"cannot prove it is safe to reorder floating-point operations; allow reordering by specifying '#pragma clang loop vectorize(enable)' before the loop or by providing the compiler option '-ffast-math' [-Rpass-analysis=loop-vectorize]"`.

### INFERRED:
- Standard C/C++ IEEE 754 precision semantics require strict left-to-right floating-point addition order. Without explicit permission to reassociate operations in the reduction dimension, the compiler cannot legally transform scalar summation into a horizontal SIMD reduction vector.

### UNRESOLVED:
- The extent to which loop interchange heuristics in upstream MLIR affine passes (e.g. converting $i, j, k \rightarrow i, k, j$) interact with C++ compiler vectorization across different optimization levels without explicit pragma directives.

### ARCHITECTURAL IMPLICATION:
- MatcoreDSL's explicit eDSL policy (`explicit-gemm-f32-v1`) must explicitly authorize reassociation within the contraction dimension at the semantic MLIR level (`mdsl.gemm`), freeing upstream MLIR passes to generate vector contractions without relying on global unsafe `-ffast-math` compiler flags.

---

## 2. Aliasing Annotations & Control Flow

### OBSERVED:
- In unannotated GEMM, Clang emits runtime pointer overlap checks (`subq`, `cmpq`, `jae`) branching to a scalar fallback loop if buffers overlap.
- In `__restrict__` annotated GEMM, Clang eliminates the runtime alias check prologue entirely, executing the vectorized loop directly. Scalar loop peeling/remainder branches remain for loop bounds not divisible by the vector width.

### INFERRED:
- Alias-free contracts eliminate runtime verification overhead and simplify control flow graphs.

### ARCHITECTURAL IMPLICATION:
- In MatcoreDSL, `out(C)` establishes write-only destination semantics. When static proof guarantees non-aliasing, backend guards can be safely omitted; when unproven, dominating fail-closed guards must execute prior to destination mutation.

---

## 3. Target-Aware vs. Backend-Retargeted Instruction Selection

### OBSERVED:
- **Backend Retargeting Only** (`generic O3.ll -> llc -mattr=+avx2`): Produces 128-bit `XMM` instructions with VEX prefixes (`YMM = 0`, `ZMM = 0`). The vectorizer operates on generic 128-bit SSE defaults.
- **Frontend Target-Aware Compilation** (`clang -O3 -mavx2 -mfma`): Emits true 256-bit `YMM` vector instructions (26 YMM registers in `gemm_f32_tiled`).
- **Frontend Target-Aware AVX-512** (`clang -O3 -mavx512f -mavx512dq -mavx512vl`): Emits true 512-bit `ZMM` vector instructions (14 ZMM registers in `gemm_f32_tiled`).

### INFERRED:
- Vector width and register pressure heuristics are frozen during the middle-end LoopVectorizer pass. Target flags must be supplied at frontend optimization time.

### ARCHITECTURAL IMPLICATION:
- MDSLC's Clang driver and MLIR target pipeline must pass explicit target architecture and ISA feature flags (`-target-cpu`, `-target-feature`) at all optimization stages.
