# Adversarial CPU Architecture Red Team Audit

**Role**: Scientific Skeptic / Architecture Red Team  
**Mission**: Deliberately attack, challenge, and attempt to falsify minimal MDSLC ownership and delegation architecture claims using real experiments.

---

## 1. Adversarial Falsification Attacks & Defenses

### ATTACK-ARCH-01: "If Matcore Only Preserves Semantics and Delegates All Lowering to LLVM, It Will Suffer a Severe Performance Cliff on Large Matrices."
* **Skeptic Attack**: If Matcore relies purely on standard LLVM loop vectorization without generating memory packing or 5-loop cache hierarchies, large strided matrix multiplications ($N \ge 256$) will run at $< 30\%$ of hardware peak due to TLB misses and cache thrashing.
* **Corpus Verification**:
  - *Confirmed*: As proven in `packing-cache-atlas.md`, the standard LLVM middle-end loop optimizer does not synthesize global or thread-local memory packing buffers automatically.
* **Architectural Realignment**:
  - Matcore does NOT rely purely on generic LLVM loop lowering for large matrices.
  - The minimal architecture explicitly enforces a **Hybrid Delegation Frontier**:
    1. For compound / fused epilogue operations: Lower via MLIR `vector.contract` (in-register store fusion).
    2. For large standard matrix multiplications ($N \ge 64$): Dispatch directly across the C ABI to an authenticated CBLAS / OpenBLAS provider (`matcore_runtime_gemm_f32_v0`) or expose caller-owned workspace buffers (`sa`, `sb`).

---

### ATTACK-ARCH-02: "Does Upstream MLIR Really Lower to Spill-Free Assembly Without Hand-Crafted Microkernels?"
* **Skeptic Attack**: Can LLVM CodeGen genuinely match hand-written OpenBLAS assembly when starting from MLIR vector contractions, or does it suffer register spills and sub-optimal instruction density?
* **Experimental Verification**:
  - As proven in `mlir-microkernel-validation.md`, when the compiler is supplied with structural tile caps ($M_R \le 16, N_R \le 4$ on AVX2), the LLVM Greedy Register Allocator produces **ZERO inner-loop stack spills**, mapping accumulators directly to `ymm` registers with 100% saturation.
  - *However*, the red team confirmed that OpenBLAS microkernels include hand-tuned software prefetching (`prefetcht0 512(%0)`), un-aligned multi-row load shuffles, and hand-tuned tail cleanup that generic MLIR/LLVM loop lowering does not automatically synthesize.
* **Recalibrated Conclusion**: Hand-crafted assembly is unnecessary for dense, compute-bound FMA loop bodies with bounded unroll tiles, but specialized optimizations (software prefetching, non-unit stride packing, custom tail masks) remain valuable for extreme edge performance.

---

### ATTACK-ARCH-03: "Is Post-Op Epilogue Fusion Genuinely Worth the Complexity of Maintaining an MLIR Dialect?"
* **Skeptic Attack**: Can't users simply call `cblas_sgemm` followed by an OpenMP vector loop for ReLU/Bias? Is the performance difference significant?
* **Experimental Verification**:
  - As measured in `fusion-performance-validation.md`, for square compute-bound GEMMs, the arithmetic cost of the GEMM dominates the execution time ($>99.5\%$), making a fused ReLU deliver minimal whole-operation speedup ($\approx 0.95\text{–}1.0\times$).
  - *However*, on **elementwise memory-bandwidth-bound chains** (Bias + LayerNorm + ReLU + Residual Add), fusing passes eliminates intermediate round-trips through memory, delivering major bandwidth savings.
* **Recalibrated Conclusion**: Post-op fusion provides substantial architectural value on compound operator chains and inference graphs, while pure standalone GEMMs are efficiently served by library dispatch.
