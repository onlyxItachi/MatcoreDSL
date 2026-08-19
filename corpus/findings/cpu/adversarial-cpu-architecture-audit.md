# Adversarial CPU Architecture Red Team Audit

**Role**: Scientific Skeptic / Architecture Red Team  
**Mission**: Deliberately attack, challenge, and attempt to falsify the minimal MDSLC ownership and delegation architecture.

---

## 1. Adversarial Falsification Attacks & Defenses

### ATTACK-ARCH-01: "If Matcore Only Preserves Semantics and Delegates All Lowering to LLVM, It Will Suffer a Severe Performance Cliff on Large Matrices."
* **Skeptic Challenge**: If Matcore relies purely on LLVM loop vectorization without generating memory packing or 5-loop cache hierarchies, large strided matrix multiplications ($N \ge 256$) will run at $< 30\%$ of hardware peak due to TLB misses and cache thrashing.
* **Corpus Verification**:
  - *Confirmed*: As proven in `packing-cache-atlas.md`, LLVM cannot synthesize global memory packing buffers automatically.
* **Architectural Correction**:
  - Matcore does NOT rely purely on generic LLVM loop lowering for large matrices.
  - The minimal architecture explicitly includes a **Hybrid Delegation Frontier**:
    1. For compound / fused epilogue operations: Lower via MLIR `vector.contract` (in-register store fusion).
    2. For large standard matrix multiplications: Expose caller-owned workspace buffers (`sa`, `sb`) for pre-packed evaluation OR dispatch directly to an authenticated CBLAS / OpenBLAS provider (`matcore_runtime_gemm_f32_v0`).

---

### ATTACK-ARCH-02: "Does Upstream MLIR Really Lower to Spill-Free Assembly Without Hand-Crafted Microkernels?"
* **Skeptic Challenge**: Can LLVM CodeGen genuinely match hand-written OpenBLAS assembly when starting from MLIR vector contractions, or does it inevitably suffer register spills?
* **Corpus Verification**:
  - As proven in `register-pressure-findings.md` and `register-pressure-generalization.md`, when the compiler is supplied with structural tile caps ($M_R \le 16, N_R \le 6$ on AVX2; $M_R \le 32, N_R \le 6$ on AVX-512), the LLVM Greedy Register Allocator produces **ZERO inner-loop stack spills**, mapping accumulators directly to `ymm4`–`ymm15` / `zmm` registers with 100% saturation.
* **Conclusion**: Hand-crafting thousands of assembly microkernels is completely unnecessary when structural vector unrolling caps are enforced.

---

### ATTACK-ARCH-03: "Is Post-Op Epilogue Fusion Genuinely Worth the Complexity of Maintaining an MLIR Dialect?"
* **Skeptic Challenge**: Can't users simply call `cblas_sgemm` followed by an OpenMP vector loop for ReLU/Bias? Is the performance difference significant?
* **Corpus Verification**:
  - Standard CBLAS followed by a separate activation loop requires writing the entire $M \times N$ matrix to RAM, then reading it back into CPU cache, executing ReLU, and writing it back to RAM.
  - This incurs $2 \times M \times N \times 4$ bytes of redundant memory bus traffic.
  - Fusing ReLU into the register writeback stage before storing to memory eliminates 100% of this traffic, delivering up to a **$2\times$ speedup on memory-bandwidth-bound inference workloads**.
* **Conclusion**: Fused epilogue support is the decisive technical justification for Matcore's semantic compilation pipeline.
