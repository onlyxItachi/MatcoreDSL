# Adversarial Red Team Audit (Iteration 3)

**Role**: Corpus Red Team / Scientific Skeptic  
**Objective**: Actively attack, falsify, and stress-test all high-confidence claims across the MatcoreDSL Windows Lowering Corpus.

---

## 1. Adversarial Falsification Log

### ATTACK-001: Is the Multi-Stage Lowering Ladder an "Architectural Invariant"?
* **Claim Attacked**: [CLAIM-001](claim-registry.jsonl) ("Tensor contractions follow an immutable multi-stage lowering ladder").
* **Skeptic Challenge**: Is this an immutable architectural invariant of compiler design, or merely one structured lowering path among several alternatives?
* **Counter-Evidence**: 
  - Direct runtime provider dispatch (e.g., Matcore JIT v0 / OpenBLAS / oneMKL / prepacked microkernels) bypasses Linalg/Vector lowering entirely and emits direct C ABI calls (`matcore_runtime_gemm_f32_v0`).
  - LLVM itself supports direct LLVM IR loop synthesis without MLIR.
* **Resolution**: The claim was overstated. Multi-stage lowering is an invariant **of the structured upstream MLIR optimization pipeline**, not of matrix compilation in general.
* **Confidence Recalibration**: Downgraded from `ARCHITECTURAL_INVARIANT` $\rightarrow$ `STRONGLY_SUPPORTED`.

---

### ATTACK-002: Does Destination-Passing Style (DPS) Unconditionally Eliminate Defensive Copies?
* **Claim Attacked**: [CLAIM-002](claim-registry.jsonl) ("DPS unconditionally prevents memory allocations during bufferization").
* **Skeptic Challenge**: Does DPS guarantee zero copies in all cases?
* **Counter-Evidence**:
  - If a tensor operand has overlapping read-write slices or complex non-identity affine indexing maps, One-Shot Bufferization fails in-place bufferization and inserts defensive `memref.alloc` / `memref.copy`.
* **Resolution**: DPS eliminates copies **only when destination writes are disjoint and operand indexing maps allow in-place alias proofs**.
* **Confidence Recalibration**: Downgraded from `ARCHITECTURAL_INVARIANT` $\rightarrow$ `STRONGLY_SUPPORTED`.

---

### ATTACK-003: Did Aliasing Block Naive GEMM Inner Loop Vectorization?
* **Claim Attacked**: [CLAIM-012](claim-registry.jsonl) ("Aliasing is the primary blocker of naive GEMM reduction vectorization").
* **Skeptic Challenge**: Is pointer aliasing the real blocker?
* **Falsification**:
  - Compiler optimization remarks (`-Rpass-analysis=loop-vectorize`) explicitly state:
    `"cannot prove it is safe to reorder floating-point operations; allow reordering by specifying '#pragma clang loop vectorize(enable)' before the loop or by providing the compiler option '-ffast-math'"`.
  - The outer parallel column loop $j$ was successfully vectorized (`vectorization width: 8, interleaved count: 4`).
* **Resolution**: The previous claim was **factually incorrect / misattributed**. Floating-point non-associativity (strict IEEE 754 precision) is the primary blocker for inner $k$ reduction vectorization in C99.
* **Confidence Recalibration**: Marked **CONTRADICTED & REWRITTEN**.

---

### ATTACK-004: Were Register Claims in Baseline Corpus True?
* **Claim Attacked**: [CLAIM-010](claim-registry.jsonl) & [CLAIM-011](claim-registry.jsonl) ("AVX2 baseline uses 256-bit YMM; AVX-512 uses 512-bit ZMM").
* **Falsification**:
  - Actual inspection of all 30 baseline assembly files showed `YMM = 0` and `ZMM = 0`.
* **Resolution**: Baseline v1 only retargeted generic 128-bit IR in `llc`. Target-aware compilation was required to achieve true YMM/ZMM utilization.
* **Confidence Recalibration**: Baseline claim marked **CONTRADICTED**. Corrected by adding dedicated `target_aware_avx2` and `target_aware_avx512` lanes in Corpus v2.

---

### ATTACK-005: Can GPU Hardware Behavior Be Claimed Without On-Device Execution?
* **Claim Attacked**: [CLAIM-014](claim-registry.jsonl) & [CLAIM-015](claim-registry.jsonl) (NVIDIA Ada and AMD Matrix Core physical behavior).
* **Skeptic Challenge**: Can we assert physical GPU behavior on a Windows host without live GPU execution?
* **Resolution**: Target IR / PTX generation is verified. Physical execution and timing remain an explicit evidence gap (`GAP-0001`, `GAP-0002`).
* **Confidence Recalibration**: Capped at `SUPPORTED`. `ARCHITECTURAL_INVARIANT` is strictly forbidden.
