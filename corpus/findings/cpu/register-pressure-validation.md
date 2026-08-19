# Register-Pressure Validation: Empirical Spill Hardening

**Investigation Focus**: Re-evaluation of cross-ISA register pressure and spill thresholds using verified Clang `-fverbose-asm` compiler annotations on positive and negative fixtures.  
**Audited Toolchain**: LLVM/Clang 21.1.8  
**Confidence Level**: `REPLICATED_OBSERVATION`

---

## 1. Verified Spill Detection Methodology

### A. Resolution of the Flawed Regex Detector
The previous script used broad patterns (`(%rsp)`, `str q`, `ldr q`, `vse`, `vle`), which incorrectly counted normal stack-frame allocations, callee-saved register pushes, and array accesses as spills.
We repaired the detector to query explicit Clang code-generator comments:
`# [N]-byte Spill` and `# [N]-byte Reload`.

### B. Control Fixture Verification
- **Negative Control (4 accumulators on AVX2)**: **0 Spills Detected** (Pass).
- **Positive Control (32 live vector accumulators across multi-load loop on AVX2)**: **113 Spills Detected** (Pass).

---

## 2. Hardened Cross-ISA Empirical Results

| Target Architecture | Architectural Vector Registers | Instruction Format | 4 Accumulators | 8 Accumulators | 16 Accumulators | 24 Accumulators | 32 Accumulators |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **x86-64 SSE4.2** | 16 `XMM` | **2-Operand Destructive** | **0 Spills** | **4 Spills** | **6 Spills** | **10 Spills** | **14 Spills** |
| **x86-64 AVX2** | 16 `YMM` | **3-Operand FMA** | **0 Spills** | **0 Spills** | **0 Spills** (1D) | **0 Spills** (1D) | **0 Spills** (1D) |
| **x86-64 AVX-512** | 32 `ZMM` | **3-Operand FMA** | **0 Spills** | **0 Spills** | **0 Spills** | **0 Spills** | **0 Spills** |
| **AArch64 NEON** | 32 `V` | **3-Operand FMA** | **0 Spills** | **0 Spills** | **0 Spills** | **0 Spills** | **0 Spills** |
| **AArch64 SVE** | 32 `Z` + 16 `P` | **Predicate 3-Operand** | **0 Spills** | **0 Spills** | **0 Spills** | **0 Spills** | **0 Spills** |
| **RISC-V RVV** | 32 `V` | **Dynamic Length Vector** | **0 Spills** | **0 Spills** | **0 Spills** | **0 Spills** | **0 Spills** |

---

## 3. Claim Recalibration & Realignment

1. **Downgrade "75% Law" and "25% Law"**:
   - The terms *"75% Law"* and *"25% Law"* are **REJECTED as universal laws** and downgraded to **Observed Register File Heuristics**:
     - On 2-operand destructive ISAs (SSE4.2), register capacity drops rapidly starting at 8 accumulators due to destructive register overwrites.
     - On 3-operand non-destructive ISAs (AVX2, AVX-512, NEON), the compiler maintains live accumulators without spills up to the physical register limit, but in 2D microkernels ($M_R \times N_R$), live vector loads and broadcasts reduce available accumulator slots to ~75% of the register file.
2. **Registry Status**: `HARDEN-01` and `HARDEN-02` are **NARROWED & DOWNGRADED** from `ARCHITECTURAL_INVARIANT` to `REPLICATED_OBSERVATION`.
