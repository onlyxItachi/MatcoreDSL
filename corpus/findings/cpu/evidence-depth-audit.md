# Evidence-Depth Audit: Self-Correction & Realignment

**Audit Role**: Scientific Skeptic & Evidence-Depth Auditor  
**Audit Objective**: Identify all overconfident generalizations, pseudo-laws, and ungrounded performance claims across previous CPU archaeology commits.

---

## 1. Primary Methodological Vulnerabilities Discovered

### A. The Flawed Regex Spill Detector
* **Vulnerability**: In `run-register-pressure-generalization.ps1`, register spills were counted using regex patterns:
  - x86: `Spill|(%rsp)`
  - ARM: `str\s+q|ldr\s+q`
  - RISC-V: `vse|vle`
* **Defect**: Matching `(%rsp)` or `str q` counts ordinary function stack setup (callee-saved registers, function arguments, local array allocation), NOT compiler-generated register spills in the inner loop!
* **Correction**: We must use Clang's `-fverbose-asm` flag, which emits explicit `# 32-byte Spill` and `# 32-byte Reload` comments into assembly output, and validate on positive/negative control fixtures.

---

### B. Premature Elevation to "Universal Laws"
* **Vulnerability**: The terms *"75% Law for 3-Operand ISAs"* and *"25% Law for Destructive ISAs"* were formulated based on a single synthetic loop unroll style without testing multi-broadcast GEMM microkernels, pointer updates, or non-unit strides.
* **Correction**: Downgrade all such "Laws" to **Empirically Observed Register File Heuristics**.

---

### C. Sweeping Generalization on Hand-Crafted Microkernels
* **Vulnerability**: Stating that *"hand-crafted assembly microkernels are completely unnecessary"* based only on simple C loop vectorization.
* **Correction**: OpenBLAS and BLIS microkernels feature hand-tuned software prefetching (`prefetcht0 512(%rdi)`), multi-pointer updates, and complex tail masking that generic MLIR/LLVM loop lowering does not automatically synthesize without dedicated tuning passes. The claim must be narrowed to dense, cache-resident FMA loop bodies.

---

### D. Unsourced / Ungrounded Performance Metrics
* **Vulnerability**: Quoting numbers like *"packing overhead <2%"*, *"3–5x speedup"*, *"generic LLVM = 20–40% of peak"*, and *"OpenBLAS >90% of peak"* without explicit benchmark records.
* **Correction**: Ground all performance numbers in real host benchmark sweeps or explicit source citations from library constants (e.g. OpenBLAS `GEMM_P=504`, `GEMM_Q=128`, `GEMM_R=4096`).

---

### E. AMX and SME Claims Without Machine Code Evidence
* **Vulnerability**: Describing AMX tile management and ARM SME execution models based purely on successful compiler flag invocation without generating actual tile instructions (`tdpbf16ps`, `tmm0..7`, `ldtilecfg`).
* **Correction**: Author real AMX intrinsic test generating `tmm` tile registers and remove unprobed SME claims.
