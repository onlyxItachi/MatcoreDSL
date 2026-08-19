# Confidence Recalibration & Claim Status Ledger

**Audit Context**: Evidence-Depth Realignment Pass auditing all 8 high-priority claim families from Commits `76b7246` through `cec0700`.

---

## 1. Complete Hardened Claim Status Ledger

| Claim ID | Focus Domain | Original Confidence | Final Status | Final Calibrated Confidence | Final Realignment Summary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HARDEN-01** | Register 75% Rule | `ARCHITECTURAL_INVARIANT` | **DOWNGRADED** | `REPLICATED_OBSERVATION` | Downgraded from universal law to empirical 3-operand FMA microkernel sizing heuristic. |
| **HARDEN-02** | Destructive 25% Rule | `ARCHITECTURAL_INVARIANT` | **DOWNGRADED** | `REPLICATED_OBSERVATION` | Downgraded from universal law to observed SSE4.2 spill heuristic (spills begin at 8 accumulators). |
| **HARDEN-03** | MLIR Microkernel Parity | `ARCHITECTURAL_INVARIANT` | **NARROWED** | `STRONGLY_SUPPORTED` | Rejected "completely unnecessary"; narrowed to spill-free dense FMA loop generation. |
| **HARDEN-04** | Packing Performance Numbers | `STRONGLY_SUPPORTED` | **RECALIBRATED** | `REPLICATED_OBSERVATION` | Sourced library thresholds (OpenBLAS `GEMM_P=504`, `GEMM_Q=128`); replaced global % with formula. |
| **HARDEN-05** | Fusion RAM Traffic & Speedup | `ARCHITECTURAL_INVARIANT` | **NARROWED** | `REPLICATED_OBSERVATION` | Clarified program-level pass elimination vs physical DRAM; rejected 2x speedup on compute-bound GEMM. |
| **HARDEN-06** | LLVM 20-22 Interface Stability | `STRONGLY_SUPPORTED` | **CONFIRMED** | `VERIFIED_DETERMINISTIC_EVIDENCE` | Verified exact declarations of `getRegisterBitWidth`, `getNumberOfRegisters`, `getMaxInterleaveFactor`. |
| **HARDEN-07** | AMX & SME Evidence | `STRONGLY_SUPPORTED` | **CONFIRMED (AMX) / DOWNGRADED (SME)** | `VERIFIED_DETERMINISTIC_EVIDENCE` | Verified real AMX tile instructions (`tileloadd`, `tdpbf16ps`, `tilestored`); downgraded unprobed SME. |
| **HARDEN-08** | LLVM Packing Synthesis Bounds | `ARCHITECTURAL_INVARIANT` | **NARROWED** | `STRONGLY_SUPPORTED` | Refined "cannot" to describe standard LLVM middle-end loop optimizer default pass constraints. |

---

## 2. Quantitative Summary of Audit Results
- **Total Claims Audited**: 8 high-priority claim families (spanning 116 risky claim instances)
- **Confirmed**: 2 (`HARDEN-06`, `HARDEN-07` for AMX)
- **Narrowed**: 3 (`HARDEN-03`, `HARDEN-05`, `HARDEN-08`)
- **Downgraded / Recalibrated**: 3 (`HARDEN-01`, `HARDEN-02`, `HARDEN-04`)
- **Rejected Claims**:
  1. *"75% and 25% Universal Laws"* $\rightarrow$ **REJECTED** as universal laws.
  2. *"Hand-crafted assembly microkernels are completely unnecessary"* $\rightarrow$ **REJECTED**.
  3. *"Up to 2x speedup on GEMM+ReLU"* $\rightarrow$ **REJECTED** for compute-bound matrix multiplication.
