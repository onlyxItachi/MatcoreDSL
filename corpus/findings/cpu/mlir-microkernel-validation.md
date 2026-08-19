# MLIR & Vector Microkernel Lowering Validation

**Investigation Scope**: Validation of structured vector contraction lowering to LLVM CodeGen and verification of real Intel AMX matrix tile generation.  
**Audited Toolchain**: LLVM/Clang 21.1.8  
**Confidence Level**: `REPLICATED_OBSERVATION`

---

## 1. Structured Vector Contraction ($16 \times 4$ and $16 \times 6$) Assembly Analysis

### A. Assembly Dataflow & Register Allocation
When compiling structured vector contractions (`vector.contract` tile unrolls):
- **$16 \times 4$ Tile (8 YMM accumulators)**:
  - Inside the inner loop (`.LBB0_2`), LLVM CodeGen generates a pure compute loop:
    - 2 vector loads (`vmovups (%rdx), %ymm0` and `vmovups 32(%rdx), %ymm1`)
    - 4 broadcast loads (`vbroadcastss (%r8), %ymm2` ..)
    - 8 fused multiply-accumulates (`vfmadd231ps`)
    - **Zero inner-loop stack spills or reloads**.
  - All prologue/epilogue stack accesses are Windows x64 ABI callee-saved register saves (`%xmm6`..`%xmm10` non-volatile registers).
- **Comparison with Hand-Written OpenBLAS Assembly**:
  - OpenBLAS microkernels (`sgemm_kernel_8x4_haswell_2.c`) include explicit software prefetch instructions (`prefetcht0 512(%0)`), multi-pointer offset arithmetic, and hand-tuned tail cleanup.
  - While structured vector lowering in LLVM produces dense, zero-spill FMA loops, it does **not** automatically synthesize cache prefetching or non-unit stride transpositions without specialized passes.

---

## 2. Real Intel AMX Intrinsic Generation (`HARDEN-07` Resolved)

We compiled a real BF16 matrix multiply microkernel using AMX intrinsics (`_tile_loadd`, `_tile_dpbf16ps`, `_tile_stored`, `_tile_release`):
```nasm
tileloadd   (%rcx,%rax), %tmm1
tileloadd   (%r8,%rax), %tmm2
tdpbf16ps   %tmm2, %tmm1, %tmm0
tilestored  %tmm0, (%r9,%rax)
tilerelease
```
- **Observed**: Clang 21.1.8 successfully selected 2D tile registers `%tmm0`, `%tmm1`, `%tmm2`, emitting hardware 2D matrix multiply instructions.
- **ARM SME Scope Correction**: Because ARM SME requires specialized compiler flags (`+sme`) on AArch64 and was not physically validated on host silicon, ARM SME claims are formally narrowed to **Literature Architectural Reference** (`UNRESOLVED_PHYSICAL_VALIDATION`).

---

## 3. Claim Hardening & Recalibration

- **`HARDEN-03`**: **NARROWED & RECALIBRATED**. The sweeping claim that *"hand-crafted assembly microkernels are completely unnecessary"* is **REJECTED**.
  - **Surviving Calibrated Claim**: *"For tested dense contraction shapes and bounded tile geometries ($16 \times 4$), upstream vector contraction lowering generates spill-free, dual-issue FMA microkernels without handwritten assembly. However, specialized optimizations such as software prefetching, non-unit stride packing, and custom tail masks remain domain-specific."*
- **`HARDEN-07`**: **CONFIRMED FOR AMX** (direct assembly evidence); **DOWNGRADED FOR SME** (literature reference).
