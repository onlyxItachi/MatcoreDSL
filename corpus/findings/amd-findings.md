# AMD / ROCDL Surface Archaeology Findings

## 1. Open AMD Compiler & Kernel Evidence

This document synthesizes open/public evidence from the LLVM AMDGPU backend, MLIR `amdgpu`/`rocdl` dialects, and open-source template libraries (Composable Kernel, rocWMMA) targeting AMD Matrix Core architectures (CDNA2/3 `gfx90a`/`gfx942` and RDNA3 `gfx1100`).

> **Platform Evidence Status:** *Target code generation is verified via `clang -target amdgcn-amd-amdhsa` and MLIR AMDGPU tests. Hardware execution is marked `hardware-unavailable` on this Windows host.*

---

## 2. AMD Matrix Core Lowering Contract

### Execution Flow:
```text
Global Memory (HBM / GDDR)
    │
    ▼ (Vectorized global_load_dwordx4 transactions)
Local Data Share (LDS) (Double-buffered ping-pong tiles)
    │
    ▼ (ds_read_b128 into Vector General Purpose Registers / VGPRs)
Registers (VGPRs across 64 lanes in Wave64 or 32 lanes in Wave32)
    │
    ▼ (v_mfma_f32_32x32x8f16 matrix core operation)
Accumulator VGPRs
    │
    ▼ (Fused epilogue)
Global Writeback (global_store_dwordx4)
```

### Key Invariants:
1. **Wavefront Matrix Instructions**: Unlike scalar SIMD loops, `amdgpu.mfma` instructions execute across the entire wavefront (64 lanes on CDNA, 32 lanes on RDNA3).
2. **LDS Bank Alignment**: LDS memory is split into 32 or 64 banks; strides must avoid bank conflicts during 128-bit wide `ds_read_b128` operations.
3. **Register Pressure Balance**: Matrix accumulators occupy dedicated VGPRs; tile sizes must balance occupancy (waves per compute unit) against register spill risks.
