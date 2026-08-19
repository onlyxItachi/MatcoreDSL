# NVIDIA / NVVM Surface Archaeology Findings

## 1. Open NVIDIA Compiler & Kernel Evidence

This document synthesizes open/public evidence from LLVM NVPTX backends, MLIR `nvgpu`/`nvvm` dialects, and open-source template libraries (CUTLASS, CuTe) targeting NVIDIA Ada (`sm_89`) tensor cores.

> **Proprietary Implementation Rule:** *cuBLAS is referenced solely as an ABI/dispatch oracle; no proprietary binary internals are fabricated as evidence.*

---

## 2. NVIDIA Ada (`sm_89`) Lowering Contract

### Execution Flow:
```text
Global Memory (FP16 tensors)
    │
    ▼ (Vectorized cp.async global->shared, 16 bytes/copy, double/multi-buffered)
Shared Memory (128x128 tile with swizzled 8-byte row padding for bank conflict freedom)
    │
    ▼ (Warp-cooperative ldmatrix.x4 fragment loads into lane register pairs)
Registers (%r0..%r15 per thread)
    │
    ▼ (Warp-level mma.sync.m16n8k16 with FP16 inputs and FP32 accumulators)
Accumulator Registers (%f0..%f7 per thread)
    │
    ▼ (Register epilogue / bias addition)
Global Writeback (Vectorized 128-bit st.global transactions)
```

### Key Invariants:
1. **Warp Uniformity**: `mma.sync` is a warp-collective primitive requiring all 32 lanes to execute synchronously with identical matrix descriptors.
2. **Fragment Layout for `ldmatrix`**: Shared memory is not an arbitrary cache; its layout must be pre-padded to match the 8x8 matrix fragment address requirements of `ldmatrix`.
3. **Async Pipeline Overlap**: Overlapping the global-to-shared copy of tile $k+1$ (`cp.async.commit_group`) with the compute of tile $k$ hides global memory latency entirely behind compute arithmetic.
