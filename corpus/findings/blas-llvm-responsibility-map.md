# Compiler vs. BLAS Responsibility Map

**Investigation Objective**: Define the exact boundary where compiler optimization ends and high-performance BLAS design begins.  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. The Global Responsibility Matrix

| Decision Domain | Information Required | LLVM Automatic Capability | High-Performance BLAS Strategy (OpenBLAS / BLIS) | Target Dependence | MDSLC Architectural Responsibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Reassociation Legality** | Contract permitting floating-point reordering in reduction. | **Fails by Default** (strict IEEE 754 precision blocks horizontal SIMD reduction). | Hardcodes reassociation in microkernel arithmetic. | Generic | **MUST PRESERVE** (`explicit-gemm-f32-v1` policy authorizes reordering). |
| **2. Non-Aliasing Proof** | Disjointness of matrix storage buffers $A, B, C$. | Emits runtime check fallback unless `__restrict__` present. | Enforces non-aliasing preconditions via C API contract. | Generic | **MUST PRESERVE** (explicit write-only `out(C)` destination semantics). |
| **3. Memory Alignment** | 32/64-byte boundary guarantees. | Emits unaligned moves (`vmovups`) unless `assume_aligned` proven. | Aligns thread workspace buffers (`sa`, `sb`) to 4096-byte page / cache boundaries. | Architecture Specific | **MUST PRESERVE** (Propagate alignment as verified metadata or runtime guard). |
| **4. Cache Blocking ($M_C, K_C, N_C$)** | L1, L2, L3 cache sizes and TLB capacities. | **Poor / Fragile** (Generic polyhedral tiling lacks precise cache latency models). | **Explicit 5-Loop Hierarchy** ($J_C \rightarrow P_C \rightarrow I_C$) hardcoded per CPU architecture. | Highly Target Dependent | **MAY PLAN / TARGET-SPECIFIC** (Matcore deterministic planner assigns macro-tiles). |
| **5. Data Packing (SA, SB)** | Contiguous micro-panel memory layouts. | **Cannot Synthesize Automatically** (Cannot legally insert global buffer allocations). | **Explicit Packing Kernels** (`ICOPY`, `OCOPY`) transforming non-contiguous strides to contiguous panels. | Target & Layout Dependent | **MAY PLAN / SHOULD DELEGATE** (Caller-owned workspace & prepacked $B$ interfaces). |
| **6. Register Blocking ($M_R \times N_R$)** | Architectural vector register count & FMA pipelines. | Tends to select small/conservative unroll factors to avoid spills. | **Saturates Register File** ($16 \times 6$ on AVX2, $32 \times 6$ on AVX-512, $8 \times 12$ on ARM). | Target Architecture Dependent | **MAY PLAN / STRUCTURAL** (Enforce hard bounds in `vector.unroll`). |
| **7. Vector Width Selection** | Target ISA features (`+avx2`, `+avx512f`, `+sve`). | **Fully Automated** via `TTI::getRegisterBitWidth()` if target flags are passed. | Hand-selects ISA microkernel translation unit. | Target Specific | **SHOULD DELEGATE** (Pass target triple and subtarget flags to LLVM). |
| **8. Instruction Scheduling** | Pipeline latencies, execution ports, register renaming. | **Fully Automated & High Quality** (LLVM Machine Scheduler / `llvm-mca`). | Hand-scheduled inline assembly in older kernels; C intrinsics in modern kernels. | Target Microarchitecture | **SHOULD DELEGATE** (Rely entirely on LLVM CodeGen / Target Scheduling). |
| **9. Epilogue Fusion (ReLU, Bias)** | Computational DAG of adjacent elementwise operations. | Fuses elementwise point loops cleanly if in same translation unit. | **Unavailable in Standard CBLAS** (Separated by C ABI function boundary). | Generic | **MATCORE ADVANTAGE** (Fuse elementwise maps into register writeback). |
| **10. Shape Dispatch (GEMM vs GEMV)** | Runtime dimensions $M, N, K$. | Generates uniform loop nest for all shapes. | Dispatches $M=1$ or $N=1$ directly to GEMV to avoid packing overhead. | Generic / Algorithmic | **MAY PLAN** (Dispatcher selects vector kernel for rank-1 slices). |
