# MDSLC Architectural Value Extraction

**Mission Context**: Deriving precise MDSLC compiler design conclusions from the Compiler vs. BLAS archaeology investigation.  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. The Core Architectural Boundary: What to Lower vs. What to Delegate

The empirical decision map establishes a clear division of responsibilities across the Matcore compilation pipeline:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. WHAT MATCORE MUST PRESERVE & ENCODE AT SEMANTIC BOUNDARIES:                                   │
│    - Numerical Reassociation Contract : Authorize K-reduction reassociation in `mdsl.gemm`       │
│    - Explicit Destination Overwrite   : Map `matcore::mdsl::out(C)` to MLIR Destination Passing  │
│    - Alignment & Alias Preconditions  : Treat as contracts requiring static proof or guards      │
│    - Multidimensional Shapes & Strides: Preserve `affine_map` until `vector.contract`            │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. WHAT MATCORE DETERMINISTIC PLANNER OWNS:                                                      │
│    - Macro-Tile Cache Sizing          : Select L1/L2/L3 blocks ($M_C, K_C, N_C$) per target      │
│    - Register Tile Constraints        : Enforce $M_R \times N_R$ upper bounds in `vector.unroll` │
│    - Shape-Specialized Dispatch       : Route $M=1$ to GEMV; bypass packing for small shapes     │
│    - Workspace Allocation             : Expose caller-owned packing buffers (`sa`, `sb`)         │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. WHAT MATCORE SHOULD DELEGATE UPSTREAM (DO NOT REINVENT):                                      │
│    - Low-level Instruction Scheduling : Delegated to LLVM MachineScheduler                      │
│    - Hardware Vector Register Width   : Delegated to LLVM TTI (`-mavx2`, `-mavx512f`, `-march`)   │
│    - Register Allocation & Spilling   : Delegated to LLVM RegAlloc (kept spill-free by tile caps)│
│    - Peak Library Execution           : Delegated to authenticated CBLAS / OpenBLAS providers    │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. THE UNIQUE MATCORE ADVANTAGE OVER TRADITIONAL BLAS:                                           │
│    - Fused Epilogues                  : Fuse elementwise activations (ReLU, Bias, GELU) directly │
│                                         into the microkernel register store stage, eliminating   │
│                                         the memory round-trip mandated by standard CBLAS C ABIs. │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Specific Responsibility Classifications

| Pipeline Responsibility | Classification | Rationale from Empirical Corpus Evidence |
| :--- | :--- | :--- |
| **Reduction Reassociation Policy** | **MUST PRESERVE** | Without explicit reassociation in the eDSL contract, standard compiler vectorizers are legally barred by IEEE 754 from performing vector reduction. |
| **Destination Buffer Mutation** | **MUST PRESERVE** | Explicit `out(C)` enables One-Shot Bufferization to alias results directly to destination memory without allocating defensive copies. |
| **Register Tile Upper Bounds** | **MAY PLAN / STRUCTURAL** | Enforcing $M_R \le 16, N_R \le 6$ on AVX2 guarantees that the LLVM register allocator never encounters the register pressure cliff (avoiding stack spills). |
| **Workspace & Pre-Packing API** | **MAY PLAN** | Exposing caller-visible workspace allows pre-packing matrix $B$ once for repeated GEMM evaluations, matching BLAS efficiency. |
| **Vector Contraction Lowering** | **SHOULD DELEGATE** | Upstream MLIR `vector.contract` $\rightarrow$ `vector.fma` maps cleanly to hardware FMA instructions across x86, AVX-512, and AArch64. |
| **Instruction Selection & Scheduling** | **SHOULD DELEGATE** | LLVM CodeGen optimizes instruction pipelining and execution ports; hand-crafting assembly is unnecessary when structured vector unrolling is provided. |
| **Post-Op / Epilogue Fusion** | **MATCORE CORE VALUE** | Fusing elementwise operations into register writeback is an architectural capability that fixed CBLAS interfaces cannot provide. |
