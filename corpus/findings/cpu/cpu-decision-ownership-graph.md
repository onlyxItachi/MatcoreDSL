# Comprehensive CPU Decision Ownership Graph

**Investigation Scope**: End-to-end decision ownership across the modern CPU compute stack  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. The Decision Ownership Architecture

```text
┌──────────────────────────────────────┬──────────────────────────────────┬──────────────────────────────────────────┐
│ Decision Domain                      │ Primary Legality & Semantics     │ Optimal Decision Owner & Delegated Layer │
├──────────────────────────────────────┼──────────────────────────────────┼──────────────────────────────────────────┤
│ 1. Floating-Point Reassociation      │ Matcore eDSL Contract            │ **MUST PRESERVE** (`reassoc` in MLIR)     │
│ 2. Output Buffer Non-Aliasing        │ Matcore eDSL `out(C)` Semantics  │ **MUST PRESERVE** (MLIR Destination Pass)│
│ 3. Memory Alignment (32/64-byte)     │ Caller Allocation Contract       │ **MUST EXPOSE** (Verified guard/metadata)│
│ 4. Cache Macro-Tiling ($M_C, K_C, N_C$)│ Matcore Planner / BLIS Driver   │ **MAY PLAN** (Target cache-aware tiling) │
│ 5. Data Packing (SA, SB Panels)      │ Amortized Reuse Model ($R \gg 1$)│ **MAY PLAN** (Caller workspace buffers)  │
│ 6. Register Tile Bounds ($M_R, N_R$) │ Target Register Budget ($N_{phys}$)│ **MAY PLAN / STRUCTURAL** (Caps in unroll)│
│ 7. Hardware Vector Bit Width         │ Target Triple / Subtarget Flags  │ **SHOULD DELEGATE** (LLVM TTI Interface) │
│ 8. Instruction Pipelining & Renaming │ Execution Ports & Latencies      │ **SHOULD DELEGATE** (LLVM MachineSched)  │
│ 9. Post-Op Epilogue Fusion           │ Expression Computational DAG     │ **MATCORE CORE VALUE** (Fused writeback) │
│ 10. Small-Shape / GEMV Dispatch      │ Runtime Dimension Thresholds     │ **MAY PLAN** (Dispatcher fast paths)     │
└──────────────────────────────────────┴──────────────────────────────────┴──────────────────────────────────────────┘
```

---

## 2. Ownership Boundary Invariants

1. **Semantic Invariants (Must Never Be Re-Inferred Downstream)**:
   - LLVM cannot guess that floating-point operations in a reduction loop are allowed to be reassociated. If Matcore does not explicitly attach `reassoc` or `fast-math`, LLVM is legally obligated by IEEE 754 to emit slow scalar code.
   - LLVM cannot synthesize memory packing buffers without violating standard C memory allocation semantics.
2. **Delegation Invariants (Must Never Be Re-Implemented in Matcore)**:
   - Matcore should not hand-schedule assembly instructions for every CPU revision. LLVM's Machine Scheduler and SelectionDAG perform optimal instruction selection across Haswell, Zen, Sapphire Rapids, Neoverse, and RVV automatically.
