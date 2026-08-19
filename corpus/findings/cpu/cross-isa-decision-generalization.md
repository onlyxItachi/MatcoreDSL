# Cross-ISA Decision Generalization & Invariants

**Investigation Scope**: Invariant analysis across SSE4.2, AVX2, AVX-512, AMX, AArch64 NEON, SVE/SVE2, and RISC-V RVV  
**Confidence Level**: `ARCHITECTURAL_INVARIANT`

---

## 1. Taxonomic Invariant Breakdown

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ I. UNIVERSAL DECISION INVARIANTS (True Across ALL Architectures)                                 │
│    1. Numerical Precision & Reassociation: Strict IEEE 754 precision blocks horizontal SIMD     │
│       reductions universally. The compiler requires explicit authorization (`reassoc`) to unroll.│
│    2. Memory Non-Aliasing: Pointer overlap checks (`vector.memcheck`) are emitted across all ISAs │
│       unless disjoint memory spaces (`noalias` / `__restrict__`) are proven at compile time.     │
│    3. Register Saturation Law: Sizing accumulators beyond physical register budget forces spills.│
│    4. Packing Amortization: Memory packing is profitable if and only if panel reuse $R \gg 1$.   │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ II. FIXED-SIMD SPECIFIC DECISIONS (SSE4.2, AVX2, AVX-512, NEON)                                  │
│    1. Static Vectorization Factor: $VF = \text{VectorBitWidth} / \text{ElementBitWidth}$.         │
│    2. Tail Peeling: Schedulers emit scalar remainder loops or vector blend masks for loop tails. │
│    3. Destructive Format Penalty: 2-operand ISAs (SSE4.2) cut effective accumulator budget by 50%│
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ III. SCALABLE-VECTOR SPECIFIC DECISIONS (AArch64 SVE, RISC-V RVV)                                │
│    1. Vector-Length Agnostic (VLA) Scheduling: Schedulers do not hardcode static element counts.│
│    2. Predicate-Driven Tail Elimination: Hardware masks out excess elements on final loop step.  │
│    3. Runtime Length Configuration: Requires `vsetvli` / predicate initialization instructions.  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ IV. 2D MATRIX / TILE ACCELERATION SPECIFIC DECISIONS (Intel AMX, ARM SME)                        │
│    1. 2D Tile Abstraction: Hardware operates on 2D matrix tiles (`tmm0..7`), not 1D SIMD lanes. │
│    2. Explicit State Management: Requires OS/hardware tile configuration (`ldtilecfg`).          │
│    3. Epilogue Staging: Epilogue operations (ReLU, Bias) must be staged after matrix store.     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
