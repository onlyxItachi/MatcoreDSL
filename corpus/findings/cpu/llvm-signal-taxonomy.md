# Comprehensive LLVM Decision Signal Taxonomy

**Investigation Scope**: LLVM/Clang 20.1.8, 21.1.8, 22.1.8 Source & Analysis Headers  
**Confidence Level**: `STRONGLY_SUPPORTED` (Source-backed routing across `LoopAccessAnalysis`, `VPlan`, `TargetTransformInfo`, `SelectionDAG`, and Target Lowering)

---

## 1. Global Signal Taxonomy: Signal $\rightarrow$ Analysis $\rightarrow$ Target Interface $\rightarrow$ Consequence

```text
┌───────────────────────────┬──────────────────────────────────┬──────────────────────────────────────┬───────────────────────────────────────────┐
│ Signal Family             │ Source / IR Provenance           │ LLVM Analysis / Pass                 │ Target Transform Interface (TTI) Hook     │
├───────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
│ 1. Memory Disjointness    │ `__restrict__`, `noalias`,       │ `LoopAccessAnalysis`                 │ `TargetTransformInfo::areInlineCompatible`│
│    (Aliasing Legality)    │ `!alias.scope`, `!noalias`       │ Evaluates minimum pointer distance.  │ Removes runtime `vector.memcheck` branch. │
├───────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
│ 2. FP Reassociation       │ `FastMathFlags` (`reassoc`),     │ `LoopVectorizationLegality`          │ `TargetTransformInfo::getArithmeticCost`  │
│    (Reduction Legality)   │ `#pragma clang loop vectorize`   │ `RecurrenceDescriptor::isReductionPHI│ Authorizes horizontal SIMD tree reduction.│
├───────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
│ 3. Pointer Alignment      │ `assume_aligned(p, 32/64)`,      │ `SelectionDAG`, `AAManager`          │ `TargetTransformInfo::getMemoryOpCost`    │
│    (Memory Movement)      │ `align 32` parameter attribute   │ Verifies natural boundary alignment. │ Selects aligned load/store (`vmovaps`).   │
├───────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
│ 4. Vector Register Width  │ `-mavx2`, `-mavx512f`, `-march`, │ `LoopVectorize::selectVectorFactor`  │ `TTI::getRegisterBitWidth(RGK_RegClass)`  │
│    (Vectorization Factor) │ Subtarget Feature Flags          │ Computes optimal $VF$ for loop body. │ Returns 128 (SSE), 256 (AVX), 512 (AVX512)│
├───────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
│ 5. Register File Pressure │ Target CPU Register Class        │ `VPlanTransforms`, `CostModel`       │ `TTI::getNumberOfRegisters(ClassID)`      │
│    (Unroll Budget)        │ (16 YMM, 32 ZMM, 32 V, 32 Z)     │ Prevents unroll from exceeding budget│ Caps unroll factor to prevent spills.     │
├───────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
│ 6. Execution Port ILP     │ Target CPU Microarchitecture     │ `LoopVectorize::selectInterleaveCount│ `TTI::getInterleaveCount(VF, LoopCost)`   │
│    (Interleave Factor)    │ (Haswell 2x FMA, Zen 2x FMA)     │ Interleaves independent accumulators │ Selects $IC=2$ or $IC=4$ to saturate ports│
├───────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────┤
│ 7. Static Trip Counts     │ `SCEVConstant`, static loop      │ `ScalarEvolution`                    │ `TTI::getUnrollingPreferences`            │
│    (Loop Peeling)         │ bounds ($N=64, K=64$)            │ Proves bound divisibility by $VF*IC$.│ Eliminates scalar loop remainder blocks.  │
└───────────────────────────┴──────────────────────────────────┴──────────────────────────────────────┴───────────────────────────────────────────┘
```
