# LLVM Decision Signal Map (Compiler Subsystem Archaeology)

**Investigation Scope**: LLVM 20.1.8, 21.1.8, 22.1.8 Source & Analysis Headers  
**Confidence Level**: `STRONGLY_SUPPORTED` (Source-backed routing across `LoopVectorize`, `VPlan`, `TargetTransformInfo`, and backend target implementations)

---

## 1. The Decision Signal Taxonomy

When LLVM's middle-end optimizer evaluates a loop nest for vectorization, unrolling, and instruction selection, it queries an explicit taxonomy of signals:

```text
┌─────────────────────────┬───────────────────────────────────┬───────────────────────────────────────────┐
│ Signal Family           │ Source / IR Provenance            │ LLVM Subsystem / Query Interface          │
├─────────────────────────┼───────────────────────────────────┼───────────────────────────────────────────┤
│ 1. Legality & Aliasing  │ `__restrict__`, `noalias`,        │ `LoopAccessAnalysis::canVectorizeMemory`  │
│                         │ `!alias.scope`, `!noalias`        │ Checks memory dependence distance.        │
├─────────────────────────┼───────────────────────────────────┼───────────────────────────────────────────┤
│ 2. Floating-Point Order │ `FastMathFlags` (`reassoc`),      │ `LoopVectorizationLegality`               │
│                         │ `#pragma clang loop vectorize`    │ `RecurrenceDescriptor::isReductionPHI`    │
├─────────────────────────┼───────────────────────────────────┼───────────────────────────────────────────┤
│ 3. Alignment Precond    │ `__builtin_assume_aligned`,       │ `TargetTransformInfo::getMemoryOpCost`    │
│                         │ `align 32` metadata on pointers   │ (Penalizes unaligned loads on x86).       │
├─────────────────────────┼───────────────────────────────────┼───────────────────────────────────────────┤
│ 4. Vector Register Width│ `-mavx2`, `-mavx512f`, `-march`   │ `TargetTransformInfo::getRegisterBitWidth`│
│                         │ Target Triple / Subtarget Features│ Returns 128, 256, or 512 bits.            │
├─────────────────────────┼───────────────────────────────────┼───────────────────────────────────────────┤
│ 5. Register File Size   │ Architectural Target Features     │ `TargetTransformInfo::getNumberOfRegisters`│
│                         │ (AVX2=16, AVX-512=32, ARM=32)     │ Constrains VPlan register pressure budget.│
├─────────────────────────┼───────────────────────────────────┼───────────────────────────────────────────┤
│ 6. Interleaving & ILP   │ Target CPU Execution Ports        │ `TargetTransformInfo::getInterleaveCount` │
│                         │ (Haswell 2x FMA ports -> IC=4)    │ Interleaves independent vector streams.   │
├─────────────────────────┼───────────────────────────────────┼───────────────────────────────────────────┤
│ 7. Trip Count / Shape   │ ScalarEvolution (SCEV) trip count,│ `LoopVectorize::selectVectorizationFactor`│
│                         │ compile-time static loop bounds   │ Masks or emits scalar loop tail fallback. │
└─────────────────────────┴───────────────────────────────────┴───────────────────────────────────────────┘
```

---

## 2. Signal Evolution Across LLVM 20 ↔ 21 ↔ 22

1. **VPlan Transition**:
   - In LLVM 20, initial VPlan-to-VPlan recipes handled vector cost selection with fallback to legacy cost models.
   - In LLVM 21/22, VPlan directly models unroll and interleave factors (`VPlanTransforms::optimizeForSizeAndThroughput`), making cost modeling more holistic across reductions.
2. **TTI Register Class Queries**:
   - `getRegisterBitWidth(RGK_RegisterClass)` remains the invariant interface queried by the vectorizer to distinguish 128-bit SSE, 256-bit AVX2, and 512-bit AVX-512 vector targets.
3. **Fast-Math Flag Granularity**:
   - `reassoc` (reassociation) is specifically isolated from `fast` (which includes unsafe reciprocal approximations), enabling precise reduction vectorization without altering NaN/Inf semantics.
