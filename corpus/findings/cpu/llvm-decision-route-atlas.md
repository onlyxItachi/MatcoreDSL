# LLVM Decision Route Atlas: Source Signals to Machine Assembly

**Investigation Scope**: End-to-end decision routing across Clang/LLVM 21.1.8  
**Confidence Level**: `STRONGLY_SUPPORTED` (Direct pass-level provenance and optimization remark verification)

---

## 1. Decision Route 1: Loop Vectorization & Reassociation Decision Route

```text
Step 1: SOURCE SIGNAL
        `#pragma clang loop vectorize(enable)` OR `fast-math` flag on reduction loop
        │
Step 2: CLANG FRONTEND METADATA
        Emits loop metadata `!{!"llvm.loop.vectorize.enable", i1 1}`
        │
Step 3: MIDDLE-END PASS ENTRY
        Pass `LoopVectorizePass` is invoked on Loop Nest
        │
Step 4: LEGALITY ANALYSIS
        `LoopVectorizationLegality::canVectorize(Range)`:
        - `LoopAccessAnalysis::canVectorizeMemory` -> checks memory dependencies
        - `RecurrenceDescriptor::isReductionPHI` -> detects accumulator reduction PHI node
        - Queries `FMF.allowReassoc()` -> returns TRUE due to pragma / fast-math flag
        │
Step 5: TARGET TRANSFORM COST EVALUATION
        `LoopVectorizationPlanner::selectVectorizationFactor()`:
        - Calls `TTI->getRegisterBitWidth(TargetTransformInfo::RGK_RegisterClass)` -> returns 256 bits (AVX2)
        - Computes $VF = 256 / 32 = 8$ (for 32-bit floats)
        - Calls `TTI->getInterleaveCount(VF=8, LoopCost)` -> returns $IC=2$ or $4$
        │
Step 6: VPLAN TRANSFORMATION & CODEGEN
        `VPlanTransforms::optimizeForSizeAndThroughput()` emits vectorized loop body:
        `%vec.phi = phi <8 x float> [ zeroinitializer, %entry ], [ %vec.next, %vector.body ]`
        `%vec.fma = call <8 x float> @llvm.fma.v8f32(...)`
        │
Step 7: BACKEND CODEGEN & INSTRUCTION SELECTION
        `X86DAGToDAGISel` maps `@llvm.fma.v8f32` to hardware instruction:
        `vfmadd231ps %ymm1, %ymm2, %ymm4`
        Machine scheduler maps instruction to Execution Ports 0 & 1 on Intel Haswell.
```

---

## 2. Decision Route 2: Memory Alignment & Instruction Selection Route

```text
Step 1: SOURCE SIGNAL
        `const float *ptr = (const float *)__builtin_assume_aligned(A, 32);`
        │
Step 2: LLVM IR ATTRIBUTE
        Emits pointer load with explicit alignment: `load <8 x float>, ptr %ptr, align 32`
        │
Step 3: SELECTION DAG LOWERING
        `SelectionDAGBuilder::visitLoad` creates `SDNode` of type `ISD::LOAD` with `MemSDNode::getAlign() == 32`.
        │
Step 4: TARGET LOWERING SELECTION
        `X86TargetLowering::LowerLOAD` queries target register size (32 bytes for AVX):
        Checks `Alignment >= 32` -> TRUE.
        │
Step 5: MACHINE INSTRUCTION EMISSION
        Emits `X86::VMOVAPSrm` (`vmovaps (%rdi), %ymm0`) instead of `X86::VMOVUPSrm` (`vmovups`).
        Consequence: CPU loads vector across natural 32-byte cache boundaries with zero split-load penalty.
```
