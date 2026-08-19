# LLVM Decision Routes: Source to Machine Code

**Confidence Level**: `STRONGLY_SUPPORTED` (Source-mapped routing verified through Clang/LLVM 20–22 compiler passes and optimization remarks)

---

## 1. Route 1: Floating-Point Reduction Vectorization

```text
1. SOURCE FACT            : Standard C99 reduction loop: `sum += A[i*lda + k] * B[k*ldb + j]`
                                 │
2. LLVM IR PROPERTY       : Binary operator `%add = fadd float %mul, %sum_phi` (without `reassoc` flag)
                                 │
3. ANALYSIS               : `LoopVectorizationLegality::canVectorizeFP`
                                 │
4. LEGALITY QUERY         : `RecurrenceDescriptor::isReductionPHI` queries `FMF.allowReassoc()`
                                 │
5. DECISION               : Vectorization of inner K reduction loop REJECTED
                            Diagnostic: "cannot prove it is safe to reorder floating-point operations"
                                 │
6. TRANSFORMATION         : Reverts to scalar reduction loop; shifts vectorization to outer J-loop
                                 │
7. MACHINE CONSEQUENCE    : Emits sequential scalar `vfmadd231ss` chain or vectorizes along parallel columns
```

---

## 2. Route 2: Pointer Aliasing Check Elimination

```text
1. SOURCE FACT            : Pointers annotated with `__restrict__`: `const float * __restrict__ A`
                                 │
2. LLVM IR PROPERTY       : Function arguments have `noalias` parameter attributes
                                 │
3. ANALYSIS               : `LoopAccessAnalysis::canVectorizeMemory` evaluates pointer pairs (A, C) and (B, C)
                                 │
4. LEGALITY QUERY         : `ScalarEvolution::computeAccessIntervals` queries `AA->isNoAlias(A, C)`
                                 │
5. DECISION               : Disjoint memory spaces statically proven; runtime checks NOT required
                                 │
6. TRANSFORMATION         : `LoopVectorize` omits runtime alias check basic block (`vector.memcheck`)
                                 │
7. MACHINE CONSEQUENCE    : Eliminates branch prologue (`subq`, `cmpq`, `jae`), branching directly to vector body
```

---

## 3. Route 3: Target Vector Register Width Selection

```text
1. SOURCE FACT            : Compiler invoked with `-mavx2 -mfma` (or `-mavx512f`)
                                 │
2. LLVM IR PROPERTY       : Function attribute `target-features="+avx2,+fma"` (or `+avx512f`)
                                 │
3. ANALYSIS               : `X86TargetTransformInfo::getRegisterBitWidth(RGK_RegisterClass)`
                                 │
4. TARGET QUERY           : Queries `X86Subtarget::hasAVX2()` -> returns 256 bits (or 512 bits for AVX-512)
                                 │
5. DECISION               : Selects Vectorization Factor `VF = 8` (or `VF = 16` for AVX-512) for float32
                                 │
6. TRANSFORMATION         : Emits vector IR types `<8 x float>` (or `<16 x float>`)
                                 │
7. MACHINE CONSEQUENCE    : Backend instruction selector allocates 256-bit `YMM` (or 512-bit `ZMM`) registers
```

---

## 4. Route 4: Alignment Instruction Selection

```text
1. SOURCE FACT            : Pointer wrapped in `__builtin_assume_aligned(ptr, 32)`
                                 │
2. LLVM IR PROPERTY       : Pointer marked with `align 32` attribute or `llvm.assume` alignment intrinsic
                                 │
3. ANALYSIS               : `SelectionDAG::getLoad` queries `MemSDNode::getAlign()`
                                 │
4. TARGET QUERY           : `X86TargetLowering::LowerLOAD` checks if `Align >= 32`
                                 │
5. DECISION               : Aligned memory operation is legal
                                 │
6. TRANSFORMATION         : Selects `X86ISD::LOAD_ALIGNED` over `X86ISD::LOAD_UNALIGNED`
                                 │
7. MACHINE CONSEQUENCE    : Emits `vmovaps` instead of `vmovups`, guaranteeing zero cache-line split penalties
```
