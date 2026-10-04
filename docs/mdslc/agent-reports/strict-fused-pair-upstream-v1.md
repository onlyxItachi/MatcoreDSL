# Strict two-GEMM row-panel fusion: bounded upstream feasibility

Status: **research prototype, physically executed, no source execution authority**.
Date: 2026-10-04. Starting tree: `43ce3cb2557aad4dc3b810ea715c00531c76628f`.
Branch: `research/mdslc-strict-fused-pair-v1`. The original whole-region paired
witness, its verifier, `MatcoreCpuGemmCandidate`, driver and Runtime are unchanged.
These fixtures are not part of the production CMake/CTest graph.

## Result and exact scope

Upstream MLIR 21.1.8 can realize `C=A*B; E=C*D` with a serial four-row producer
panel rather than a full `M*N` intermediate, without custom compute loops,
algebraic reassociation, fast-math or FMA. Each producer element is computed
once. Both reductions remain increasing scalar order, and the first result is
stored and loaded as `f32` before the second GEMM consumes it.

This establishes a Linux x64 research leaf and upstream scheduling/allocation
seam, **not** an authenticated source transformation, supported candidate,
failure-frontier implementation, automatic dispatch or speedup. The fixture's
`research.*` attributes identify Transform handles only; they are neither
legality proof nor execution authority. The research bufferizer is deliberately
not a secure importer or a replacement for a closed compiler issuer.

The admitted specimen is only an lhs chain: `A[M,K]`, `B[K,N]`, `D[N,P]`, private
single-use `C[M,N]`, private `E[M,P]`. It contains no extra epilogue, publication,
observation, host/control operation or shape-dependent source operation. A
future source selector must additionally prove that all three input reads occur
before the adjacent strict pair. It must reject rhs carry, `C*C`, any surviving
publication or other computation use of C, and even C shape-query/branch uses in
the initial implementation: existing orchestration needs Value handles for
these, so treating them as free metadata would be a separate change.

## Coherent toolchain and reproduction

Observed `mlir-opt`, `mlir-translate`, Clang and LLVM development headers are
21.1.8; Clang is Ubuntu `1:21.1.8-6ubuntu1`. Selected MLIR prefix:
`/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`.
Compiler: `/usr/bin/clang++-21`; LLVM development libraries: `/usr/lib/llvm-21`.
Physical CPU: AMD Ryzen AI 9 HX 370, Linux x86-64. The generated object used
Clang's x86-64 baseline, not native/AVX/FMA target flags.

From this worktree:

```sh
bash compiler/tests/research/strict_fused_pair_v1/run.sh
```

The script builds only one tiny C++ tool at a time, dynamically linking the
existing `libMLIR.so.21.1`; it does not build/install LLVM or compile MDSLC. It
then lowers the fixture, compiles one generated object, normally links a tiny
independent host oracle and executes it. Generated artifacts remain in ignored
`build-research-fused-pair/`. No timings were taken.

Committed research inputs:

- [pair.mlir](../../../compiler/tests/research/strict_fused_pair_v1/pair.mlir):
  tensor pure chain, positive-zero fills, pinned legal and negative schedules.
- [workspace_bufferize.cpp](../../../compiler/tests/research/strict_fused_pair_v1/workspace_bufferize.cpp):
  One-Shot callback mapping the certified panel request to caller workspace.
- [execute.cpp](../../../compiler/tests/research/strict_fused_pair_v1/execute.cpp):
  independent increasing-order volatile-f32 oracle and bounded physical tests.
- [run.sh](../../../compiler/tests/research/strict_fused_pair_v1/run.sh):
  complete standalone reproduction and negative controls.

## Upstream derivation and physical intermediate

The pinned Transform recipe tiles only consumer M with
`transform.structured.tile_using_for [4,0,0]`, fuses the producer and its
positive-zero fill into that `scf.for`, and applies the standard
`transform.apply_patterns.tensor.fold_tensor_empty` pattern. No reduction or
consumer P tiling, interchange, vector contraction or padding is requested.
The [21.1.8 Transform operations](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td)
provide the tiling/fusion mechanism; the
[21.1.8 empty-tensor pattern](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/mlir/lib/Dialect/Tensor/Transforms/EmptyOpPatterns.cpp)
replaces a slice of an unspecified empty tensor with a smaller empty tensor.

The research One-Shot `allocationFn` maps precisely one dynamic identity-layout
rank-2 f32 request into argument 4, the caller's private row panel. It accepts
only the direct function-body serial row loop, lower zero, step four,
`min(M-i,4)`, and N originating from B's second dimension (through One-Shot's
own tensor/buffer casts). Other types, layouts, shapes, nested loops or repeated
allocation requests fail. The extra argument is not a no-alias claim about
source Storage. The harness itself owns disjoint output/workspace, and inputs
may alias. Actual caller capacity, lifetime, offset/strides, representability,
FP controls and disjointness remain preconditions of this unchecked leaf.

The resulting buffer IR contains the following physical structure; this is an
excerpt, not an imported authority certificate:

```mlir
scf.for %i = %zero to %M step %four {
  %rows = affine.min #map(%i)[%M] // min(M-i,4)
  // upstream subviews select A's rows and all of B.
  %panel = memref.reinterpret_cast %workspace to
    offset: [0], sizes: [%rows, %N], strides: [%N, 1]
    : memref<?x?xf32> to memref<?x?xf32>
  linalg.fill ins(%positive_zero : f32) outs(%panel : memref<?x?xf32>)
  linalg.matmul ins(%A_rows, %B) outs(%panel)
  linalg.matmul ins(%panel, %D) outs(%E_rows)
}
```

Canonicalize, CSE, canonicalize and drop-equivalent-buffer-results remove the
redundant self-copy and returned output descriptor. Standard Linalg-to-loops,
strided metadata expansion, Affine/SCF/CF/Arith/MemRef/Func-to-LLVM conversion
and LLVM translation provide all compute/machine lowering. Expansion precedes
`lower-affine` because it introduces new affine operations.

Row ranges `[i,min(i+4,M))` partition M. The producer computes every column N
for each row exactly once, with complete increasing K; the consumer then
computes every column P with complete increasing N. Panel elements are f32
stores before consumer f32 loads. Peak **C scratch** capacity is
`min(4,M)*N` floats, compared with `M*N`; this is not overall peak RSS or total
memory (inputs, E, host oracle and bookkeeping are excluded).

Pre-optimization LLVM has two `fmul float` and two `fadd float` instruction
sites, no fast-math/noalias, no storage allocation/deallocation/copy operation
and only `llvm.smin.i64` plus the C-interface-to-leaf call. The optimized object
contains separate `mulss`/`addss`, no FMA, and **does have undefined `memset`**:
LLVM O2 lowered zero fills to conforming libc calls. This is not a call-free
machine-code claim. A future trusted candidate must explicitly account for
conforming `memset`/FP/write-range behavior or select and verify a loop-idiom
strategy; absence of allocator calls in pre-optimization LLVM is insufficient
to authenticate arbitrary backend/library behavior.

## Executed evidence and falsifiers

Final standalone run: **6,786 checks, 0 failures**. This is physical local CPU
execution, not a hosted test or generated sanitizer qualification.

- 3,360 combinations: `M=0..9`, `K=0..6`, `N=0..7`, `P=0..5`; every E element
  compared bitwise with an independently compiled strict f32 two-stage oracle.
  NaN payloads are not compared; both sides must be NaN. Other values, including
  signed zero, require identical bits. Null empty operand/output/workspace data
  and row tails are exercised.
- Output and exact `min(4,M)*N` panel canaries are checked for every geometry.
  Identical and partially overlapping A/B/D inputs execute; no input-input
  no-alias assumption is made.
- First and second increasing-reduction order discriminators execute.
  The FMA discriminator uses `-1 + (1+2^-23)*(1-2^-23)`: strict returns positive
  zero, while a fused multiply-add differs.
- The intermediate-rounding discriminator has `A=[1+e]`, `B=[1-e,1]`,
  `D=[1,-1]^T`, `e=2^-23`. The result is `-e`; retaining unrounded first products
  until final f32 conversion produces `-e-e*e` instead. This is tested, not
  inferred from an ordinary small integer GEMM.
- Cross-GEMM reassociation is falsified by scalar
  `A=B=FLT_MAX`, `D=0`: `(A*B)*D` is NaN whereas `A*(B*D)` is positive zero.
  NaN/Inf, signed zero and positive/negative subnormals exercise both stages.

Negative results are part of the conclusion:

1. **Tiling/fusing without empty-tensor folding retains full M*N storage.**
   `__no_empty_fold` leaves a full empty tensor outside the row loop. Default
   bufferization allocates that full intermediate, and the bounded workspace
   callback rejects it with `rejected request shape/origin`.
2. **Default One-Shot allocation is unsuitable for the existing trusted leaf.**
   Even with smaller panels it emits `memref.alloc/dealloc` inside each row
   iteration; actual LLVM translation produced unchecked malloc/free. The
   research leaf does not use this result. The caller workspace callback fixes
   that particular physical-allocation problem, not runtime guard authority.
3. **Consumer-column tiling can recompute the producer.**
   `__column_recompute` tiles consumer `[4,2,0]` and fuses into its inner column
   loop. Generated IR recomputes the same complete C row panel once per P tile.
   For `M=8,N=11,P=7`, 88 distinct C elements are computed four times. The
   workspace mapper rejects the nested-loop/origin mismatch. Merely matching
   two matmuls after a fusion pass is not a compute-once proof.
4. **Pipeline ordering matters.** Expanding strided metadata after lowering
   Affine leaves `affine.apply` and translation fails. Running equivalent-result
   dropping before canonicalizing the loop result leaves an output return;
   ownership deallocation then clones the full E output. The fixed recipe
   avoids both, and CSE is necessary to eliminate identical-subview self-copy.
5. **Embedded API setup is not the mlir-opt process setup.** The tiny tool must
   register SCF/Linalg/Tensor/Func/Arith bufferization external models and load
   Bufferization/MemRef dialects. An early incomplete setup aborted on creating
   an unloaded `bufferization.alloc_tensor`; it was not counted as evidence.
   Dynamic types use `ShapedType::kDynamic`, not literal `-1` in LLVM 21.

No deliberate invalid descriptor was executed. The leaf does not guard invalid
sizes/capacities or restore caller FP state; production Runtime must do so.
No ASan claim follows from linking or from canaries. No GPU/ARM/Windows pair,
whole MDSLC build/test result or source-issued pair is claimed by this branch.

## Saved generated evidence

Final ignored artifact hashes (reproducible script outputs):

| Artifact under `build-research-fused-pair/` | SHA-256 |
| --- | --- |
| `scheduled.mlir` | `f337883b3417ec4c3808dfce444645b66f97560b31a274f9d848eb0a97c90542` |
| `workspace.mlir` | `37b8bf78305a5580e55a085b2ce8213c5299ab18680a7359a2ebd59675736dd9` |
| `workspace.ll` | `56f108e0bb4ecefb74c854b9d02968d5281a0467d246fa62f69549082275d0e0` |
| `workspace.o` | `02067bbea0241541eef823de874eab20dd3654ebd9127185db4c15a360461eac` |

The report records observed IR structure and hashes while generated files stay
out of Git as required by AGENTS. The input fixture intentionally contains the
complete reviewed recipes.

## Future source-issued pair integration gates, not implementation

1. Select only the admitted adjacent lhs pure chain from the authenticated
   sealed Program; count all uses, retain first/second source sites, guards,
   numerical contracts and failure/completion frontiers. The original exact
   whole-region witness stays unchanged. A separate opaque in-process derivation
   must prove this selected candidate; serialized attrs never issue authority.
2. Keep forced CPU pair selection separate from existing/default dispatch.
   Reject non-strict, unsupported target/ISA/provider or any ambiguous source
   window without fallback. Neither ordinary read descriptor inequality nor
   forwarding creates physical input-input no-alias.
3. Retain full logical C extent/byte/index checks **before** consuming the
   smaller physical panel. Required counterexample:
   `A[INT64_MAX,0], B[0,2], D[2,0]`. Reads and final E are empty, panel is tiny,
   but logical `C[INT64_MAX,2]` must fail extent-overflow at the first GEMM.
   Do not submit this unchecked descriptor tuple to the research leaf.
4. Runtime owns checked private E and row-panel workspace; the issued leaf
   owns no allocation/free. Allocation opportunities may change, but required
   source shape/candidate checks must not disappear. Keep source-required
   empty/zero-reduction checks even if physical math is skipped. An explicit
   derivation must decide where first/second candidate completion and FP-state
   checks retire; pure adjacency alone does not settle `completed_frontier` or
   failing source attribution. Do not mark the first GEMM complete merely
   because the second guard passed.
5. Test second-shape/candidate/alloc failures, first-result overflow, sticky
   prefix/site diagnostics, caller FP controls/status, reentry/ownership,
   artifact substitutions, original-witness mutation, publication/observation,
   shape-query and multi-use rejection. Reuse existing driver DSO ownership,
   frozen host/helper isolation and checked thunk contracts rather than manually
   linking a production private archive.
6. Implement the allocation callback inside the closed built-in issuer only
   after exact structured/scheduled/buffer/LLVM verifier gates; prove scratch
   pointer provenance and reject every unexpected request/operation/call.
   Independently validate actual generated ASan instrumentation with a negative
   control, then ordinary source compile/link/execute and affected/full suites.

## Predeclared phase-2 diagnostic protocol

Only after source authority and runtime gates, compare fused versus unfused
**same-source** chains `A[M,K], B[K,N], D[N,P]`, same strict candidate/toolchain/
FP context. Compare the full output bitwise with the strict oracle. Tiny/tails/
zero cases remain correctness cases. Representative diagnostics are
`(M,K,N,P)=(8,13,11,7),(65,63,97,31),(257,64,96,73),(3,128,256,64)`;
fixed warmup 2, repetitions 10. Report every result and regression with
allocation-inclusive source execution time. No minimum speedup is an acceptance
gate, and these measurements alone cannot claim BLAS parity or generally
optimal performance. C scratch upper bound is `min(4,M)*N` versus `M*N`; total
peak memory needs a separate measurement.
