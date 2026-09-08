# Bounded CPU foundation architecture audit v1

Audited implementation: `849082e6c9e8f7e40ffe90e75babc2705ee8acc1`, clean at
inspection. Canonical main was `199a841e5049130fffe5a3067c75ba1016613903`.
Scope: read-only adversarial architecture inspection, followed only by this
documentation commit. No build, test, production edit, or merge was performed
by this audit; the integration owner's Release and sanitizer runs were pending.

## Conditional verdict

**Accept the model as a coherent bounded CPU compiler foundation, conditional
on the three integration gates below.** No inspected counterexample requires a
different mathematical/value/resource/effect foundation. This is not terminal
integration acceptance, proof of a transformed whole-region optimizer, or a
claim that pending tests passed. Independent ownership/runtime findings can
still invalidate acceptance of the implementation.

The new route is bounded to Linux x64 and coherent Clang/LLVM/MLIR 21.1.8.
Existing native/provider and legacy/Windows paths remain distinct compatibility
contracts. There is no new Windows generated-execution, GPU/NPU, performance
parity, public ABI stability, zero-copy, or general device-residency claim.

## Implementation evidence

- **A compiler, not an executable proxy or interpreter.** Public source
  [Value](../../../compiler/include/matcore/region.h#L34) has no arithmetic or
  buffer representation; intrinsics have declarations but no runtime fallback.
  [Admission](../../../compiler/lib/frontend/ClosedRegionAdmission.cpp#L644)
  authenticates resolved Clang declarations, rejects indirect/hidden callee
  evaluation, and expands pure helpers at compilation. Helpers cannot introduce
  external storage effects. The frontend-neutral
  [Program](../../../compiler/lib/mlir/MatcoreClosedRegion.h#L13) carries values,
  resources, dimensions, numerics and branches without Clang objects. The
  [emitter](../../../compiler/lib/codegen/ClosedHostEmitter.cpp#L47) generates
  ordinary statements, branches and direct checked adapter calls; no runtime
  AST or graph traversal executes the source program.
- **Value meaning is separate from minimum host storage.** The source
  [Storage descriptor](../../../compiler/include/matcore/region.h#L24) records
  pointer, shape, capacity and access, not allocation ownership or noalias proof.
  [Read and candidate execution](../../../compiler/lib/runtime/closed_host_v1.cpp#L470)
  snapshot at the actual read frontier and isolate each candidate's output.
  Old values therefore survive overlapping publication while late reads see
  updated resources. Snapshots and allocation opportunities are implementation
  details, not permanent mathematical trace commitments; valid objects,
  lifetime, race freedom and conforming allocators remain caller preconditions.
- **Publication, observation and failure are defended separately.**
  [Publication](../../../compiler/lib/runtime/closed_host_v1.cpp#L602) has no
  recoverable check after its first write. Observation retires only after its
  snapshot and record allocation succeed. Sticky failure prevents subsequent
  resource access; earlier effects survive later failure. Candidate legality,
  completion and FP-environment checks precede value issuance. These are strong
  normal-return host-adapter guarantees, not crash/concurrency atomicity or a
  universal external-transfer contract.
- **Candidate and machine ownership remain explicit.** The closed
  [registry](../../../compiler/lib/runtime/closed_host_v1.cpp#L501) separates
  strict native/generated candidates from numerically legal legacy/provider
  choices. The production [build definition](../../../compiler/lib/regions/CMakeLists.txt#L154)
  links the canonical shared runtime, rather than embedding a second provider
  policy adapter. Original host compilation, sealed entry replacement and
  compiler-owned helper input isolation have distinct
  [compiler](../../../compiler/lib/codegen/ExperimentalRegionCompiler.cpp)
  and [thunk](../../../compiler/lib/codegen/AuthenticatedHostThunk.cpp)
  boundaries. Static inspection does not replace their pending hostile-link
  and installed-execution proof.

## Exact optimizer boundary

Whole-region MLIR is currently an **exact, untransformed paired witness**.
The [emitter](../../../compiler/lib/codegen/ClosedHostEmitter.cpp#L125) verifies
that witness and emits orchestration from the sealed `Program`;
[pair verification](../../../compiler/lib/mlir/MatcoreClosedRegion.cpp#L803)
rejects a changed module. Only the compiler-owned strict GEMM primitive traverses
the verified [Linalg, One-Shot Bufferize and LLVM pipeline](../../../compiler/lib/mlir/MatcoreCpuGemmCandidate.cpp#L257)
to a generated object. This is genuine generated computation, not evidence that
an arbitrary transformed region module already owns execution.

The semantic graph retains immutable tensors, ordered checks, resource epochs
and per-operation rounding boundaries. Those preserve room for legal reuse,
tiling and scheduling without redefining source values or effects; they do not
prove implemented fusion/reuse or every future HPC transformation. A future
transformed lowering needs its own legality-preserving derivation and cannot
present exact source pairing as authority for arbitrary transformed IR. No new
architecture campaign is justified by the inspected evidence alone.

## Three remaining integration gates

1. Finish exact-head Release and affected ASan/UBSan execution, including real
   weak-cleanup/helper adversaries, allocation-failure prefixes and generated
   load/store instrumentation negative controls.
2. Finish installed/source-hidden driver execution and final feature-OFF,
   provider and existing-platform regressions, including exact-head hosted
   checks. Record the actual tested SHA and outcomes, not expected commands.
3. Reconcile canonical state and the earlier driver-hold documentation with
   the accepted implementation and evidence. At this audited checkpoint those
   records still describe the earlier unresolved boundary; this audit does not
   supersede them with an unearned terminal claim.
