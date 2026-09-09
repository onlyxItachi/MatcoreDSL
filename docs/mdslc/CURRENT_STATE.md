# MDSLC current state

Engineering checkpoint: canonical merge
`917b5ecf1d325e525bc24e3c94764d2958980a93`, [PR #63](https://github.com/onlyxItachi/MatcoreDSL/pull/63).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema admission, source-visible helpers, frozen source/header/toolchain
  -> frontend-neutral immutable values + separate all-MAY-alias host resources
  -> dynamic shapes, per-operation numerics, ordered checks/effect frontiers
  -> exact untransformed Matcore MLIR paired witness
  -> static orchestration from the sealed semantic Program (no interpreter)
Original Clang host -> sealed ABI-checked entry thunk + isolated helper LLVM
Strict GEMM primitive -> verified Linalg/One-Shot -> optional Transform M/K/N
  -> LLVM (private output-data isolation; optional column SIMD) -> generated x64
Trusted registry -> generated/native strict or numerically legal legacy/provider
  -> private candidate output, checked completion/FP state, immutable value
  -> ordered host publication, owning observation, sticky failure prefix
Reads snapshot at their frontier; snapshots are realization, not value identity.
Private candidate DSO owns implementation; canonical Runtime owns provider policy.
Installed mdslc-region pins artifacts; coherent dynamic loading remains trusted.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Runtime checks dynamic legality/capability and preserves selected effect contracts.
Legacy mutating GEMM -> IR v1/MLIR -> existing CPU/runtime route is unchanged.
Older recovered/structured/buffer/vector/two-GEMM specimens stay inspection-only.
Linux region execution, standalone Windows and Linux Python/JIT are separate lanes.
```

## Material change

Mathematical Value/Shape helpers can now live in source-visible headers, including
bounded selected template specializations. Physical file identity and selected
body provenance remain bound through admission, MLIR verification and execution;
hidden effects and escaped Value helpers still fail closed.
Pre-merge head `75c78f586801a108ec3cc49b4003b83c34a9cb26`: Release **151/151**,
affected ASan/UBSan **97/97**, independent review and **21/21 hosted checks**.
No numerical, runtime, schedule or provider policy changed.
See [library evidence](agent-reports/source-visible-math-libraries-v1.md),
[frontend contract](EXPERIMENTAL_REGION_FRONTEND_V1.md) and [user guide](REGION_COMPILER_V1.md).

## Unsupported or unproven

Experimental region execution is Linux x86-64 / coherent 21.1.8 only; syntax and
API/ABI are unfrozen. No transformed whole-region execution, fusion, automatic
reuse, general views, asynchronous/device/export effects or generated-region Windows.
No GPU/NPU, zero-copy, universal performance or BLAS-parity claim. Opaque semantic
imports and authenticated multi-source program linking are not yet supported. Valid caller
objects/lifetimes, race-free storage, conforming allocation and trusted library
loading remain preconditions; this is not a sandbox or crash-atomic transaction.
Manual archive/object linking does not inherit the complete driver contract.
Uncoordinated provider users/duplicate adapters are outside shared policy exclusion.
Provider conformance is bounded, not universal. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
remains partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

## Exactly one next boundary

**Authenticate and link a bounded multi-source program in one driver invocation.**
Header helpers now preserve their semantic source identity; independent regions
can next share an ordinary C++ host program without inventing opaque mathematical
imports. Freeze every translation unit, authenticate cross-file declarations and
global symbol ownership before linking, and retain per-region effects/failures.
This does not authorize cross-region optimization or new candidate execution.
