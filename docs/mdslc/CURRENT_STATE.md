# MDSLC current state

Engineering checkpoint: canonical merge
`0b2c21f126288501fe27afea9320ec9fce392dcc`, [PR #60](https://github.com/onlyxItachi/MatcoreDSL/pull/60).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema admission and frozen source/header/toolchain authentication
  -> frontend-neutral immutable values + separate all-MAY-alias host resources
  -> dynamic shapes, per-operation numerics, ordered checks/effect frontiers
  -> exact untransformed Matcore MLIR paired witness
  -> static orchestration from the sealed semantic Program (no interpreter)
Original Clang host -> sealed ABI-checked entry thunk + isolated helper LLVM
Strict GEMM primitive -> verified Linalg/One-Shot -> optional Transform M/K/N
  -> LLVM (row schedule: output-column SIMD) -> generated x64 code
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

The first upstream-Transform schedule is source-connected and executable.
It preserves strict per-output arithmetic and existing effects; selection is
build-time opt-in, not a runtime threshold. Scalar remains the default.
Pre-merge head `58193878394c774c3365dbf746d1622e951b0813`: Release **138/138**,
affected ASan/UBSan **83/83**, independent review and **21/21 hosted checks**.
Six-shape local measurements favored this simpler schedule over the old scalar
route; explicit-vector and cache-tile alternatives remain research, not policy.
See [schedule/evidence](ROW_CONTIGUOUS_CPU_SCHEDULE_V1.md),
[user guide](REGION_COMPILER_V1.md) and [CPU foundation](CPU_FOUNDATION_CHECKPOINT_V1.md).

## Unsupported or unproven

Experimental region execution is Linux x86-64 / coherent 21.1.8 only; syntax and
API/ABI are unfrozen. No transformed whole-region execution, fusion, automatic
reuse, general views, asynchronous/device/export effects or generated-region Windows.
No GPU/NPU, zero-copy, universal performance or BLAS-parity claim. Semantic helper
libraries in headers and separate semantic modules are not yet supported. Valid caller
objects/lifetimes, race-free storage, conforming allocation and trusted library
loading remain preconditions; this is not a sandbox or crash-atomic transaction.
Manual archive/object linking does not inherit the complete driver contract.
Uncoordinated provider users/duplicate adapters are outside shared policy exclusion.
Provider conformance is bounded, not universal. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
remains partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

## Exactly one next boundary

**Preserve the private GEMM output's proven non-aliasing through LLVM lowering.**
The runtime already owns an isolated output; the current lowered data pointer
does not communicate that fact to LLVM. Preserve only this existing guarantee,
keep input/input aliasing legal, and test actual numerical/storage behavior.
This precedes more schedule complexity; it is not external-storage noalias,
new generated authority or whole-region storage reuse.
