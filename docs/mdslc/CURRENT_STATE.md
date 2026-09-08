# MDSLC current state

Engineering checkpoint: canonical merge
`8e0cf0f39833966227dee44d4367b99927a7ae03`, [PR #58](https://github.com/onlyxItachi/MatcoreDSL/pull/58).
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
Strict GEMM primitive -> verified Linalg/One-Shot/LLVM -> generated x64 code
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

The opt-in installed compiler now compiles real named-region C++ into generated
CPU executables while preserving the original host and private implementation
ownership. Hostile helper/cleanup/linker/provider tests closed [#56](https://github.com/onlyxItachi/MatcoreDSL/issues/56).
Exact implementation head: Release **133/133**, affected ASan/UBSan **80/80**;
hosted **19/19 checks** green before normal merge. Final-head installed and
tests-disabled production checks passed; earlier source-hidden artifact evidence
retains its own chronology. Earlier hosted failures are retained.
The bounded synchronous CPU foundation is complete; optimization work remains.
See [user guide](REGION_COMPILER_V1.md) and [final evidence](CPU_FOUNDATION_CHECKPOINT_V1.md).

## Unsupported or unproven

Experimental region execution is Linux x86-64 / coherent 21.1.8 only; syntax and
API/ABI are unfrozen. No transformed whole-region execution, fusion, automatic
reuse, general views, asynchronous/device/export effects or generated-region Windows.
No GPU/NPU, zero-copy, performance advantage or BLAS-parity claim. Valid caller
objects/lifetimes, race-free storage, conforming allocation and trusted library
loading remain preconditions; this is not a sandbox or crash-atomic transaction.
Manual archive/object linking does not inherit the complete driver contract.
Uncoordinated provider users/duplicate adapters are outside shared policy exclusion.
Provider conformance is bounded, not universal. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
remains partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

## Exactly one next boundary

**One effect-preserving two-GEMM schedule with one bounded private-storage reuse.**
The sealed graph now retains the needed value, alias, numerical and effect facts.
A new authenticated derivation must preserve live old values, required checks,
per-GEMM f32 rounding and the selected adapter's stronger publication/failure
prefix contract. Exact source pairing is not transformed execution authority.
This is the next optimization boundary, not work started by this checkpoint.
