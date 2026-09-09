# MDSLC current state

Engineering checkpoint: canonical merge
`c04215e787832b35bbd6a85a51d9285b053181bc`, [PR #62](https://github.com/onlyxItachi/MatcoreDSL/pull/62).
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

The issued GEMM now carries its already-proven private output-data isolation
through the exact LLVM descriptor boundary. Input/input aliasing stays legal;
no source noalias, runtime, numerical or default-schedule contract changed.
Pre-merge head `d371f8b492667b9dcebab24387c06b08314da79c`: Release **143/143**,
affected ASan/UBSan **88/88**, independent review and **21/21 hosted checks**.
Independent overlap and source-level numerical/effect oracles remain gates.
See [output-storage evidence](PRIVATE_OUTPUT_ALIAS_FACT_V1.md),
[schedule](ROW_CONTIGUOUS_CPU_SCHEDULE_V1.md) and [user guide](REGION_COMPILER_V1.md).

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

**Admit authenticated source-visible mathematical helper libraries in headers.**
The existing restricted helper grammar can be reused without opaque imports or
a second language. Preserve each semantic file's identity, selected definition,
call chain and failure frontier, then prove source/helper ownership across headers.
The reviewed implementation is still unmerged; this boundary does not promise
separate-TU semantic linking or cross-module optimization.
