# MDSLC current state

Engineering checkpoint: canonical merge
`c3b7b0b9a05e11d1b8d8cd3eff1c7ef3551a8280`, [PR #66](https://github.com/onlyxItachi/MatcoreDSL/pull/66).
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
Permitted GEMM -> separate Transform/Linalg/Vector register-retained AVX2/FMA
  -> LLVM (fresh output-data noalias, not input disjointness) -> generated x64
Trusted registry -> generated/native strict or numerically legal legacy/provider
  -> private candidate output, checked completion/FP state, immutable value
  -> ordered host publication, owning observation, sticky failure prefix
Reads snapshot at their frontier; snapshots are realization, not value identity.
Private candidate DSO owns implementation; canonical Runtime owns provider policy.
Installed mdslc-region pins artifacts; coherent dynamic loading remains trusted.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM -> IR v1/MLIR -> existing CPU/runtime route is unchanged.
Older recovered/structured/buffer/vector/two-GEMM specimens stay inspection-only.
Linux region execution, standalone Windows and Linux Python/JIT are separate lanes.
```

## Material change

Forced `generated-reassociate` now reaches a separately authenticated generated
CPU implementation; it requires per-GEMM permission and AVX2/FMA hardware/OS legality.
Strict/default selection and provider policy are unchanged. Full local Release
**172/172**, affected ASan/UBSan **117/117**, then two added installed/adversarial
gates passed in both profiles; reviewed head `76bcf62183712544b202ce58290f04fef7502e48`
passed **21/21 hosted checks**. [Contract and exact evidence](GENERATED_REASSOCIATE_CPU_V1.md).
One same-permission, six-shape study measured 27.28–57.62% lower latency than our
previous generated candidate; OpenBLAS still won five. [Audited measurements](agent-reports/generated-reassociate-source-evidence-v1.md).

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
Source-visible helpers and separately legal candidates are established; independent
regions can next share an ordinary C++ program with every translation unit and
cross-file interface authenticated before linking. [PR #67](https://github.com/onlyxItachi/MatcoreDSL/pull/67)
is the unmerged implementation under combined validation, not canonical capability.
Preserve per-region effects/failures and the new candidate's numerical refusal;
this does not authorize cross-region optimization or opaque mathematical imports.
