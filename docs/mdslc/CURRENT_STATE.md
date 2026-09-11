# MDSLC current state

Engineering checkpoint: canonical merge
`f69f13ac17e05da7b26f9f1d948a880b168f1265`, [PR #71](https://github.com/onlyxItachi/MatcoreDSL/pull/71).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema admission, source-visible helpers, frozen source/header/toolchain
  -> frontend-neutral immutable values + separate all-MAY-alias host resources
  -> dynamic shapes, per-operation numerics, ordered checks/effect frontiers
  -> exact untransformed Matcore MLIR paired witness
  -> static orchestration from the sealed semantic Program (no interpreter)
2–8 original source TUs -> per-file authentication -> checked host program link
Original Clang host -> sealed ABI-checked entry thunk + isolated helper LLVM
Strict GEMM primitive -> verified Linalg/One-Shot -> optional Transform M/K/N
Permitted GEMM -> separate Transform/Linalg/Vector AVX2/FMA (x64 only)
  -> LLVM (fresh output noalias, not input disjointness) -> native x64 / ARM64
Trusted registry -> generated/native strict or numerically legal legacy/provider
  -> private candidate output, checked completion/FP state, immutable value
  -> ordered host publication, owning observation, sticky failure prefix
Reads snapshot at their frontier; snapshots are realization, not value identity.
Private candidate DSO owns implementation; canonical Runtime owns provider policy.
Installed mdslc-region pins artifacts; coherent dynamic loading remains trusted.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM -> IR v1/MLIR -> existing CPU/runtime route is unchanged.
Linux x64/ARM64 regions, standalone Windows and Linux Python/JIT are separate lanes.
```

## Material change

The same authenticated source-to-generated-executable path now runs natively on
Linux ARM64: strict native/generated/automatic candidates, full FPCR/FPSR
save/normalize/restore, native ELF ownership and installed-source execution.
Hosted ARM scalar **93/93** and row-contiguous **95/95** passed with **zero skips**;
x86 Release/Debug/sanitizers/OpenBLAS on/off, Windows and legacy checks are green.
Four initial ARM fixture failures were preserved and repaired without weakening
admission or execution gates. See [exact evidence and exclusions](agent-reports/arm64-strict-integration-v1.md#qualified-canonical-checkpoint)
and the [correctness-first target campaign](MULTITARGET_CORRECTNESS_CAMPAIGN_V1.md).

## Unsupported or unproven

Experimental region execution is native Linux x86-64 / ARM64, coherent 21.1.8; syntax and
API/ABI are unfrozen. No transformed whole-region execution, fusion, automatic
reuse, general views, asynchronous/device/export effects or generated-region Windows.
No canonical GPU/NPU, general rank-N execution, zero-copy, universal performance or
BLAS-parity claim. ARM legacy/provider and reassociate candidates, SVE/SME and
cross-compiled execution remain unsupported. Opaque mathematical imports and cross-region optimization are unsupported. Valid caller
objects/lifetimes, race-free storage, conforming allocation and trusted library
loading remain preconditions; this is not a sandbox or crash-atomic transaction.
Manual archive/object linking does not inherit the complete driver contract.
Uncoordinated provider users/duplicate adapters are outside shared policy exclusion.
Provider conformance is bounded, not universal. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
remains partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

## Exactly one next boundary

**Qualify explicit strict x86 ISA candidates without changing dispatch policy.**
The native-target seam now provides a control for proving that AVX/AVX2/AVX512F
machine realizations preserve the same strict mathematics, runtime capability
guards and installed-source contract. [PR #73](https://github.com/onlyxItachi/MatcoreDSL/pull/73)
is independently reviewed work awaiting complete hosted qualification, not yet
canonical. GPU qualification proceeds independently in the linked campaign.
