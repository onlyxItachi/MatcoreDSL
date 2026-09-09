# MDSLC current state

Engineering checkpoint: canonical merge
`bfbf0b36adc9998e948601735758b49b630c11f2`, [PR #67](https://github.com/onlyxItachi/MatcoreDSL/pull/67).
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
Linux region execution, standalone Windows and Linux Python/JIT are separate lanes.
```

## Material change

`mdslc-region --program` now compiles 2–8 real source files, including independent
regions and ordinary header-free host utilities. Source closures, foreign call
ABI/effects and protected symbol ownership are checked before linking. Both
generated numerical policies compose without changing per-region failure/effect
order, the default selector or the existing runtime/provider route.
Combined local Release **184/184** and affected ASan/UBSan **129/129** passed at
`e56e5a6`; ancestry-only reviewed head `c22ba079` then passed **21/21 hosted checks**.
See [multi-source contract](MULTI_SOURCE_PROGRAMS_V1.md),
[exact validation](agent-reports/authenticated-multi-source-v1.md) and
[completed campaign, measurements and rejected alternatives](MLIR_HPC_EXECUTION_CAMPAIGN_V1.md).

## Unsupported or unproven

Experimental region execution is Linux x86-64 / coherent 21.1.8 only; syntax and
API/ABI are unfrozen. No transformed whole-region execution, fusion, automatic
reuse, general views, asynchronous/device/export effects or generated-region Windows.
No GPU/NPU, general rank-N execution, zero-copy, universal performance or BLAS-parity
claim. Opaque mathematical imports and cross-region optimization are unsupported. Valid caller
objects/lifetimes, race-free storage, conforming allocation and trusted library
loading remain preconditions; this is not a sandbox or crash-atomic transaction.
Manual archive/object linking does not inherit the complete driver contract.
Uncoordinated provider users/duplicate adapters are outside shared policy exclusion.
Provider conformance is bounded, not universal. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
remains partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

## Exactly one next boundary

**One source-authenticated publication-to-read forwarding derivation.**
Reuse a retained immutable value only after a dominating successful publication
to the same checked resource/version, with no intervening possibly aliasing write.
Keep the authoritative Program/witness unchanged and retain read guards, ordered
frontiers and the adapter's failed-publication destination guarantee. This tests
real optimization freedom by eliminating one redundant snapshot, without fusion
or a general storage planner. [Required falsifiers and rationale](MLIR_HPC_EXECUTION_CAMPAIGN_V1.md#remaining-ownership-and-exactly-one-next-boundary).
