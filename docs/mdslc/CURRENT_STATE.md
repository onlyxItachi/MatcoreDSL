# MDSLC current state

Engineering checkpoint: canonical merge
`355e281fe0de4e54c03243d332eafbbe2095591b`, [PR #78](https://github.com/onlyxItachi/MatcoreDSL/pull/78).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema admission, source-visible helpers, frozen source/header/toolchain
  -> immutable values + separate all-MAY-alias host resources
  -> dynamic shapes, per-operation numerics, ordered checks/effect frontiers
  -> exact untransformed Matcore MLIR paired witness
  -> static orchestration from the unchanged sealed Program (no interpreter)
Compiler-private immutable derived plan -> source replay + exact consumption check
Reads snapshot at their frontier; opt-in forwarding retains the checked value.
Every publication invalidates older MAY-alias bindings; branch entry/join barriers.
2–8 original source TUs -> per-file authentication -> checked host program link
Original Clang host -> sealed ABI-checked entry thunk + isolated helper LLVM
Build-issued GEMM -> Linalg/One-Shot -> CPU Transform or GPU outlining
  -> LLVM x64 baseline/AVX/AVX2/AVX512F, ARM64; NVVM sm_89 / ROCDL gfx1150
Per-operation permission -> separate AVX2/FMA realization (x64 only)
Trusted registry -> guarded candidates, private staging/output, checked completion
  -> ordered host publication, owning observation, sticky failure prefix
Private candidate DSO owns implementation; canonical Runtime owns provider policy.
Installed driver pins artifacts; coherent loading remains trusted, not sandboxed.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM and separate Linux/ARM64/Windows/Python lanes remain intact.
```

## Material change

One authenticated publication-to-read forwarding derivation is now qualified.
`--optimization publication-read-forwarding` reuses a retained immutable value
after its dominating successful publication to the same checked resource/version;
`none` stays the default, orthogonal to candidate/target selection. Required read
guards, source/frontier identity, observations and failure prefixes remain intact.
The original Program/witness and per-GEMM f32 boundaries are unchanged.

Final local qualification covered **393/393 distinct tests, no skips**, in
separate 391-test and 2-package-test runs; exact-head hosted lanes passed.
See the [forwarding contract](PUBLICATION_READ_FORWARDING_V1.md),
[independent review](agent-reports/publication-forwarding-independent-v1.md#exact-commit-forwarding-review)
and [exact merge/qualification record](agent-reports/publication-forwarding-qualified-v1.md).
Previously qualified CPU/ARM64 and staged GPU bounds remain unchanged; forwarding
adds no new mathematical operation, residency, fusion or performance claim.

## Unsupported or unproven

Syntax/API/ABI remain experimental; product semantic tooling is coherent 21.1.8.
GPU support stays opt-in Linux x64 under the [exact device/toolchain/work bounds](STAGED_GPU_CANDIDATES_V1.md),
not arbitrary NVIDIA/AMD hardware. Real HIP + global host-ASan is unqualified
and configure-refused. No Metal/NPU, whole-region transformation/fusion, automatic
reuse policy, resident/asynchronous device values, general rank-N/views, zero-copy,
generated-region Windows or performance/parity claim. Not every AVX extension,
AMX, SVE/SME, ARM provider/reassociate or cross-compiled execution is qualified.
Opaque mathematical imports and cross-region optimization remain unsupported.
Valid caller objects/lifetimes, race-free storage, conforming runtimes/allocation
and trusted loading remain preconditions; no sandbox or crash-atomic guarantee.
Manual linking and uncoordinated provider adapters are outside the driver contract.
Provider conformance is bounded. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
remains partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

## Exactly one next boundary

**Compiler-issued strict two-GEMM CPU row-panel derivation.**
First qualify an isolated upstream Transform/Linalg/One-Shot/LLVM realization of
the pure lhs-chain `C=A*B; E=C*D`, with each producer element computed once,
separate increasing f32 reductions and the intermediate f32 rounding boundary.
Bound caller-owned C scratch to `min(4,M)*N`, preserve original logical extents,
and prove no hidden tensor allocation/copy or uncontrolled call in the leaf.
This isolated proof precedes any source/runtime authority; the original Program
and paired witness must remain exact. No broader fusion or new operation follows.
