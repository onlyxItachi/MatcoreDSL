# MDSLC current state

Engineering checkpoint: canonical merge
`c478e49d56719caaa898ff517a09bc0379d31aa6`, [PR #80](https://github.com/onlyxItachi/MatcoreDSL/pull/80).
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
Opt-in forwarding retains values; publications/branches invalidate MAY-alias facts.
2–8 original source TUs -> per-file authentication -> checked host program link
Original Clang host -> sealed ABI-checked entry thunk + isolated helper LLVM
Build-issued GEMM -> Linalg/One-Shot -> CPU Transform or GPU outlining
  -> LLVM x64 baseline/AVX/AVX2/AVX512F, ARM64; NVVM sm_89 / ROCDL gfx1150
Per-operation permission -> separate AVX2/FMA realization (x64 only)
Opt-in strict CPU lhs pair -> authenticated plan -> issued row-panel4 leaf
Trusted registry -> guarded candidates, private staging/output, checked completion
  -> ordered host publication, owning observation, sticky failure prefix
Private candidate DSO owns implementation; canonical Runtime owns provider policy.
Installed driver pins artifacts; coherent loading remains trusted, not sandboxed.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM and separate Linux/ARM64/Windows/Python lanes remain intact.
```

## Material change

Authenticated source now connects strict `C=A*B; E=C*D` to the issued CPU leaf
through explicit `--optimization strict-fused-pair --candidate generated-strict`
on Linux x64. Only adjacent pure lhs pairs with dominating immutable Read inputs
and a private single-use intermediate qualify. The original Program/witness,
both source guard frontiers, full logical C extent, prior effects and FP state
remain; checked private C scratch is `min(4,M)*N`, not a full C allocation.
Increasing reductions and intermediate f32 rounding are preserved. Default
execution and independent [forwarding](PUBLICATION_READ_FORWARDING_V1.md) are unchanged.

Exact premerge head `9a4dfeff70db57f3522e66fe9e13e8602c5a167c`; local qualification
at compiler-identical frozen `6e25df7` (only the final Debug timeout changed):
**406/406 distinct tests, zero skips**, in disjoint 404-test and 2-package runs.
All **22 hosted checks passed**; hosted Debug had **243 passed and 14 existing
AVX512 capability skips**, not a zero-skip result. See the
[source contract](STRICT_FUSED_PAIR_SOURCE_V1.md),
[implementation/review](agent-reports/strict-fused-pair-source-v1.md) and
[issued-leaf qualification](agent-reports/strict-fused-pair-issuer-qualified-v1.md).
No GPU fusion or performance inference follows from this CPU connection.

## Unsupported or unproven

Syntax/API/ABI remain experimental; product semantic tooling is coherent 21.1.8.
GPU support stays opt-in Linux x64 under the [exact device/toolchain/work bounds](STAGED_GPU_CANDIDATES_V1.md),
not arbitrary NVIDIA/AMD hardware. Real HIP + global host-ASan is unqualified
and configure-refused. No Metal/NPU, broad fusion/whole-region witness transformation, automatic
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

**Separately qualify the strict pair's combined GPU source/runtime law.**
Bound it to NVVM sm_89 and ROCDL gfx1150, with original source guards/full C
extent, checked private workspace, launch/completion/cleanup and shared
poison/quarantine semantics. The CPU guard-retirement proof alone does not
authorize fallible GPU execution. Draft [#85](https://github.com/onlyxItachi/MatcoreDSL/pull/85)
owns the isolated issuer/image; draft [#86](https://github.com/onlyxItachi/MatcoreDSL/pull/86)
owns the combined source/runtime connection. Neither is canonical GPU pair
authority yet; this next boundary is not broad fusion or performance work.
