# MDSLC current state

Engineering checkpoint: canonical merge
`f55b86e5d0fbeaa8d10d5857bc2bf2ce71168df4`, [PR #79](https://github.com/onlyxItachi/MatcoreDSL/pull/79).
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
Isolated strict two-GEMM CPU issuer -> upstream row-panel fusion + checked workspace
Trusted registry -> guarded candidates, private staging/output, checked completion
  -> ordered host publication, owning observation, sticky failure prefix
Private candidate DSO owns implementation; canonical Runtime owns provider policy.
Installed driver pins artifacts; coherent loading remains trusted, not sandboxed.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM and separate Linux/ARM64/Windows/Python lanes remain intact.
```

## Material change

The compiler now issues a checked strict CPU leaf for `C=A*B; E=C*D`, using
upstream MLIR Transform fusion and One-Shot bufferization. Caller-owned C scratch
is bounded to `min(4,M)*N`; each increasing-order reduction and the intermediate
f32 rounding boundary remain intact. This is an isolated primitive proof, not
authenticated source fusion or a new runtime candidate.

Local qualification: **402/402 distinct tests, no skips**, in 400-test and
2-package-test runs; focused normal/ASan execution and all qualifying hosted
lanes passed. See the [issuer contract and falsifiers](agent-reports/strict-fused-pair-issuer-v1.md)
and [exact qualification](agent-reports/strict-fused-pair-issuer-qualified-v1.md).
Merged [forwarding](PUBLICATION_READ_FORWARDING_V1.md) and existing CPU/provider/GPU
execution remain unchanged. No performance claim follows.

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

**Authenticated source connection for the strict two-GEMM CPU leaf.**
Derive only adjacent pure strict lhs pairs with dominating immutable inputs and
a single-use unobserved intermediate. Preserve both original guard/source
frontiers, full logical C extent checks, earlier effects and FP state while
removing the full C allocation. Require checked private workspace and the exact
issued leaf, fail closed on incompatible candidates, and qualify installed
source execution. The CPU proof does not authorize fallible GPU/provider fusion.
