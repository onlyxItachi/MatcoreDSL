# MDSLC current state

Engineering checkpoint: canonical merge
`0fe537b4246510c57d801577a8d915876e40a941`, [PR #72](https://github.com/onlyxItachi/MatcoreDSL/pull/72).
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
Build-issued strict GEMM -> Linalg/One-Shot -> CPU Transform or GPU outlining
  -> LLVM x64 baseline/AVX/AVX2/AVX512F, ARM64; NVVM sm_89 / ROCDL gfx1150
Permitted GEMM -> separate Transform/Linalg/Vector AVX2/FMA (x64 only)
Trusted registry -> guarded generated/native/provider; GPU explicitly forced
  -> private staging/output, checked completion/FP state, immutable value
  -> ordered host publication, owning observation, sticky failure prefix
Reads snapshot at their frontier; snapshots are realization, not value identity.
Private candidate DSO owns implementation; canonical Runtime owns provider policy.
Installed mdslc-region pins artifacts; coherent dynamic loading remains trusted.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM -> IR v1/MLIR -> existing CPU/runtime route is unchanged.
Linux x64/ARM64 regions, standalone Windows and Linux Python/JIT are separate lanes.
```

## Material change

Authenticated source now executes explicitly forced strict GEMM on the actual
RTX 4060 Laptop and Radeon 890M, through compiler-issued MLIR target kernels.
Private staging, worker isolation, guarded completion and retained failure
prefixes compose with the merged CPU/ARM paths; default dispatch is unchanged.
Final combined local **379/379**, no skips; all qualifying hosted lanes passed.
Metal strict arithmetic was falsified on the tested paravirtual device and was
not integrated. See the [campaign/evidence map](MULTITARGET_CORRECTNESS_CAMPAIGN_V1.md#completed-correctness-campaign)
and [exact final qualification](agent-reports/multitarget-composed-v1.md#qualified-canonical-checkpoint).

## Unsupported or unproven

Syntax/API/ABI remain experimental; product semantic tooling is coherent 21.1.8.
GPU support is opt-in Linux x64 with the [exact device/toolchain/work bounds](STAGED_GPU_CANDIDATES_V1.md),
not arbitrary NVIDIA/AMD hardware. Real HIP + global host-ASan is unqualified
and configure-refused. No Metal/NPU, whole-region transformation/fusion, automatic
reuse, resident/asynchronous device values, general rank-N/views, zero-copy,
generated-region Windows or performance/parity claim. Not every AVX extension,
AMX, SVE/SME, ARM provider/reassociate or cross-compiled execution is qualified.
Opaque mathematical imports and cross-region optimization remain unsupported.
Valid caller objects/lifetimes, race-free storage, conforming runtimes/allocation
and trusted loading remain preconditions; no sandbox or crash-atomic guarantee.
Manual linking and uncoordinated provider adapters are outside the driver contract.
Provider conformance is bounded. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
remains partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

## Exactly one next boundary

**One authenticated publication-to-read forwarding derivation.**
Reuse a retained immutable value after a dominating successful publication to the
same checked resource/version, with no intervening possibly aliasing write.
Retain required checks and ordered observable/failure frontiers under the existing
resource contract. This now tests one target-independent cross-operation
optimization across the validated candidate mechanisms, without broad fusion or
a residency language. [Required falsifiers and ownership](MULTITARGET_CORRECTNESS_CAMPAIGN_V1.md#ownership-and-exactly-one-next-frontier).
