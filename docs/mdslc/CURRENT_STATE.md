# MDSLC current state

Engineering checkpoint: canonical merge
`bb5cc1dd80f79ac2dc09259b6d85a04c14f39a34`, [PR #85](https://github.com/onlyxItachi/MatcoreDSL/pull/85).
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
Build-issued GPU pair -> serial row-panel4 kernel; no source/runtime authority
Opt-in strict CPU lhs pair -> authenticated plan -> issued row-panel4 leaf
Trusted registry -> guarded candidates, private staging/output, checked completion
  -> ordered host publication, owning observation, sticky failure prefix
Private DSO owns implementation; Runtime owns provider policy; driver pins artifacts.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM and separate Linux/ARM64/Windows/Python lanes remain intact.
```

## Material change

The closed strict two-GEMM GPU issuer/images are now canonical, bounded to
NVVM sm_89 and ROCDL gfx1150. Upstream row-panel4 derivation preserves strict
reductions and intermediate f32 rounding in one deliberately serial kernel.
The merged compiler tree equals the physically tested isolated issuer; all 23
hosted checks succeeded, with existing AVX512 skips retained. This is an
issuer/image boundary only: source/runtime still refuses GPU `strict-fused-pair`.
See the [merge qualification checkpoint](agent-reports/gpu-pair-issuer-merged-checkpoint-v1.md)
and [component evidence](agent-reports/gpu-fused-pair-issuer-v1.md).

CPU [PR #80](https://github.com/onlyxItachi/MatcoreDSL/pull/80) (`c478e49`) remains connected only through explicit Linux x64 `generated-strict` strict-pair opt-in.
Its unchanged Program/witness, both guards/full C extent and checked row-panel4 remain.
See the [contract](STRICT_FUSED_PAIR_SOURCE_V1.md), [CPU qualification checkpoint](https://github.com/onlyxItachi/MatcoreDSL/pull/87) and [issued-leaf record](agent-reports/strict-fused-pair-issuer-qualified-v1.md).
Default execution and independent [forwarding](PUBLICATION_READ_FORWARDING_V1.md) stay unchanged.

## Unsupported or unproven

Syntax/API/ABI remain experimental; product semantic tooling is coherent 21.1.8.
GPU source support stays opt-in Linux x64 under the [exact device/toolchain/work bounds](STAGED_GPU_CANDIDATES_V1.md),
not arbitrary NVIDIA/AMD hardware. Real CUDA host-ASan for the pair remains unqualified
after prelaunch cuInit status2; real HIP + global host-ASan is unqualified and configure-refused.
API mock sanitizers, device Memcheck and normal physical execution are separate evidence.
No Metal/NPU, broad fusion/whole-region witness transformation, automatic
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

**Qualify and admit the strict pair's combined GPU source/runtime law in [PR #86](https://github.com/onlyxItachi/MatcoreDSL/pull/86).**
Bound it to NVVM sm_89 and ROCDL gfx1150, with original source guards/full C
extent, checked private workspace, launch/completion/cleanup and shared
poison/quarantine semantics. The CPU guard-retirement proof alone does not
authorize fallible GPU execution. The issuer/image is merged; the combined
source/runtime connection remains draft. Its [local/package receipt](https://github.com/onlyxItachi/MatcoreDSL/pull/86#issuecomment-5983601325)
is accepted at compiler-identical `9f7f9af`; exact `954a8e6` hosted qualification
and final integration acceptance remain pending. Main has no combined GPU
pair source authority yet; this is not broad fusion or performance work.
