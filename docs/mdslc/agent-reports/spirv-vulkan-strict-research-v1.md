# MLIR SPIR-V / Vulkan strict-f32 research v1

Status: **bounded static physical feasibility observed; no product authority**.
Implementation frozen at `706e44e` on `research/mdslc-spirv-strict-f32-v1`,
based on engineering merge `f55b86e5d0fbeaa8d10d5857bc2bf2ce71168df4`.
Final independent review was interrupted by the agent service usage limit.

## What was tested

The private `compiler/experiments/spirv_strict_f32_v1` probe lowers static
rank-2 f32 GEMM M=2,K=3,N=4 through genuine upstream Linalg/parallel-loop/GPU
outlining/SPIR-V machinery on exact MLIR 21.1.8. It is not a handwritten shader,
source-connected MDSLC candidate, or LLVM SPIR-V backend qualification.
Generated fill and GEMM modules receive explicit float32 DenormPreserve,
RoundingModeRTE, SignedZeroInfNanPreserve, and arithmetic NoContraction controls.
The research script checks that removing those additions restores the generated
body and records 13 metadata/control negatives. These controls still need an
independently reviewed production lowering path before any product integration.

Physical execution selects an explicit Vulkan device, checks its capabilities,
uses private staged storage, explicit memory barriers and checked fences. It
does not silently fall back to another adapter or CPU. Unknown completion is a
research fail-stop, not the production runtime's quarantine contract.

| Actual adapter | Observed result |
|---|---|
| AMD Radeon Graphics, RADV STRIX1, Mesa 26.0.8-1ubuntu0.3 | 10 cases, 20 dispatches, 10 checked completions; 80 strict comparisons, 0 mismatches; 1280 canary and 180 immutable-input checks |
| NVIDIA RTX 4060 Laptop, driver 615.71.09 | Refused before dispatch: reported float32 denormal preservation false |
| llvmpipe LLVM 21.1.8 | Refused before dispatch for the same missing capability; no emulation claim |

Cases include rectangular arithmetic, subnormal inputs/results and negative
subnormals, signed zero, FMA and increasing-K discriminators, RTE-versus-RTZ,
infinity and NaN. Non-NaNs compare bits; NaNs compare classification, not a
promised payload. Host oracle controls execute before Vulkan discovery.
All 30 physical allocations observed were host-coherent; the noncoherent
flush/invalidate code path exists but was not physically exercised.

## Retained identities

External evidence root:
`/home/hamza-usta/mdslc-work/region-optimization-v1/builds/spirv-strict.9jpv1U7G`.
No raw logs or binaries are committed. The manifest includes commands, exact
tool identities and all generated stage identities.

| Relative evidence file | SHA256 |
|---|---|
| `artifacts-final/manifest.json` | `41da45d7965bf4494fe2d1d94fcc68916d9ed04385a10788cdafc685354adeb4` |
| `final/result-0.json` | `b65d0e25baf749de125ff34ffb6a1d489d57043c733076b7e8e98c3732dcad4d` |
| `final/result-1.json` | `ccb87420e4ac805c824147e604e3a3774538cd470fd0009732a58527340d1bb1` |
| `final/result-2.json` | `ff65f102be6a4070a820ac185bc202c744a3dd6d6a47195059057cead982211f` |
| `final/probe-build.json` | `f1279e9afefa5c091204b580b61892fba9f207a314b917edf9b2456bd53fe18a` |

Generated strict fill/GEMM SPIR-V SHA256s:
`1bfc915dc5c018f2cf6474aade818d8965ef742582ead8c1a60b4175f0811edb` /
`139dcb3930b858104b6a016530a16401b3c73df125e164cc2399f96d1721f53b`.
Run the checked-in `reproduce.py --help` and `probe.py --help` for the bounded
commands; exact executed arguments are retained in the manifest/build records.

## Boundary

The recorded `strict_contract_qualified` remains **false**. No dynamic shapes,
source authentication, production adapter/failure qualification, source
residency, general Vulkan/Intel/Metal/NPU support, or performance was proven.
Independent arithmetic/control/barrier review is the next gate for this
research branch, not automatic promotion into the compiler.
