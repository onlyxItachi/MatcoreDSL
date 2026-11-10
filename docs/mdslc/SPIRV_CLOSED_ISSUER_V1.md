# Closed bounded static strict GEMM SPIR-V issuer v1

This is a compiler-private **shader construction and correctness-inspection
boundary**, starting from canonical `b2597bed52de46604574f76eb33663788a5ab994`.
It is not a source backend, runtime candidate, public API or performance feature.
Existing CPU, NVVM/ROCDL, source-driver and default target choices are unchanged.
[Issue #89](https://github.com/onlyxItachi/MatcoreDSL/issues/89) is the engineering
scope; draft PR #84 research is a separate control, not imported authority.

## WHAT, HOW and MACHINE in this boundary

| Layer | Exact role here |
| --- | --- |
| WHAT | The compiler-owned rank-2 strict-f32 GEMM primitive, specialized by checked literal M/K/N. Its semantic Program and verified tensor stage retain shape, operand order, positive-zero initialization and the numerical profile. No target fact enters semantic IR. |
| HOW | Existing checked One-Shot destination bufferization, followed by upstream Linalg parallel-loop conversion and GPU mapping/outlining. M/N outputs are independent; each output retains its complete increasing scalar K reduction with separate f32 multiply and add. |
| MACHINE | Upstream MLIR GPU-to-SPIR-V lowering, production entry/interface ABI APIs and production serialization. The result is a fill shader and a GEMM shader for SPIR-V 1.3 / Vulkan >=1.2, Logical addressing and GLSL450 memory model. |

This is one bounded route for one mathematical operation, not a claim that all
GEMMs, arbitrary C++ or all targets are now covered. Static dimensions specialize
the existing compiler-owned primitive; they do not authorize source capture or
erase any source Read, extent, alias/effect or failure-frontier obligation. Existing
dynamic and zero-dimensional source mathematics is unchanged and unsupported by
this particular issuer. No new source syntax, numerical permission or Matcore IR
schema is introduced.

The shared primitive begins with positive f32 zero and, for each `(m,n)`, computes
`sum = round_f32(sum + round_f32(A[m,k] * B[k,n]))` for increasing k. Arithmetic
uses nearest-ties-even and gradual underflow, preserves signed zero/Inf/NaN
behavior, does not assume finite operands and does not specify NaN payloads.
There is no FMA, reassociation, K tiling, padding, dot/matrix instruction or
handwritten GEMM shader. K=1 becomes the exact one-step body; a singleton output
axis becomes literal index zero.

## Admitted static family

Every tuple must satisfy:

- M,K,N are positive signed integers.
- M,N <= 65535, and K < INT32_MAX.
- MK, KN and MN each <= 33,554,432 f32 elements, or 128 MiB.

Checks use division after positivity validation, so invalid/extreme inputs cannot
overflow while being rejected. The array-range bound also bounds every row-stride
multiply, linear index, byte extent and scalar-K increment within signed i32.
The M/N axes and static array ranges use guaranteed Vulkan dispatch/storage
limits, not CUDA/HIP limits, empirical timing or the historical 80 control outputs.
Three storage descriptors and local size `(1,1,1)` are within baseline limits.

These are **issuance bounds**, not a proof of allocation success or device
availability. Optional f32 float controls still require the actual device's
properties. No large boundary shape is automatically dispatched for qualification.
See the [derived envelope and evidence](agent-reports/spirv-closed-issuer-implementation-v1.md).

## Exact shader ABI and ordering

Each entry is `__matcore_strict_gemm_f32_v1_kernel`, with local size `(1,1,1)` and
dispatch `(M,N,1)`. The two separate shader modules are ordered fill, then GEMM.
The outlined host function is an inspection witness of that order, not an
executable host adapter. A real command stream needs a fill-write to GEMM-read/
write memory dependency, then checked device-to-host visibility/completion.

All descriptors use set 0, StorageBuffer, static f32 arrays, element stride 4,
member offset 0 and exactly the stated byte range:

| Role | Fill binding | GEMM binding | Bytes | Shader writes |
| --- | --- | --- | --- | --- |
| A[M,K] | none | 0 | 4MK | no |
| B[K,N] | none | 1 | 4KN | no |
| C[M,N] | 0 | 2 | 4MN | yes |

Addresses are exactly A=`m*K+k`, B=`k*N+n`, C=`m*N+n`. The fill writes positive
zero to each output. The GEMM reads and updates only that output across scalar K.
Writable metadata describes checked shader effects, not authenticated host
descriptor permissions. A/B are not presumed disjoint; a future realization must
isolate C and establish live input/output ranges independently of source Storage
aliasing. No allocation, copy, source publication or completion protocol belongs
to these shader modules.

Both entries require f32 `DenormPreserve`, `RoundingModeRTE` and
`SignedZeroInfNanPreserve`; every FMul/FAdd result has `NoContraction`.
Relaxed/fast-math/alternative rounding controls and unexpected calls/imports,
entries, memory modifiers, member decorations or effects are rejected.

## Closed issuance, complete replay, no import authority

The private `spirv_candidate::issueStrictGemmArtifactV1(context, shape)` accepts
only checked dimensions. It cannot consume caller MLIR/SPIR-V, a callback,
filesystem shader path, manifest or executable certificate. It reuses
`buildStrictGemmStagesV1(context, shape)` rather than adopting a research specimen.

An artifact binds the complete semantic/structured/bufferized/outlined/final IR,
both exact serialized word vectors, shape and typed ABI. Full replay reissues
from dimensions and compares every field, including all integer constants, CFG
edges, indexing operands, layout, bindings, geometry and float controls. Separate
structural checks prove the scalar reduction and affine addresses. Actual freshly
issued bytes also traverse native deserialization and structural verification.
None of these inspection verifiers makes supplied artifacts executable or
authenticates a source operation.

The build selection is explicitly opt-in:

```text
-DMDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON
-DMDSLC_ENABLE_EXPERIMENTAL_SPIRV_ISSUER=ON
```

The existing regions gate requires native Linux x64/AArch64, the coherent
Clang/LLVM/MLIR 21.1.8 tuple and no cross compilation. Issuer libraries, headers
and the `matcore-spirv-gemm-candidate` inspection tool are not installed or linked
into the runtime/source candidate registry. The private CLI accepts only
`--shape M,K,N --output-directory NEW_DIRECTORY`; its outputs are no-clobber
inspection data, not a shader cache or runtime import surface. `MDSLC_BUILD_SPIRV_VULKAN_PROBE`
is a separate default-OFF test-only option; automatic tests run its software
oracle, not device dispatch.

## Evidence and deliberate exclusions

Qualification distinguishes compiler/unit proof, independent well-formed graph
mutations, external SPIRV-Tools validation, software oracle controls, physical
samples and explicit device refusals. Missing tools, device properties, unsupported
shapes or skipped numerical preconditions are not passing physical evidence.
Maximum-K and 128 MiB compiler cases are not claims of large physical execution.
See the [implementation receipt](agent-reports/spirv-closed-issuer-implementation-v1.md),
[independent review](agent-reports/spirv-closed-issuer-independent-review-v1.md)
and [test-only physical harness](agent-reports/spirv-physical-harness-v1.md).
Physical observations remain bounded samples, not full-family/device qualification.

Before any future source connection, choose finite preissued catalog versus
source-time specialization explicitly. Then bind the original site/profile/
shape/Program/witness, preserve original Read/extent/effect/failure frontiers,
authenticate selected Vulkan ICD/layer/dependency identities, prove synchronous
reentry, completion/quarantine/poison and caller-state preservation, and qualify
owned artifacts plus relocated installs. None is implied by correct shader
issuance, a successful fence or matching sample outputs. There is no automatic
dispatch, residency, fusion, HPC-performance or cross-target parity claim.
