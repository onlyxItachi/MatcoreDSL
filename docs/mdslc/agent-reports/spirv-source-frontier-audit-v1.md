# SPIR-V/Vulkan: source-frontier mapping v1

Status: **read-only design audit; no new candidate or execution authority**.
Source architecture is pinned to CPU merge `c478e49d56719caaa898ff517a09bc0379d31aa6`;
research is [PR #84](https://github.com/onlyxItachi/MatcoreDSL/pull/84), exact head
`5dc14db686b6b2668a2d3a47f5ef24c0e2bb4b97`, code `706e44e85ecc9f6bbb77fb520d4859d701be7ab3`.
This audit ran no compiler, SDK, prototype, device inventory or physical workload.
It does not change the ongoing NVVM/ROCDL pair qualification frontier.

## Existing observations, not transferable authority

The [research qualification record](https://github.com/onlyxItachi/MatcoreDSL/blob/5dc14db686b6b2668a2d3a47f5ef24c0e2bb4b97/docs/mdslc/agent-reports/spirv-vulkan-strict-repro-v1.md)
records upstream MLIR 21.1.8 lowering of **one static rank-2 GEMM**, M=2,K=3,N=4.
It starts from an op-level Linalg specimen, not sealed Matcore source. Two shader
dispatches implement +0 fill then GEMM; this is not fused-pair evidence.
AMD RADV STRIX1/Mesa 26.0.8-1ubuntu0.3 matched 80 outputs in ten cases,
with 20 dispatches/ten checked fences, 1280 canaries and 180 input checks.
All 30 allocations were coherent; no physical noncoherent-memory claim follows.
NVIDIA and llvmpipe refused missing f32 denormal preservation before dispatch.
These historical observations are not a refreshed target-availability claim.
The oracle was frozen before Vulkan calls; NaNs compare classification, other
outputs compare bits. Every observation retains `strict_contract_qualified:false`.

The generated GEMM uses increasing scalar K, separate `OpFMul`/`OpFAdd`, f32
storage, arithmetic `NoContraction`, and entry-point float32 `DenormPreserve`,
`RoundingModeRTE`, `SignedZeroInfNanPreserve`. All three actual device properties
are required. Correct rounding is instruction-specific: the Vulkan
[precision rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision)
cover scalar add/multiply; the controls do not justify arbitrary dot, matrix,
extended-instruction or fast-math replacements. Capabilities plus sampled
outputs alone are not a general arithmetic proof.

## Control specimen versus smallest reusable contract (proposal)

Fixed 2x3x4 is the smallest trusted **control specimen**, not a useful general
source backend. A forced fixed candidate could guard actual Value shapes at the
original GEMM frontier, but must refuse every other shape, including zero cases;
it must not reinterpret those valid source shapes as language errors.

The proposed smallest reusable engineering boundary is a **closed parameterized issuer
for a bounded, fully-static, positive rank-2 shape family**, followed by separately
qualified adapter/source binding. Its internal input is a checked `(M,K,N)` tuple,
not user MLIR/SPIR-V or an executable certificate. A source-derived record binds
that tuple, original GEMM site/profile and unchanged Program/witness. The family
envelope must be predeclared from index/byte/dispatch limits and new qualification,
not borrowed from CUDA/HIP bounds or inferred from the 80 existing outputs.
Initially zero dimensions remain incompatible with this realization until
separately qualified; strict source semantics and their guards remain valid.
This is a usefulness-oriented generalization, not a proven minimal requirement:
fixed 2x3x4 remains the smallest proposed source connection to the existing proof.

Static specialization needs **no semantic rewrite**. In the pinned tree,
`MatcoreClosedRegion.h` already represents Literal, ShapeParameter, ValueRows and
ValueColumns; `MatcoreClosedRegion.cpp::dimension/verifyBody/dimensionExtent`
propagate dominating literal Read shapes and GEMM result shapes into exact ranked
tensor types. ShapeParameter stays dynamic. `ClosedRegionAdmission.cpp::bound`
admits exact shape literals through Sema, not arbitrary host expression folding.
A narrow sealed-Program shape query can reuse that analysis; it must respect
branch-local scope, retain live dimension uses and reject unknown shapes rather
than infer constants from runtime descriptors, sample values or serialized types.
Constant shape does not discharge Read capacity/access/extent checks.

## Missing production gates

| Boundary | Required before authority; present research limitation |
|---|---|
| Closed issuance | Reuse `buildStrictGemmStagesV1` and checked static specialization; pin full scheduled/final shader graph, +0 seed, increasing reduction, addresses and writable range. `probe.cpp::readControlled` authenticates neither every integer constant nor AccessChain/CFG operand. |
| Shader/host ABI | Bind exact entry, Logical/GLSL450 model, local 1x1x1 and dispatch MxNx1, storage-buffer set/bindings, static array extents/stride 4/member offset 0, exact descriptor ranges/layout and fill-to-GEMM binding. No unexpected imports, instructions or relaxed attributes. The current control uses A/B/C bindings 0/1/2, ranges 24/48/32 bytes; fill binding 0 is C. |
| Target/FP | Actual loader/device Vulkan >=1.2, all three f32 properties, authenticated selected device/implementation and relevant limits; no identity-only inference, ordinal-dependent selection or target fallback. Check controls on the final embedded binary, not only pre-serialization MLIR. |
| Source/host effects | Existing sealed Program, original GEMM frontier and immutable snapshots stay intact; external Storage remains MAY-alias. Preserve earlier publication/observation, sticky failure, old result on failure, synchronous reentry and complete caller FP/errno state. Research process fail-stop and unchanged FP controls do not establish a reusable joined-worker contract. |
| Artifact/install | Embed only closed-issued shaders in owned private artifacts, recheck source/artifact/dependency identities and symbol ownership. No runtime shader path, mutable manifest or manual linking confers authority. Relocated installed-source execution needs its own qualification, with no consumer SPIR-V tools or development-prefix dependency. |

Two artifact routes compete. A **finite preissued static catalog** naturally fits
the existing installed candidate DSO and per-site shape selection. Arbitrary
source-time specialization avoids catalog growth but needs a new trusted
issuance/embedding/driver-artifact lifecycle; it is not implied by today's DSO
snapshot checks. Choose explicitly before connecting the shape family.

MLIR provides a plausible reviewed-control seam: pinned
[TargetAndABI.h](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/SPIRV/IR/TargetAndABI.h)
exposes EntryPointABIAttr/interface ABI utilities, and
[Serializer.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Target/SPIRV/Serialization/Serializer.cpp)
supports NoContraction decoration attributes. Replacing the research test-only ABI
pass and post-assembly metadata patch with these production APIs is a **proposed
derivation**, not an executed or accepted pipeline. The
[LLVM SPIR-V backend](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/docs/SPIRVUsage.rst)
and external SPIRV-LLVM-Translator are distinct paths; neither inherits this
MLIR SPIR-V/Vulkan proof merely by producing a `.spv` file.

## Completion and implementation identity are concrete blockers

The research submits both dispatches in one command buffer, uses host/compute,
fill/GEMM and compute/host barriers, then waits on a fence. A
[timeout is not successful completion](https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitForFences.html).
A reusable adapter needs phase-aware partial-acquisition cleanup and conservative
quarantine/poison on unresolved submit/completion, retaining all possibly live
instance/device/queue/command/fence/descriptor/pipeline/buffer/mapped storage.
No output or later source operation may retire from uncertain completion.
Void destroy/unmap calls do not return fallible-cleanup status; safe quiescence
and host synchronization must be established before calling them. Begin with
coherent host-visible memory only; the unexercised maintenance path is not a pass.

Pinning `libvulkan` alone cannot authenticate the execution implementation:
the [loader architecture](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md)
includes discovered ICDs and intercepting layers. A bounded loading law must bind
the actual selected ICD/layer/dependency chain, reject unsupported overrides and
protect worker/querying symbols under the existing driver ownership contract.
Do not change process-global Vulkan selection environment to manufacture support.
Vendor pipeline compilation remains trusted runtime JIT, not build-time machine
image qualification or a sandbox guarantee.

## Predeclared future falsifiers (not run by this audit)

- Mutate only K-loop bound or AccessChain operands while preserving controls/op
  counts: closed graph issuance must reject. Remove fill/barrier or misbind C,
  change stride/range/local size, add an entry/import: reject before dispatch.
- Family shapes need rectangular/skinny and boundary/tail cases, independent
  increasing-K/FMA/RTE/subnormal/Inf/NaN/signed-zero and overflow-cancellation
  oracles. Qualify each emitted ABI and index bound; repeated 2x3x4 is not coverage.
- Dynamic, zero and out-of-family shapes refuse without dispatch. Every admitted
  invocation must preserve original guard precedence: required dimension/product/
  byte overflow remains its original extent failure, not generic candidate
  incompatibility; a well-formed unsupported shape is candidate_incompatible.
  Constant Read shape with mismatching
  Storage still fails its original Read frontier; specialization cannot erase it.
- Same/overlapping source resources retain immutable Read snapshots. A later
  candidate or publication failure preserves earlier effects and old output.
- Mock every acquisition/map/submit/wait/visibility failure; timeout/device loss
  after submission issues no Value, frees no possibly live storage and poisons
  subsequent use. Joined-worker failure/reentry and caller state need controls.
- Missing FP property/coherent memory, ICD/layer/image substitution and installed
  relocation must refuse or pass their explicitly owned gates, never fall back.

Recommendation: first review a parameterized closed static-shape shader issuer
against the existing 2x3x4 control, then declare and falsify a reusable family.
Source adapter/artifact authority remains a subsequent boundary. No backend,
target availability, fusion, residency or performance milestone is claimed here.
