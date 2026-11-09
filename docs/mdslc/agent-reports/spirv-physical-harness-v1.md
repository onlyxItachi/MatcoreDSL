# Test-only strict SPIR-V physical harness v1

Status: **implemented, not yet compiled or executed** in this lane. No physical,
sanitizer, issuer, source or runtime qualification is claimed by this report.
Date: 2026-11-10. The integration owner coordinates the one shared build, compiler
prerequisites and actual execution receipts; no additional build ran here.

## Scope and authority

Owned files are `compiler/tests/generated_spirv/vulkan_probe.cpp`, its
`strict_oracle.h`, and this report. The follow-up branch
`agent/mdslc-spirv-physical-harness-v1` starts from documentation head `9a9e8d2`;
the focused harness commit is independently cherry-pickable after the issuer
prerequisite. No CMake, production issuer, runtime, source API or existing roadmap
was changed in this lane. No shader-file or serialized-artifact import exists.

The test directly calls the private checked-shape `issueStrictGemmArtifactV1`,
then `verifyStrictGemmArtifactV1` for full replay, and checks the expected role,
shape/bytes, stride, offset, mutability, entry and dispatch ABI before loading
Vulkan or creating any shader module. The verified issuer's actual words,
entry points, bindings and dispatch arrays feed the Vulkan calls. Inspection
does not itself grant production execution authority: this executable is an
isolated test process, not a source driver or runtime candidate.

## Numerical and addressing controls

All expected output bits are computed and frozen before `dlopen` of the Vulkan
loader, device enumeration or dispatch. LLVM APFloat uses binary32 and explicit
nearest-ties-even for every multiplication and addition; no host floating-point
arithmetic, host FP environment or provider result is the oracle.

The 17 special cases use independently calculated rational/integer f32 constants
provided by the adversarial lane: positive/negative subnormal input, exact
subnormal product, positive/negative inexact-underflow RNE ties, signed-zero seed,
non-FMA, increasing-K order, overflow order, RTE-versus-RTZ, both tie-even cases,
positive/negative infinity, infinity-times-zero, opposed infinities and quiet NaN.
Two additional APFloat controls require the FMA alternate `0xa8800000` and reverse
reduction alternate `0x3f800000`; strict results for those fixtures are +0.
NaNs compare by class only; all non-NaN bits, including zero signs, must match.

Nonuniform row/column-dependent small integers distinguish rectangular indexing;
their expectations are cross-checked with an independent integer sum whose every
prefix is explicitly checked to stay exactly representable in f32. A separate
last-K-only row/column contribution detects reduction truncation. All special
fixtures require K>=3; smaller positive K executes the two addressing cases and
lists each special case as skipped with its precondition. Negative-zero products
pad *every* K position in the signed-zero case rather than erasing its seed
discriminator with positive-zero padding.

The independent test work cap is `2^20` scalar fill/multiply/add operations across
all cases: `case_count * M*N * (2*K+1)`. Division-based refusal occurs before
freezing or issuing a shape. This is not the issuer envelope or a source semantic
dimension restriction. Larger compile-only family cases belong in separate
issuer/validator tests, not in this physical probe.

Output prefix/suffix guards surround an aligned exact-range descriptor. Before
each fill, every interior is poisoned with a nonmatching bit/class: qNaN for a
non-NaN expectation and finite 123 for a NaN expectation. Therefore a missing
write cannot pass the NaN-class comparison. Inputs are checked bit-for-bit after
completion. Device comparison, input and guard mismatches are counterexamples,
not skipped tests.

## Device, memory and completion limits

Only the explicitly indexed device can execute; there is no automatic fallback.
Require Vulkan >=1.2, `shaderDenormPreserveFloat32`, `shaderRoundingModeRTEFloat32`
and `shaderSignedZeroInfNanPreserveFloat32`. A positive sample additionally
requires a physical integrated/discrete AMD device with `VK_DRIVER_ID_MESA_RADV`.
NVIDIA or software devices lacking the controls refuse without dispatch; even a
software device exposing them cannot make the physical claim.

Actual dispatch axes, local size/invocations, storage-buffer ranges, descriptor
counts, binding layout, memory type compatibility and allocation count are
checked. Three separate allocations use only HOST_VISIBLE|HOST_COHERENT memory.
The independent staging alignment envelope is <=65536 bytes. No noncoherent
flush/invalidate path, device-local copy policy or hidden transfer route exists.
The descriptor prefix respects the actual minimum storage-buffer offset alignment;
it is not confused with a cache-maintenance offset.

Commands contain HOST-to-COMPUTE, fill-to-GEMM and COMPUTE-to-HOST memory
dependencies, followed by checked fence completion before output inspection,
next host writes or resource retirement. A failed queue submit or fence wait
prints the error receipt and calls `std::_Exit(1)` without unwinding or freeing
possibly live resources. Other unexpected Vulkan API failures are also fail-stop.
This process-only behavior is not a production adapter quarantine/recovery law,
crash atomicity, sandbox or promise of caller FP-state restoration.

The adversarial lane read both source files and reported no numerical, visibility
or ordering blocker by inspection, not execution. Its follow-up hardening requests
are incorporated: integer-prefix representability is checked rather than inferred
from the work cap, ABI bindings are exact, and fail-stop logging catches allocation
exceptions before unconditional `_Exit` so unknown completion cannot unwind.

The receipt captures device/driver UUIDs and versions, physical type, float-control
properties, selected index, shape, shader SHA-256 digests, ABI dispatch/local size,
test work count, submitted dispatches, checked completions, per-case expectation
digests, output/input/guard check counts and mismatch information. Every receipt
sets `source_runtime_authority:false` and `strict_contract_qualified:false`.
Only an actually matching run is classified `BOUNDED_PHYSICAL_SAMPLE`; an
unavailable configuration is `REFUSED` and a numerical/storage defect is
`COUNTEREXAMPLE`.

The relevant primary contracts were checked against the installed Vulkan headers
and Khronos documentation for [float-control properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFloatControlsProperties.html),
[physical limits](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceLimits.html)
and [memory dependencies/fences](https://docs.vulkan.org/spec/latest/chapters/synchronization.html).
These API contracts are not evidence that this implementation has run correctly.

## Integration commands and pending evidence

The integration owner provides the opt-in test-only
`MDSLC_BUILD_SPIRV_VULKAN_PROBE=ON` target `matcore_spirv_vulkan_probe` in the shared
`build-release/bin` directory. It needs installed Vulkan headers, the coherent
issuer/MLIR/LLVM stack and `${CMAKE_DL_LIBS}`, but must not link libvulkan: loading
before `main` would defeat the pre-load oracle boundary. No system dependency
installation, benchmark or default physical CTest is requested.

```sh
build-release/bin/matcore_spirv_vulkan_probe --oracle-only
build-release/bin/matcore_spirv_vulkan_probe --inventory
build-release/bin/matcore_spirv_vulkan_probe --device N --shape 2,3,4
build-release/bin/matcore_spirv_vulkan_probe --device N --shape 3,5,2
build-release/bin/matcore_spirv_vulkan_probe --device N --shape 1,3,7
build-release/bin/matcore_spirv_vulkan_probe --device N --shape 7,1,1
```

Replace N with the observed desired device index; never infer device ordering.
`--oracle-only` runs the APFloat/analytic controls and a padded 2x7x3 family,
without issuing or loading Vulkan, for nonphysical CTest coverage. Inventory also
does not dispatch. Execution returns 0 only on match, 78 on explicit refusal and
1 on a counterexample/error. Unknown/duplicate options, malformed dimensions,
unsupported/inapplicable shapes and excess test work refuse before dispatch.
Capture stdout as the JSON receipt and preserve stderr separately.

Pending: issuer prerequisite integration, compilation, software-oracle execution,
inventory/refusal tests, physical matching or counterexample receipts, and
independent implementation review. `git diff --check` is the only executable
validation performed in this lane before handoff; no passing compiler/device
test is inferred from source inspection or constants supplied by another agent.
