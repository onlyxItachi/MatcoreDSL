# Test-only strict SPIR-V physical harness v1

Status: this is the **harness author's source-level record**, not the current
integration verdict. The integration owner's subsequent compilation, software
and bounded physical receipts are in the
[qualification record](spirv-issuer-integration-qualification-v1.md).
This lane did not build or dispatch. The first root build had 22 expected
missing-field-initializer warnings. No physical, sanitizer, issuer, source or
runtime qualification is inferred from this source-level record alone.
Date: 2026-11-10. The integration owner coordinates the shared build, compiler
prerequisites and actual execution receipts; no additional build ran in this lane.

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

## Receipt qualification wrapper

`compiler/tests/generated_spirv/physical_qualification.py` adds a separate,
no-overwrite campaign over an already built probe. It neither builds nor chooses
a device. The integration owner must first pass the final issuer and independent
upstream SPIRV-Tools checks; that ordering is not replaced by receipt inspection.

```sh
python3 -B compiler/tests/generated_spirv/physical_qualification.py --self-test
python3 -B compiler/tests/generated_spirv/physical_qualification.py \
  --probe build-release/bin/matcore_spirv_vulkan_probe \
  --device N --output /path/to/new-physical-receipts
python3 -B compiler/tests/generated_spirv/physical_qualification.py \
  --probe build-release/bin/matcore_spirv_vulkan_probe \
  --device N --expect-refusal --output /path/to/new-refusal-receipts
```

The output directory must not exist; its parent must exist. Each sequential
subprocess has a 20-second timeout, exclusive raw stdout/stderr files, exact
arguments/status and a strictly parsed JSON receipt. Duplicate keys, nonfinite
JSON constants, malformed output, nonzero matching exits, crashes and timeouts
fail the campaign with no retry or fallback. A compact stdout summary points to
`summary.json`; its artifact catalog hashes the raw/parsed records and an explicit
source snapshot rather than dumping every log into chat. Probe and source
identities are frozen and checked again after the campaign, including failures.
The source catalog is evidence of those exact bytes, not an attestation that an
arbitrary executable was built from them.

The positive family is exactly the twelve proposed M,K,N shapes: 2,3,4; 1,1,1;
1,7,9; 9,7,1; 3,5,7; 17,31,13; 65,3,67; 65535,1,1; 1,1,65535; 3,2,5;
1,257,1; and 5,1,7. All fit the independent total scalar-operation cap.
The wrapper requires exact 19/2 case inventories, K1/K2 skip names, all expected
output/input/guard counts, 2 dispatches and 1 completion per case, zero mismatches,
the selected device's unchanged identity/properties, correct ABI geometry,
shader digests and `source_runtime_authority:false`. It independently checks
non-NaN constant expectation digests and the integer-addressing expectations
without evaluating host floating-point arithmetic. NaN payload hashes are
recorded, not prescribed.

Separate expected-refusal mode requires one checked 2,3,4 shape to return exactly
78 at the selected device/control gate with zero dispatches/completions. It
cannot turn a missing device index, unrelated allocation failure or numerical
counterexample into a passing target refusal. Both modes additionally require 21
unknown/duplicate/missing-option, invalid shape/range and independent-overwork
controls to return 78 with no issuance/loader/device markers. Compile-only large
issuer cases remain separate; no full-envelope physical or performance claim is
inferred from these twelve samples.

Wrapper validation in this lane: `--self-test` passed **36 synthetic checks**,
including valid 19/2-case receipts and mutated count/bits/geometry/skip/authority,
pre-loader refusal and JSON controls. The probe was not started, Vulkan was not
loaded and no hardware ran. `git diff --check` passed. Root executes the real
wrapper only after its independently checked core is settled.

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

Pending: final immutable issuer/checker integration and upstream validation,
inventory/refusal tests, physical matching or counterexample receipts, and final
independent implementation review. This lane's executable evidence is limited to
the synthetic wrapper self-test and whitespace validation. No passing compiler
or device test is inferred from inspection or constants supplied by another agent.
