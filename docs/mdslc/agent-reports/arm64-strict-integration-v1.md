# Native Linux ARM64 strict-region integration v1

Implementation-owner report; independent review and native hosted execution are
separate gates. Initial integration base: `5d5682f9189548ee80ba550da8a00fc03ec99fdc`.

## Closed support boundary

The existing source/paired-semantic graph -> MLIR strict issuer -> private ELF
leaf -> checked Session -> ordinary host executable path now has a fixed native
Linux AArch64 branch. CMake passes the closed `linux-aarch64` issuer target and
compiles that leaf for `aarch64-unknown-linux-gnu`, baseline `armv8-a`. No arbitrary
target injection, global ISA flags, source FP relaxation, imported IR authority,
benchmarking or provider-policy change is introduced.

The source driver and artifact ownership proof require little-endian ELF64 of
the authenticated host's machine. A valid x86 DSO cannot confer ARM authority,
nor vice versa. Unsupported architectures, endian layouts and cross compilation
remain refused. The existing LP64 descriptor and host thunk checks remain.

On ARM, `native-strict`, `generated-strict` and `automatic` are the executable
closed candidates. Automatic remains the existing generated-strict choice.
`generated-reassociate`, `existing-native` and OpenBLAS are explicitly unavailable
for their permissive profile; strict requests retain numerical incompatibility
precedence. Empty cases do not bypass availability. No ARM legacy planner,
provider, x86 AVX/FMA or performance qualification is claimed.

Strict increasing-K, separately rounded multiply/add remains mandatory on both
targets. ISA availability never grants numerical contraction permission.
Scalar and row-contiguous object checks authenticate the exact ELF machine and
separate arithmetic; ARM checks reject fused scalar and vector opcodes.

## FP and source acceptance

`ClosedFpEnvironmentV1` saves/restores full raw FPCR/FPSR on ARM, normalizes to
baseline controls/status privately and rejects changed controls. Native fixtures
exercise caller flush controls, rounding, exception status, gradual underflow,
overflow, failure restoration, nesting and concurrent thread isolation. Synthetic
provider/capability adversaries remain test-only substitutes, not real provider
or reassociate support.

Source-host inline assembly remains rejected by symbol ownership. The source
FP fixture therefore compares the standard `fenv_t` snapshot; raw-register
coverage belongs to the native runtime/platform tests, with x86 FTZ/DAZ controls
retained in the source fixture through supported compiler intrinsics.

The ARM qualification harness requires named core tests, executes all selected
closed-source/generated/private-runtime/driver/experimental-package tests and
fails on missing, failed or skipped tests. It does not silently remove alias,
late-read, ordered failure, source authentication, allocation, result ownership
or package acceptance. A dedicated package control builds a clean committed
producer, relocates its installation, deletes only its disposable producer source
and build, then uses the installed driver for exact two-GEMM math, observations
and late checked failures under all three supported policies.

## Validation state before hosted qualification

Local native x86_64 Release build: 325/325 targets passed, exact LLVM/Clang/MLIR
21.1.8, OpenBLAS explicitly OFF, SSD-only task paths, bounded two compile jobs
and cross-lane serialized links. Default scalar generated LLVM is byte-identical
to the original canonical issuer output. Focused FP/candidate/private-value/ELF
tests: 25/25 passed; direct host 137 checks and candidate 151805 checks passed.
Existing source execution, program composition, storage conformance and installed
package checks passed. One initial source-fixture failure correctly rejected raw
inline assembly; the fixture was corrected without weakening that gate.

ARM object inspection against the previously cross-compiled issued row leaf
passes the new separate scalar/NEON arithmetic checks. This is **not native ARM
execution evidence**. Native qualification is pending the dedicated workflow.

## Hosted gate

`.github/workflows/mdslc-arm64.yml` uses the official native
[`ubuntu-24.04-arm` runner](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
scalar and row-contiguous schedules, two build jobs, and a task directory under
`/var/tmp`. Python bytecode is disabled so fixture authentication is not perturbed.
Seven official [apt.llvm.org](https://apt.llvm.org/) LLVM/Clang/MLIR packages are
pinned to the coherent 21.1.8 ARM64 version recorded in the CPU/ARM audit; tools
and native ELF machine are checked before execution. No source LLVM build or
local system package mutation is used. Until a run passes at an exact head, this
branch is an implementation awaiting native hosted qualification, not a product
support or release claim.

## First native qualification and fixture repair

Native run `34599233232` at `e1fe9df` built and authenticated both ARM schedules,
then failed four tests: scalar 89/93 and row-contiguous 91/95 passed, zero skipped.
These were test portability failures, not a passing qualification: two tests
assumed x86 `byval` Storage arguments (including null type dereferences), one
forced-refusal fixture supplied 1x1 storage to a fixed 4x2-by-2x8 source program,
and one ordinary-host legacy C API oracle incorrectly expected ARM support.

The repair retains all x86 record/attribute rejection controls, exercises ARM's
actual indirect-pointer ABI by changing it to `byval`, mutates the triple to the
opposite architecture on either host, and supplies valid source shapes before
testing unavailable-candidate failure with unchanged inputs/output. The legacy
API must report its exact unsupported FP environment without execution/report
publication. Local x86 reruns of all four repaired tests pass; fresh hosted native
execution remains required.

Opaque ARM argument pointers do not encode Storage's pointee layout. Exact
canonical source headers and record-attribute authentication supply that closure,
not LLVM pointer equality. The existing packing and ownership-attribute source
rejection cases, host-thunk ABI tests and call-interface tests are now explicitly
mandatory in the no-skip native qualifier. Recursive `sret` record and all actual
pointer/calling-convention attributes remain checked on both targets.
