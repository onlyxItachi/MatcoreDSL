# MDSLC CPU beta v1 checkpoint contract

Checkpoint date: 2026-08-11. Status reviewed: 2026-09-04.

Status: **the bounded CPU beta was accepted, merged through PR #18 as
`6708b48a5647698469a9af191941bd4755adab7b`, post-merge validated, and marked
by the annotated `mdslc-cpu-beta-v1` checkpoint tag. Issue #17 and GitHub
milestone #6 are closed. No GitHub Release, public API/ABI/backend-contract
freeze, or native-BLAS parity is claimed.**

This document fixes the claim and validation boundary for the first CPU beta.
It does not freeze the public API, ABI, backend contract, operation set, or
long-term support policy. It does not authorize a GPU or NPU backend.

## Supported product claim

The checkpoint is a standalone valid-C++ compiler/runtime product for explicit
F32 rank-two, row-major, contiguous, synchronous host GEMM:

```text
valid C++ .mdsl
  -> native Clang 21 LibTooling frontend
  -> authenticated Matcore IR v1 capture
  -> verified Matcore MLIR mdsl.gemm
  -> private CPU runtime-dispatch lowering
  -> stable matcore_runtime_gemm_f32_v0 C boundary
  -> deterministic legal CPU planner
  -> validated zero-workspace implementation
  -> ordinary native object and executable
```

Matcore IR v1 remains the typed deterministic capture/provenance DTO. Matcore
MLIR is the compositional optimizer representation. The current executable
semantic path is a library-dispatch lowering; it is not generated
Linalg/Vector compute.

Ordinary host C++ and translation units with no Matcore site remain ordinary
C++. Explicit unsupported or unavailable acceleration requests fail closed.
No hidden host/device copy is introduced.

## Build and platform profiles

| Profile | Matcore MLIR | Configured default | Claim boundary |
| --- | --- | --- | --- |
| Source compatibility | `OFF` | `capture-v0` | Existing builds do not acquire an MLIR dependency or semantic default silently. |
| Linux x86-64 CPU beta | `ON`, exact MLIR 21.1.8 | `matcore-mlir` | Native explicit-GEMM semantic execution through CPU runtime dispatch. |
| Windows x64 compatibility | `OFF` | `capture-v0` | Existing compiler/runtime/package path only; an explicit semantic request must fail unavailable without an artifact. |

The Linux beta choice is a product configuration, not a change to the source
compatibility defaults. An explicit `--semantic-pipeline=capture-v0` remains
available in an MLIR-enabled build. The `matcore-mlir` route requires the
native frontend; the AST-JSON bootstrap compatibility frontend must be paired
explicitly with `capture-v0`.

Windows Release, Debug, and the supported clang-cl AddressSanitizer profile
remain deliberately
MLIR-disabled/default-`capture-v0` for this beta. The current Windows lane does
not claim Matcore-MLIR semantic execution, map execution, or recovered-loop
execution. Windows UBSan and a clean-machine VC++ Redistributable installation
are not claimed.

## Installed package contract

`MatcoreDSLConfig.cmake` publishes:

- `MatcoreDSL_MATCORE_MLIR_AVAILABLE`, normalized to exact `ON` or `OFF`; and
- `MatcoreDSL_DEFAULT_SEMANTIC_PIPELINE`, equal to `capture-v0` or
  `matcore-mlir`.

`matcoredsl_add_executable` accepts:

```cmake
SEMANTIC_PIPELINE capture-v0|matcore-mlir
```

Omission uses the package default. An invalid value, unavailable
`matcore-mlir`, or bootstrap/`matcore-mlir` combination fails during consumer
configuration; no automatic fallback occurs.

An MLIR-enabled package installs `matcore-mlir` as a leaf executable. It does
not export an MLIR CMake target, MLIR headers, internal semantic libraries, an
aggregate shared `libMLIR` dependency, or the development-prefix path. An
MLIR-disabled package advertises capability `OFF` and does not install the
tool.

## Execution and resource boundary

The private semantic backend calls the existing one-shot
`matcore_runtime_gemm_f32_v0` entry. That ABI requires zero caller workspace
and performs no MDSLC-owned packing/workspace allocation. It permits reference,
tiled, compiler-vectorized, and linked single-thread OpenBLAS where available.
A linked opaque OpenBLAS provider may manage internal memory under its own
contract while being probed or executed. The path cannot select
workspace-requiring packed or parallel variants.

The additive non-context v1 API evaluates OpenBLAS only for automatic or
forced-provider requests. Forced reference/tiled/vector/native requests retain
truthful linked-but-uninspected diagnostics and perform no provider call;
forced packed-B preparation and execution use that same no-provider boundary.

Native packed and persistent-parallel variants remain supported through the
existing additive caller-workspace and execution-context C APIs. Context
creation validates every compiled variant. With a linked OpenBLAS provider,
first process use runs one provider-conformance GEMM and every context creation
runs finite and special-value provider validation GEMMs only if that provider
is conformant. The provider may initialize or manage internal memory during
either boundary, before a later forced-native context-backed request. Their
workspace, ownership, thread, and lifetime contracts are not weakened or hidden
to make the semantic one-shot path appear broader.

Packed-B v1 remains caller-owned borrowed storage with synchronous serial
reuse and manual invalidation after source/packed mutation, relocation, or
deallocation. It is not a globally shared immutable-weight cache.

## Numerical, legality, and ABI boundary

The executable semantic route accepts only the verified
`explicit-gemm-f32-v1` numerical profile. On the physically validated Linux
x86-64 scope, the compile/runtime boundary enforces source-type evaluation,
round-to-nearest, non-trapping exceptions, and FTZ/DAZ disabled before packing
or destination mutation. Native workers are admitted before parallel work, and
the supported linked single-thread OpenBLAS adapter is checked for conformance.

Unsupported execution state returns additive status
`MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0` with numeric value
26. No existing C function signature or record layout changes. The Milestone H
gate revalidated the retained 15 exported runtime C functions, strict C17
consumer compilation, enum values, sizes, offsets, and installed shared-library
symbols on the immutable local full-matrix candidate.

Alias, overlap, descriptor, shape, stride, alignment, workspace, capability,
and FP-environment failures remain pre-execution failures. Existing contracts
requiring unchanged caller output on such rejection remain in force.

## Inspection-only semantic work

The internal dialect represents verified `mdsl.map`/`mdsl.sin` composition and
closed `all`, slice, indices, and predicate domain forms. The current CPU
lowerer rejects these modules. There is no public map API and no map/sine
execution or fusion claim.

One canonical ordinary-C++ GEMM loop can be conservatively recognized and
compared with an authenticated explicit GEMM in the common semantic model.
This is analysis/equivalence inspection only. It does not rewrite source and
cannot enter the executable CPU lowering. Recognition or permission failure
preserves ordinary C++ behavior.

## Performance truth

Milestone 7 native-BLAS parity remains partially passed. Issue #15 and GitHub
milestone #5 remain open, the complete authenticated forward/reverse evidence
pair does not exist, and no parity completion tag may be created. Cooperative
packed-B preparation remains production-dormant.

The beta claim is deterministic selection of the best legal validated
implementation in the supported runtime search space. OpenBLAS may be selected
when it is faster. Such a selection is not native-MDSLC parity, and parity is
not manufactured by changing the benchmark contract or envelope.

## Milestone H acceptance matrix

The final local matrix was run from a clean immutable clone of code candidate
`6796fd8`; the evidence is recorded in
[cpu-beta-final-candidate-validation.md](agent-reports/cpu-beta-final-candidate-validation.md).
It supersedes the historical `69d099e` matrix because later product changes
hardened composition roots, package provenance, and provider-conformance
routing. The later commits `f8e38ea` and `1d084a1` are focused CI/test and
Windows floating-point-profile fixes, not documentation-only commits; their
focused hosted validation is recorded separately below. The immutable full
local matrix remains the `6796fd8` evidence boundary.

| Gate | Required final evidence | Candidate status |
| --- | --- | --- |
| Linux Release | Exact 2x2 matrix: OpenBLAS required/disabled by Matcore MLIR enabled/disabled; configured defaults authenticated | passed locally at `6796fd8`: 63/63, 63/63, 58/58, 58/58 |
| Linux Debug | MLIR enabled, default `matcore-mlir`, full registered supported suite | passed locally at `6796fd8`: 63/63 with OpenBLAS required |
| ASan + UBSan | MLIR enabled/default `matcore-mlir`; supported in-process scope including FP-environment negatives | passed locally at `6796fd8`: 20/20 |
| TSan | MLIR disabled/default `capture-v0`; shared-state/runtime scope | passed locally at `6796fd8`: 4/4 |
| Package and relocation | MLIR-on and MLIR-off installs; installed consumer; relocated and source/build-inaccessible consumer | passed locally at `6796fd8` |
| Semantic artifact | Explicit `.mdsl -> v1 -> mdsl MLIR -> native object -> executable`; backend producer and stable runtime symbol inspected | passed locally at `6796fd8` |
| ABI/install exports | Strict C17, 15 retained runtime C exports, status 26, layouts, SONAME/import boundary, no private MLIR export/path leak | passed locally at `6796fd8` |
| Recognition and domains | Accepted focused verifier/equivalence suites plus executable-boundary rejection of recovered/map modules | passed locally at `6796fd8`; focused component reviews accepted |
| Windows x64 | Release, Debug, supported sanitizer, MLIR-unavailable negative, DLL/import library, installed consumer, paths with spaces, ZIP artifact | passed at hosted head `1d084a175772f286b04eb1802e2c4d8272533ede`; push and PR #18 Windows workflows succeeded |
| Performance sanity | Correctness and planner sanity only; no new native-parity claim | passed locally at `6796fd8`; one guarded 256-cubed OpenBLAS selection sanity, not parity evidence |
| Repository hygiene | Clean tree, `git diff --check`, generated-artifact and semantic-MLIR hygiene | passed locally at `6796fd8` |
| Independent review | Fresh adversarial review with no unresolved high or medium finding | passed; the immutable full-matrix baseline remains `6796fd8` and the later focused fixes have green hosted evidence |
| Hosted pull request | Push and PR #18 workflow checks green before normal merge | passed at product/test head `1d084a175772f286b04eb1802e2c4d8272533ede` |

Exact hosted evidence for `1d084a175772f286b04eb1802e2c4d8272533ede` is:

| Workflow | Push evidence | PR #18 evidence |
| --- | --- | --- |
| `mdslc-native` | [run 31522956062](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/31522956062), 7/7 jobs passed | [run 31522958911](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/31522958911), 7/7 jobs passed |
| Legacy `ci` | not a required push duplicate | [run 31522958924](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/31522958924), 1/1 job passed |
| `repository-hygiene` | [run 31522956052](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/31522956052), 1/1 job passed | [run 31522958913](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/31522958913), 1/1 job passed |
| `mdslc-windows` | [run 31522956094](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/31522956094), Release 44/44, Debug 31/31, clang-cl ASan 1/1, install/consumer/artifact/ZIP passed | [run 31522958916](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/31522958916), Release 44/44, Debug 31/31, clang-cl ASan 1/1, install/consumer/artifact/ZIP passed |

These runs closed the pre-merge hosted Linux, Windows compatibility, sanitizer,
package, legacy, and hygiene validation gate. PR #18 subsequently merged as
`6708b48a5647698469a9af191941bd4755adab7b`; post-merge checks passed, the
annotated `mdslc-cpu-beta-v1` checkpoint tag was created, and Issue #17 / GitHub
milestone #6 closed. No GitHub Release exists for the tag. Windows remains a
compatibility profile and does not gain a Matcore-MLIR semantic-execution claim.

## Explicitly unsupported claims

The CPU beta does not claim:

- native GEMM parity with OpenBLAS across the Milestone 7 envelope;
- Windows Matcore-MLIR semantic execution;
- executable `map`, `sin`, partial-domain, or recovered-loop replacement;
- generated Linalg/Vector kernels;
- public GEMV, GEVM, BF16, INT8, or fused-operation language APIs;
- a general immutable transformed-operand/cache contract;
- a final public API, ABI, backend-contract, or support-policy freeze; or
- CUDA, HIP, Metal, Vulkan, NPU, heterogeneous placement, or any GPU work.

The bounded merge and checkpoint-tag integration gates are closed. A separate
explicitly authorized milestone may evaluate the public API/ABI/backend
contract freeze. That freeze does not begin automatically.
