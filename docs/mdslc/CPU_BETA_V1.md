# MDSLC CPU beta v1 candidate contract

Date: 2026-08-11

Status: **local Milestone H matrix passed at code candidate `69d099e`; final
independent review, hosted validation, merge, and beta publication are
pending.**

This document fixes the claim and validation boundary for the first CPU beta.
It does not freeze the public API, ABI, backend contract, operation set, or
long-term support policy. It does not authorize a GPU or NPU backend.

## Supported product claim

The candidate is a standalone valid-C++ compiler/runtime product for explicit
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

Windows Release, Debug, and supported sanitizer profiles remain deliberately
MLIR-disabled/default-`capture-v0` for this beta. The current Windows lane does
not claim Matcore-MLIR semantic execution, map execution, or recovered-loop
execution.

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
26. No existing C function signature or record layout changes. The final beta
gate must revalidate the retained 15 exported runtime C functions, strict C17
consumer compilation, enum values, sizes, offsets, and installed shared-library
symbols on the exact candidate commit.

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

The full local matrix was run from clean code candidate `69d099e`; the evidence
is recorded in
[cpu-beta-local-validation.md](agent-reports/cpu-beta-local-validation.md).
The following commits through the local report and review-whitespace correction
change documentation only, so the table distinguishes authenticated local
evidence from the still-pending independent and hosted gates.

| Gate | Required final evidence | Candidate status |
| --- | --- | --- |
| Linux Release | Exact 2x2 matrix: OpenBLAS required/disabled by Matcore MLIR enabled/disabled; configured defaults authenticated | passed locally at `69d099e`: 63/63, 63/63, 58/58, 58/58 |
| Linux Debug | MLIR enabled, default `matcore-mlir`, full registered supported suite | passed locally at `69d099e`: 63/63 with OpenBLAS required |
| ASan + UBSan | MLIR enabled/default `matcore-mlir`; supported in-process scope including FP-environment negatives | passed locally at `69d099e`: 20/20 |
| TSan | MLIR disabled/default `capture-v0`; shared-state/runtime scope | passed locally at `69d099e`: 4/4 |
| Package and relocation | MLIR-on and MLIR-off installs; installed consumer; relocated and source/build-inaccessible consumer | passed locally at `69d099e` |
| Semantic artifact | Explicit `.mdsl -> v1 -> mdsl MLIR -> native object -> executable`; backend producer and stable runtime symbol inspected | passed locally at `69d099e` |
| ABI/install exports | Strict C17, 15 retained runtime C exports, status 26, layouts, SONAME/import boundary, no private MLIR export/path leak | passed locally at `69d099e` |
| Recognition and domains | Accepted focused verifier/equivalence suites plus executable-boundary rejection of recovered/map modules | passed locally at `69d099e`; focused component reviews accepted |
| Windows x64 | Release, Debug, supported sanitizer, MLIR-unavailable negative, DLL/import library, installed consumer, paths with spaces, ZIP artifact | pending hosted run |
| Performance sanity | Correctness and planner sanity only; no new native-parity claim | passed locally at `69d099e`; one guarded 256-cubed OpenBLAS selection sanity, not parity evidence |
| Repository hygiene | Clean tree, `git diff --check`, generated-artifact and semantic-MLIR hygiene | passed locally at `69d099e` and after report/whitespace-only commits |
| Independent review | Fresh adversarial review of the exact candidate with no unresolved high or medium finding | pending |
| Hosted pull request | Normal PR checks green before normal merge | pending |

The beta must not be described as shipped, merged, tagged, or validated by
hosted checks until the remaining rows are replaced by exact evidence from the
pull-request candidate.

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

After this candidate passes Milestone H and merges normally, a separate
explicitly authorized milestone may evaluate the public API/ABI/backend
contract freeze. That freeze must not begin automatically.
