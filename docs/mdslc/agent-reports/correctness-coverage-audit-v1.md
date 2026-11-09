# Correctness coverage and roadmap audit v1

Date: 2026-11-10. Source-inspection baseline:
`b2597bed52de46604574f76eb33663788a5ab994`, isolated branch
`docs/mdslc-correctness-first-roadmap-v1`. Scope: documentation only; no builds,
implementation, deletions, remote issue/PR mutations or performance runs.
`AGENTS.md` was read in full before the audit. The engineering checkpoint remains
PR #93 merge `e8e8008cb61b61a585d8bc1584ca320c6326472d` as recorded by
[CURRENT_STATE.md](../CURRENT_STATE.md), which this lane does not edit.

## Disposition

The user reaffirmed mathematical correctness before performance and broader math
coverage rather than only more elaborate GEMM. [ROADMAP.md](../ROADMAP.md) now
starts with current navigation and that priority, separates broader mathematics
from target qualification/performance, and explicitly labels its old CPU-beta and
post-beta narrative as history. Its original historical evidence is retained.
The short [correctness roadmap](../MATHEMATICAL_CORRECTNESS_ROADMAP_V1.md) proposes
GEMV semantic/admission and CPU-execution gates, then a separately chosen family.
These are proposed follow-ups, not new execution authority or promises.

Issue #89 remains the independent immediate closed static strict SPIR-V issuer
boundary at this baseline; PR #84 remains draft and unchanged. Issuer-family
qualification is not source/runtime/Vulkan authority, and does not fill the
non-GEMM source/semantics gap. No Issue #15 or #20 closure, speedup, native-BLAS
parity, API/ABI freeze or all-target promise is introduced.

## Source-grounded findings

| Finding | Evidence inspected | Consequence |
| --- | --- | --- |
| Both trusted source APIs expose GEMM only | [`mdsl.h`](../../../compiler/include/matcore/mdsl.h), [`region.h`](../../../compiler/include/matcore/region.h) | A GEMV/DOT/GER name in an internal model is not an admitted source call |
| Region data and ordered operations are explicitly bounded | [`MatcoreClosedRegion.h`](../../../compiler/lib/mlir/MatcoreClosedRegion.h): `Dimension` has row/column queries; `Operation::Kind` is Read/Gemm/Publish/Observe/ShapeIf; [`region_storage.h`](../../../compiler/include/matcore/detail/region_storage.h) has row/column f32 resources | Genuine rank-1/0/3 shapes and observations need reviewed contracts; general tensors/sparse are not supported by this route |
| Non-GEMM topology is real but explicitly lacks semantic authority | [`MatcoreContractionModel.h`](../../../compiler/lib/mlir/MatcoreContractionModel.h): topology-only comment; [`MatcoreContractionModel.cpp`](../../../compiler/lib/mlir/MatcoreContractionModel.cpp): distinct operation/rank/orientation builders | Do not infer rounding, overwrite/update, alias/effect, provenance or execution from maps |
| GEMV N/T is a bounded reusable starting point | Same implementation: GEMV ranks `[2,1,1]`, loops m/k, lhs matrix orientation and normal vector orientation; [`matcore_contraction_model_test.cpp`](../../../compiler/tests/mlir/matcore_contraction_model_test.cpp) checks N/T maps, wrong vector orientation and identity collapse | Reuse this one owner rather than inventing a second model or calling N=1 GEMM "GEMV" |
| Other families introduce distinct contracts | Same builder: DOT `[1,1,0]`, GER outer-product update `[1,1,2]`, aligned batch GEMM `[3,3,3]` | DOT requires scalar handling; GER requires initial-output/update rules; batch needs rank-3 and must not infer broadcast |
| Map/sine exist internally, not as closed-region execution | [`MatcoreOps.td`](../../../compiler/lib/mlir/MatcoreOps.td) defines map/domain and correctly-rounded non-approximate sine; [`MatcoreCpuRuntimeLowering.cpp`](../../../compiler/lib/mlir/MatcoreCpuRuntimeLowering.cpp) admits only its explicit GEMM envelope; source headers lack these operations | Sine is not a trivial route to broader source math, nor does ordinary C++ `sin` gain Matcore authority |
| Existing numerical profiles cannot be conflated | [`REGION_COMPILER_V1.md`](../REGION_COMPILER_V1.md#numerical-permissions-and-candidate-choice), [`closed_host_v1.cpp`](../../../compiler/lib/runtime/closed_host_v1.cpp): `strictGemm` and per-frontier candidate/FP checks | Preserve strict order/non-contraction and per-op f32 boundaries; legacy/provider permissive routes are not strict oracles |
| Non-f32 runtime references already exist separately | [`cpu_numeric_reference.h`](../../../compiler/lib/runtime/cpu_numeric_reference.h), [`runtime_c.h`](../../../compiler/include/matcore/runtime_c.h) | Do not claim all non-f32 arithmetic is absent; BF16/I8 runtime APIs do not establish typed region admission or physical ISA execution |
| Source helpers/fusion are not general math-library support | [`EXPERIMENTAL_REGION_FRONTEND_V1.md`](../EXPERIMENTAL_REGION_FRONTEND_V1.md), [`STRICT_FUSED_PAIR_SOURCE_V1.md`](../STRICT_FUSED_PAIR_SOURCE_V1.md), [`GPU_STRICT_FUSED_PAIR_V1.md`](../GPU_STRICT_FUSED_PAIR_V1.md) | Helpers stay in the closed grammar; only the bounded adjacent strict lhs-chain pair has its separately checked authority |
| Target evidence has exact and different scopes | [`SOURCE_TIME_GEMM_PATTERN_V1.md`](../SOURCE_TIME_GEMM_PATTERN_V1.md), [`STAGED_GPU_CANDIDATES_V1.md`](../STAGED_GPU_CANDIDATES_V1.md), [`source_pattern/README.md`](../../../compiler/tests/source_pattern/README.md) | CPU output-tile lifecycle, strict ISA execution, GPU execution, mocks, sanitizers and package tests are not interchangeable proofs |

The historical [contraction reconciliation](../CONTRACTION_CAMPAIGN_RECONCILIATION_V1.md)
accurately recorded the non-GEMM semantic gap at its own checkpoint. Its "next"
tiled/dynamic GEMM recommendation and its statements about then-only runtime
execution are historical, not current queue/authority. Updating every historical
report would erase useful chronology; current navigation now supersedes the old
roadmap's scheduling language explicitly.

## Why GEMV first, and what remains a decision

GEMV is a proposed *family* boundary because the canonical model and tests already
distinguish N/T without requiring update semantics, scalar result handling or a
batch axis. This is an architectural inference from code, not measured effort or
an established source contract. Current source Storage/Values/observations are
rank-2; that must be resolved explicitly before source admission. A size-one
matrix remains GEMM, not proof of rank-1 support. DOT might have fewer loops, but
it adds rank-0 ownership/observation; GER adds update behavior. None is a free
extension from topology alone.

The proposed first engineering boundary is therefore M1: a reviewed minimal
rank-1 contract and authenticated, inspection-only dense strict GEMV N/T. M2 is
one CPU execution slice with independent arithmetic/trace/artifact/package
evidence. No API spelling or general rank-N redesign is selected by this report.
Stop and resolve any unavoidable contract ambiguity; do not widen admission or
silently reinterpret the GEMM-only v1 capture DTO. Keep the existing sealed
Program/exact-witness architecture, not a new overlapping optimizer schema.

## Prepared GitHub issue proposal — not filed

Suggested title: **Correctness-first dense GEMV N/T: rank-1 contract and authenticated inspection**

Issue body:

> Planning proposal after the disposition of the current Issue #89 boundary;
> not authorization for concurrent implementation. Mathematical correctness and
> broader operation coverage come before tuning. Canonical audit baseline is
> `b2597bed52de46604574f76eb33663788a5ab994`.

### Problem

The trusted per-call and region APIs admit GEMM only. The internal contraction
model already has GEMV N/T maps with operand ranks `[2,1,1]`, but explicitly owns
topology only. It provides no source, numerical, value/effect, failure or execution
authority. Region resources, shape queries and observations are row/column based.
More GEMM schedules do not close this mathematical coverage gap.

### Bounded proposal

Review a minimal explicit rank-1 source storage/value/observation contract, then
implement authenticated **inspection-only** dense f32 GEMV with a normal or
transposed matrix. Preserve a distinct GEMV identity; never classify a degenerate
GEMM as GEMV. The result is immutable and unscaled, seeded with +0 and computed
by an increasing reduction with separate f32 multiply/add; no alpha/beta/update
or reassociation profile is introduced. Define shape and orientation rules,
empty/zero reduction behavior, signed/byte/capacity checks, all-MAY-alias reads,
publication/owning observations and ordered failure frontiers before admission.

Use native Clang/Sema canonical declaration/header authentication, frozen helper
source provenance, the existing sealed Program/exact untransformed MLIR witness
boundary and the single existing topology owner. Keep the explicit-GEMM capture
DTO and existing GEMM behavior intact. No new general optimizer/interchange schema.
Execution for this new operation must remain unavailable and refuse before
object/executable publication; CPU execution is a separate follow-up issue.

### Acceptance gates

- [ ] Independent review accepts a minimal rank-1 contract and exact N/T shape,
  initialization, numerical, alias/effect, observation and failure rules; no
  implicit general rank-N/view ABI design. Explicitly document the vector/result
  lifetime contract and retained caller/storage/runtime preconditions.
- [ ] After-Sema trusted source admission retains a distinct operation and matrix
  orientation, exact physical source/helper identity, dynamic shape relations and
  ordered checks in the Program/exact MLIR witness. Malformed or forged topology
  cannot grant source authority.
- [ ] Deterministic non-square N/T fixtures falsify swapped indexing/dimensions;
  tests cover unit/zero/dynamic extents, wrong vector orientation/rank, mismatch,
  overflow/capacity/access rules, aliases, branches/dead values and late failure
  specifications. Clearly separate statically testable predicates from runtime
  obligations reserved for the execution follow-up.
- [ ] Forged/shadowed/macro-altered declarations, stale source/helper closure,
  arbitrary maps, unsupported dtype/layout/stride/batch/update and executable
  requests fail closed with original physical source diagnostics and no published
  executable/object. Existing ordinary C++ and GEMM behavior stay unchanged.
- [ ] Existing frontend/IR/closed-region/compatibility and package firewalls pass
  at immutable commits with independent review. Report actual tests, refusals and
  skips; inspection tests do not become arithmetic or physical-target evidence.
- [ ] Hand off one explicitly scoped strict CPU execution proposal with independent
  order/non-FMA/special-value and effect/failure oracles, actual linked-artifact
  checks, sanitizer and installed/relocated-package gates. Do not claim it passed.

### Non-goals

No execution, provider integration, new target, SPIR-V/Vulkan runtime connection,
GPU/ISA optimization, fusion, cost model, benchmark/parity campaign, general
tensors/views/broadcasting/sparse support, public ABI freeze, Issue #15/#20 closure,
or claims that every operation will execute on every backend. Existing target and
performance evidence retains its original limits.

## Documentation validation and handoff

Validation is source/documentation inspection only, not rerun compiler evidence:

- `git diff --check`: passed; no whitespace errors.
- Compared `ROADMAP.md` from `## Authenticated pivot baseline` through EOF with
  the same baseline suffix from `git show HEAD:docs/mdslc/ROADMAP.md`: identical.
- Local Markdown link-target existence check over all three owned documents:
  33/33 targets exist. This checks local paths, not remote URL availability.
- Source claims were checked against the listed headers, operation definitions,
  builders, lowerer/runtime and contract/test sources; historical test counts are
  not re-presented as new passing results.

The final focused documentation commit is supplied in the agent handoff.
No compiler tests, new builds, device execution, benchmarks, package operations or
deletions were performed in this lane. The pre-existing `CURRENT_STATE.md` and
the compiler tree remain untouched.
