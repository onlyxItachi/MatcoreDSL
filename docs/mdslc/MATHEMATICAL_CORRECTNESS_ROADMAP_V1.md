# Mathematical correctness roadmap v1

Planning baseline: 2026-11-10, documentation checkpoint
`b2597bed52de46604574f76eb33663788a5ab994`. This is a small sequence of falsifiable
gates, not a promise to implement every operation or target. The initial request
included broader mathematical coverage; the owner's later direction selects
performance after the chosen correctness gates. These future math proposals are
not a dependency for [P1 performance evidence](PERFORMANCE_EVIDENCE_ROADMAP_V1.md)
on the already qualified strict GEMM route. [CURRENT_STATE.md](CURRENT_STATE.md)
owns the canonical implementation envelope and exactly one immediate frontier;
the [audit](agent-reports/correctness-coverage-audit-v1.md) records code evidence.

## Coverage at this baseline

| Surface | What exists | Missing authority or boundary |
| --- | --- | --- |
| Explicit dense rank-2 f32 GEMM | Authenticated per-call compatibility API and a separate closed-region immutable-value API; bounded CPU and staged GPU realizations | These APIs have different mutation/numerical contracts; no general BLAS coverage follows |
| GEMV, DOT, GER, aligned batch GEMM | Internal canonical indexing/rank/orientation models and inspection controls | No authenticated source operation, authoritative full semantic contract or execution route |
| Elementwise map/sine and partial domains | Internal verified composition model | No closed-region source API or executable map/sine lowering; ordinary host C++ is not this authority |
| Source-visible helpers and strict fused pair | Pure helpers in the existing closed grammar; separately checked adjacent strict lhs-GEMM pair | No arbitrary mathematical-library import, arbitrary DAG fusion or cross-operation optimizer |
| Other arithmetic types | Separate CPU BF16-to-f32 and I8-to-I32 runtime reference contracts | No typed closed-region source surface; ISA/device qualification remains separate |
| General tensors/views and sparse math | Not admitted by the closed-region API | Rank-N, general strides/broadcasting, sparse storage/algebra and source residency need their own contracts |

GEMM M/N output tiles are HOW parameters, not a new mathematical operation.
Unit extents do not turn GEMM into GEMV/DOT, and GER's update semantics must not
be replaced by a singleton-K GEMM. Shared topology does not establish overwrite,
initial-accumulator, rounding, alias/effect or failure behavior.

## Three future broader-math gates — not prerequisites for P1

1. **M1 — Dense GEMV semantics and authenticated inspection.** Propose one
   distinct dense f32 matrix/vector operation with normal and transposed matrix
   orientations. First review a minimal explicit rank-1 storage/value contract:
   today's region Storage, Values and observations describe rows/columns, not
   vectors. Do not conceal this gap with a size-one matrix or solve general
   rank-N/views as a prerequisite. Define a strict, unscaled immutable result,
   positive-zero seed and increasing reduction with separate f32 multiply/add;
   explicitly exclude BLAS alpha/beta/update behavior. Establish shape/orientation,
   empty/zero-reduction, checked extent/capacity and ordered read/publication/
   observation/failure rules. Authenticate declarations after Sema, retain exact
   source/helper identity and pair the admitted semantic Program exactly with an
   untransformed MLIR witness. Reuse the existing GEMV topology owner. Inspection
   succeeds only for this closed envelope; execution requests for the new operation
   still refuse before executable/object publication.
2. **M2 — One strict CPU execution slice.** Only after M1 review, connect that
   same semantics through one compiler-owned realization and checked adapter on
   one declared native platform/toolchain profile. Retain shape/rank/numerical/
   capability guards even for empty math, private output issuance only after
   success, caller FP restoration and original failure/effect prefixes. Exercise
   both orientations with non-square data and independent arithmetic/trace
   oracles, inspect the actual linked implementation, and pass source-connected,
   sanitizer and installed/relocated-package gates. A scalar correctness path is
   enough; no provider, ISA, GPU, fusion or tuning work is required.
3. **M3 — Choose one next mathematical family using the remaining gaps.** DOT
   needs rank-0 result/observation and reduction semantics; GER needs explicit
   initial-destination/update/alias rules; aligned batch GEMM needs rank-3 and
   per-batch/empty behavior without inferred broadcasting. Elementwise arithmetic
   or a GEMM-to-GEMV composition would also require its own bounded proposal.
   Select one after M2 evidence, not all at once. Sine is not an easy substitute:
   its existing internal model requires correctly rounded, non-approximate f32
   behavior. General tensors/views and sparse math remain separate future design
   decisions, not implied deliverables of these dense-operation gates.

If broader-math work is selected, GEMV is the proposed first family because its
existing canonical topology distinguishes matrix N/T orientation with ranks
`[2,1,1]`. DOT introduces a scalar
result, GER introduces update semantics, and batching introduces rank-3. This is
a code-grounded scope choice, not proof that the missing vector boundary is free
or that GEMV is already implementable without review. M1 may stop at an unresolved
contract; it may not silently broaden admission to get a passing example.

## Correctness evidence required at every applicable gate

- Define the operation before choosing its realization: exact ranks/indexing,
  overwrite versus update, initialized values, dynamic shape relations, zero/unit
  extents, invalid enums, overflow and all runtime/storage preconditions.
- Treat f32 as a numerical contract, not exact real arithmetic. Strict reductions
  retain positive-zero initialization, increasing reduction order, separate
  multiply/add and per-operation f32 result boundaries. Test cancellation/FMA and
  reduction-order discriminators, NaN/Inf, signed zero and subnormals under the
  nearest-even, gradual-underflow, masked-exception environment, with full caller
  FP controls/status restored on normal return; promise no NaN payload identity.
  Reassociation or a different dtype requires explicit separate permission and an
  appropriate oracle.
- Use independent strict-f32 arithmetic checks as well as higher-precision error
  checks where appropriate; a tolerance-only comparison cannot prove strict order
  or non-contraction. Do not weaken a profile to accommodate a backend.
- Compare observable traces, not only final arrays: all-MAY-alias external
  resources, immutable reads, owning observations, branches/dead results, earlier
  successful publications and the first failing source frontier. Check pre-mutation
  guards and absence of later effects; retain existing normal-return
  resource/failure limits rather than inventing crash rollback.
- Keep inspection, generated-artifact validation, synthetic guard/mock tests,
  physical execution, sanitizers and hosted results separate. Record immutable
  commits/profiles, exact test scope, unavailable targets/skips and retained
  counterexamples. Re-run the existing GEMM/compatibility firewalls when code changes.

## Target qualification is an independent axis

The planning baseline's immediate frontier was Issue #89's closed bounded-static
strict SPIR-V issuer; PR #99 subsequently completed that bounded issuer contract.
Positive-family generation and graph/ABI/numerical controls do not supply
Vulkan source/runtime, device identity, failure/completion or artifact authority.
PR #84 remains draft and unchanged. Disposition of that issue does
not establish GEMV semantics or authorize a new public API.

Existing source-time output tiles cover explicit single-TU native-baseline
`generated-strict` Linux x64/AArch64 executables, with no source-time `-c`, program
mode, other candidates or fused-pair composition. Existing strict pair scope is
adjacent lhs-chain GEMMs with dominating immutable reads and no retained
intermediate use. Its qualified Linux x64 CPU/NVVM/ROCDL realizations do not
generalize to another operation or target.

For each future operation/target combination, declare the compiler/SDK, ABI,
hardware/OS capabilities, native artifact, execution, numerical/failure and
package gates independently. Existing sm_89/gfx1150 physical results, hosted
issuer-only checks, synthetic capability tests and vendor-driver sanitizer
limitations keep their original distinct scopes. A backend dialect or a CPU
pass cannot substitute for physical accelerator execution. No universal x64,
ARM64, Windows, GPU, Metal or NPU support is promised.

## Performance follows the selected correctness envelope, not every math gate

The selected direction is the [performance evidence track](PERFORMANCE_EVIDENCE_ROADMAP_V1.md),
not M1/M2. P1 is now retained in merged PR #100, including its negative/inconclusive
result; #102 is the current private realization gate. New operations are not
needed to measure the existing qualified GEMM implementation. P1 preserves its
numerical/effect/artifact contract and compares matching measurement modes on an
exclusive host. Broader math may be selected later and still requires its own
gates. No speedup, automatic selection, BLAS parity, Issue #15 closure or API/ABI
freeze is implied by either workstream.
