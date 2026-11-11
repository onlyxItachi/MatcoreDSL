# Performance evidence roadmap v1

Direction update: 2026-11-10. The owner explicitly selects performance after the
chosen correctness boundary. Correctness remains mandatory for each measured
operation/realization; this does **not** mean implementing every future math
operation before timing the existing qualified GEMM route. Broader-math coverage
remains separately documented, not a dependency of this plan.

This document scopes evidence work and links its completed results. It reports
no speedup, optimal tile, dispatch threshold, provider parity or new target qualification.
[CURRENT_STATE.md](CURRENT_STATE.md) owns the current canonical engineering
checkpoint; the existing SPIR-V issuer correctness boundary is independent of the
CPU source-time route measured here and confers no Vulkan source/runtime authority.
PR #99 has since completed that issuer boundary; merged PR #100 records P1's
[bounded negative/inconclusive campaign](../performance/cpu/source-pattern-p1-2026-11-10.md),
not a speedup. The evidence milestone is complete, not a full-matrix performance
qualification. The reviewed follow-up trace is complete (#101); its private
full-N/tail correctness gate (#102, P4 below) is now merged as PR #103.
Its separately explicit source connection (#104 / PR #105, P4b) has completed
qualification. P5's fixed collection is complete and entirely timing-inconclusive.
Exactly one follow-up is [P6 / #108](https://github.com/onlyxItachi/MatcoreDSL/issues/108):
prospective source-call measurement-method qualification, not an optimization or
permission to reinterpret either completed campaign.

## P1 — Authenticated source-time strict output-tile comparison

Question: on one declared exclusive Linux x64 host, how do at least two explicit
source-time M/N output-tile instances compare with the same installation's
unrequested stock `generated-strict` route, under one matching numerical and
measurement contract? A bounded miss or inconclusive noisy run is a valid outcome.

Use the already qualified [source-time pattern lifecycle](SOURCE_TIME_GEMM_PATTERN_V1.md):
single-TU native-baseline `generated-strict`, coherent Clang/LLVM/MLIR 21.1.8,
optimization `none`, and explicit requests such as `3,5` and `5,3`. Keep the
compiler-owned persistent bundles and runtime paths intact. This is not a new
candidate, imported shader/object route, ISA request, GPU recipe or autotuner.

### P1a — Freeze the comparison before collecting timings

- Compile the same frozen C++/region/input harness through the authenticated
  source driver using the stock route and at least two checked tile requests.
  Record source/header/toolchain/installation/profile, produced host/executable,
  private leaf/DSO, runtime dependencies and tile identities. Reuse the existing
  source-pattern consumption checks to distinguish the actual requested instances
  and unchanged original witness; merely naming a tile or matching output is not
  evidence that it executed. Record the stock schedule instead of assuming it.
- Match `strict_f32`: full increasing scalar K, separate f32 multiply/add,
  positive-zero initialization and per-operation f32 boundaries. Keep shape,
  alias, capability, FP environment, publication/observation and failure guards.
  Run the independent strict arithmetic/source-effect/refusal corpus for every
  instance before timing. Neither a tolerance-only oracle nor a permissive
  FMA/OpenBLAS comparator establishes this strict comparison.
- Freeze a small matrix with squares, rectangles, skinny axes and non-divisible
  tails, deterministic input seeds, tile list, warmups, sample/repetition counts,
  timer floor, run-order controls, noise rejection, storage and wall-time budgets.
  Use explicit M/K/N field names; the legacy benchmark's tuple convention is
  M/N/K. Do not silently transpose a profile or enlarge the sweep after results.
- Declare the primary interval as a complete synchronous **source-region call**,
  with its actual checks, read/result/observation allocation/copy and completion
  work included. State exactly when returned ownership is consumed and retired;
  keep that lifetime policy identical between instances. Input generation and
  output verification stay outside timing. Compiler startup/code generation,
  process startup and any private-leaf-only diagnostic are distinct intervals.

The [CPU benchmark contract](../adr/0006-mdslc-cpu-benchmark-contract-v1.md) and
[fairness checklist](../performance/kernel-review-checklist.md#10-benchmark-fairness-and-performance-claims)
provide measurement rules, not an existing authenticated region-call harness.
Reuse applicable timing/provenance machinery, but explicitly bridge this source
route: the legacy runtime benchmark and untimed source-pattern tests cannot
silently stand in for it. Do not compare a kernel-only or allocation-excluded
interval with a complete call, or claim a mode the region API does not expose.

### P1b — Collect one bounded exclusive-host campaign

After the selected correctness gates pass, use one declared logical CPU/cache
group with fixed affinity, actual single-thread execution and a quiet exclusive
host. Record hardware/OS capabilities, governor/boost, environment, compiler
flags, thermal/load observations and cache/preparation policy. No system setting
change or benchmark campaign is performed by this planning update.

Run the frozen process/instance order and its reverse control; retain warmups,
ordered raw samples and exact repetitions. Reach the declared timer floor by
aggregation, authenticate timed outputs outside each interval, and reject
sub-floor, identity-drift, wrong-thread, failed-output or load-contaminated data.
Retain every failure and predeclared skip; no retry-until-win or favorable-shape
selection. An unavailable exclusive host stops collection rather than yielding
a performance verdict from a shared build/test workload.

### P1c — Review evidence; choose a narrow next question

Raw JSON/logs/manifests stay in campaign-only `benchmark_reports/` outside every
Git worktree (the enclosing home directory can itself be a worktree); commit only
a reviewed summary and digest inventory. Publish absolute call times and matched
ratios by shape, spread/order sensitivity, excluded cases and losses as well as
wins. Keep host/shape/interval scope explicit. A valid negative result passes the
evidence milestone and may motivate one separately scoped diagnosis.

There is no automatic winner selection, crossover threshold, default schedule
change, CPU/provider policy change, universal speedup, full BLAS parity, Issue #15
closure, new GPU/SPIR-V performance claim or public API/ABI freeze in P1. Any
subsequent tuning or policy proposal needs its own correctness and evidence gates.

## Completed P1 evidence boundary

[Issue #98](https://github.com/onlyxItachi/MatcoreDSL/issues/98):
**P1: authenticated strict source-time CPU tile performance evidence**.

Acceptance gates:

- [x] Build one bounded authenticated source-region timing harness over the stock
  route and at least two distinct explicit tile requests from one installation;
  preserve actual artifact consumption/ownership and all strict/effect guards.
- [x] Freeze and review the matrix, exact numerical/timing/resource/lifetime
  contract, correctness corpus, host limits and campaign budgets before timings.
- [x] Pass all selected correctness, refusal and artifact controls without changing
  source semantics, candidates or provider policy to improve timing results.
- [x] Collect the bounded matching forward/reverse campaign only on an exclusive
  host; authenticate raw identities, actual thread/affinity and outputs, keeping
  every failure/noise refusal instead of fabricating complete results.
- [x] Independently review the source-to-measurement binding and a digest-backed
  per-shape summary, including negative/inconclusive outcomes; make no automatic
  threshold, broad speedup or Issue #15 closure claim.

M1/M2 GEMV admission/execution, general tensors, sparse math and additional
targets are non-goals, not prerequisites. This plan predates the separately
recorded PR #100 collection and does not manufacture a passing performance result.

## P4 — Private full-N/tail correctness: merged

[Issue #102](https://github.com/onlyxItachi/MatcoreDSL/issues/102) completed the
private strict full-N/tail realization justified by the
[P2/P3 research checkpoint](https://github.com/onlyxItachi/MatcoreDSL/blob/4f3cd4c/docs/mdslc/agent-reports/strict-output-tile-peeling-prototype-v1.md).
[PR #103](https://github.com/onlyxItachi/MatcoreDSL/pull/103) normally merged at
`1f7c491f230251d04d42b5d4a0af5eaf29bcb685`, with the exact frozen tree. The
branch-aware independent proof and upstream replay, full/tail numerical/alias/
empty controls, actual generated sanitizer detections and native ARM gates
passed; clean Release was 352/352, zero skips, and all 26 hosted checks passed.
See the [bounded qualification record](agent-reports/strict-output-peel-integration-qualification-v1.md).
This private result does not select source/default routes or establish speed.

## P4b — Authenticated source connection: qualification complete

Normally merged in PR #105 at `6d30a204aa6f7224b736c2035de15ce20b0e6b3d`.

[Issue #104](https://github.com/onlyxItachi/MatcoreDSL/issues/104), implemented
in [PR #105](https://github.com/onlyxItachi/MatcoreDSL/pull/105),
connects only a separately explicit source request. Retain old request/default
behavior, original WHAT witnesses and strict numerical/effect/failure contracts.
Require a distinct bound instance, exact actual full/tail consumption, typed
source/compiler/artifact/publication ownership, conditional runtime-guard proof,
source oracles, generated sanitizer checks and relocated producer-inaccessible
installation. These gates completed: local Release 356/356, focused source
ASan 4/4, installed execution, native ARM and all 26 final source CI checks.
Retained ISA skips and unsanitized ARM limits remain explicit in the
[qualification record](agent-reports/source-output-peel-qualification-v1.md).

## P5 — New source-call comparison: completed, timing inconclusive

[PR #107](https://github.com/onlyxItachi/MatcoreDSL/pull/107) retains the fixed
old/peeled/stock harness and its one 27.28-second collection after source/harness
qualification and the operator gate in [#106](https://github.com/onlyxItachi/MatcoreDSL/issues/106).
At collection, qualified PR105/107 heads were unmerged; the result retains their
actual producer/SDK identities rather than relabeling artifacts after integration.
All correctness and identity checks passed. Of 270 measured aggregates, 78 failed
the frozen involuntary-switch predicate. No role pair qualified; **all 18 ratios
are null**, including the primary. See the [independently reviewed result](../performance/cpu/source-pattern-p5-2026-11-10.md).
No retry, sample rescue, timing-rule change or speed inference from register
promotion follows. P1 stays immutable and its different retention contract
precludes a cross-campaign timing ratio.

## Exactly one next boundary — P6 measurement-method qualification

[Issue #108](https://github.com/onlyxItachi/MatcoreDSL/issues/108) requires one
prospective observed-wall source-call method with same-binary null, verified
CPU-work and order/interval controls. Freeze the finite contract, discrimination
goal, estimator, inclusion rules and budgets before independent confirmation
data. A sleep-only design was rejected: an observer reporting a fixed baseline
plus that phase could pass while ignoring CPU work. Reuse existing infrastructure;
do not build a general framework. Failed/inconclusive qualification is legitimate.
This cannot recover P1/P5 ratios, establish peeling speedup, qualify unrelated
allocator/cache paths or select an ISA/provider/default policy. No P6 execution
or new performance claim is recorded here.
