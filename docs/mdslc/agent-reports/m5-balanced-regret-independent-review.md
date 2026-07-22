# Milestone 5 balanced-regret independent review

## Scope and baseline

This independent, read-only production review covered integration commit
`11d0116cc3fffb6ce64c49ff9aa739324c93a188`, especially:

- `f9d0730` (`fix(bench): balance planner regret measurements`);
- `11d0116` (`test(bench): authenticate balanced regret JSON`);
- the planner-regret implementation in `matcore-bench`;
- its C++ recording-runner and CLI/JSON regressions.

No production source was edited. The review examined timing order, aggregation
semantics, forced-candidate identity, status/correctness propagation,
native/OpenBLAS interval comparability, JSON evidence, and downstream aggregate
regret claims.

## Verdict

**Changes requested: three medium findings, no high findings.**

The forward-then-exact-reverse schedule is a material improvement over the old
selected-once/alternatives-forward-only method. For a monotonic linear drift,
each candidate's two temporal centers are symmetric even when candidate
durations differ. The selected candidate is now measured by the same schedule
as its alternatives, and its regret numerator is the exact value stored in its
candidate record. Execution failures identify pass and variant and fail the
whole run. Those properties are correctly tested.

The remaining findings prevent the emitted metric from yet being a fully
authenticated, versioned performance contract.

## Findings

### M1 — `median_seconds` no longer contains a median

Severity: **medium**

`RegretCandidateResultV2::median_seconds`,
`fastest_legal_median_seconds`, and `selected_median_seconds` now contain the
arithmetic midpoint of the forward-pass median and reverse-pass median. That is
a defensible drift-balanced point estimator, but it is not in general the
median of either pass or of their pooled samples. The already-versioned JSON v2
contract changed meaning without a schema-version change, and neither component
median is emitted. A prose `reason` cannot make consumers type-check that
semantic distinction or audit disagreement between passes.

This matters because Milestone 5 acceptance aggregates median and p95 planner
regret across shapes. A large pass-to-pass spread is currently hidden behind one
number that is labeled a median.

Required resolution: give the balanced estimator an accurately named,
versioned field and emit both forward and reverse medians (or introduce
benchmark schema v3). Derive regret from that explicitly named estimator. Keep
the primary result's ordinary median separate.

### M2 — a timed forced run is not authenticated as the preflight candidate

Severity: **medium**

The preflight records only `legal` and
`complete_implementation_comparison`. Each recursive timing run is then
attributed to `variants[candidate_index]` without checking that the returned
plan:

- selected that exact forced stable ID;
- retained a complete-call timing scope;
- retained the preflight actual thread count, workspace/alignment, packing,
  affinity, and planner version.

The current production `PlannerRunner` appears to honor forced requests, but
the `GemmRunnerV1` boundary does not enforce it. A future regression or test
runner can legally return a different selected implementation and the harness
will silently label its time as the requested candidate. The recording-runner
test always echoes the request, so it cannot detect this failure mode.

Required resolution: retain each preflight plan and reject any timed pass whose
selected ID or comparison-critical metadata differs. Add a test double that
returns a different implementation or timing scope on one pass. Emit the
authenticated per-candidate thread/timing-scope/workspace/affinity metadata so
native/OpenBLAS comparisons remain reviewable without parsing only the auto
plan's diagnostic string.

### M3 — intermittent wrong outputs inside a timing pass can be overwritten

Severity: **medium**

Each recursive benchmark pass verifies output only after all warmups, probe,
calibration repetitions, and measured iterations. GEMM overwrites `C`, so an
intermittently corrupt invocation can be overwritten by a later correct
invocation. The pass will then set `correctness=true` and may contribute to
valid regret. The new failure test covers an execution-status failure, not a
wrong-output-on-an-earlier-iteration case.

This is pre-existing harness behavior, but the balanced-regret path now uses
that single final check as the correctness authentication for every candidate
pass. It does not satisfy the benchmark contract's requirement to reject
non-deterministic result corruption.

Required resolution: add an untimed repeated correctness phase that verifies
each invocation (or another equivalently strong deterministic-corruption
check), and add a runner that corrupts an intermediate invocation but finishes
correctly. Keep oracle work outside the reported performance interval.

## Lower-severity observations

- Forward/reverse blocking cancels first-order monotonic order bias; it does not
  prove an unbiased estimate under nonlinear thermal drift or candidate-local
  cache/provider carryover. Claims should call it a two-block drift-balanced
  estimate, not an unbiased measurement.
- The primary auto-selected run occurs before candidate passes and therefore
  preconditions the selected code path once. Per-candidate warmups reduce this
  effect, but zero-warmup evidence should not be used for performance claims.
- Illegal/unmeasured candidates serialize `correctness=false` with an empty
  measurement reason. This is distinguishable through `legal=false`, but an
  explicit `not attempted` reason would avoid reading it as a correctness
  failure.
- The CLI helper checks selected required fields but does not execute a complete
  JSON-Schema validation. The new structural assertions are useful, but not a
  substitute for validating every candidate object against the committed
  schema.

## Confirmed properties and evidence

- Stable measurable order is forward registry order followed by the exact
  reverse of that filtered order.
- The selected candidate does not reuse the earlier primary timing.
- `selected_median_seconds` equals the selected candidate value in memory and
  serialized JSON.
- Any invalid/incorrect measurable pass prevents a valid regret result; an
  execution failure aborts with pass and stable ID.
- Only legal complete-call candidates enter the fastest-candidate search;
  compute-only diagnostics remain excluded.
- Bound-worker mode rejects multi-thread OpenBLAS whose provider affinity is
  unauthenticated. Complete native packed timings include packing, and the
  OpenBLAS interval includes provider-internal packing.
- Candidate output order and JSON member order are deterministic; numerical
  benchmark values and timestamp are intentionally measurement-dependent.

Focused existing-build validation at the reviewed commit:

```text
ctest --test-dir /tmp/matcore-m5-final-release --output-on-failure -j1 \
  -R '(runtime\.cpu\.benchmark_support|benchmark\.cpu\.(contract|cli_json)|planner\.cpu\.(v3\.advanced|cli\.(automatic|registry_v3|deterministic_v3|compact_affinity_v3)))'

8/8 passed
```

A guarded real `64x64x64`, four-thread-request, compact-affinity run showed the
ordinary selected median and balanced selected estimate were independently
derived (`16.320 us` versus `21.006 us` in that run), and the latter matched
the selected candidate's emitted value. This was a methodology probe, not a
performance claim; its raw JSON remained under `/tmp`.

## Acceptance recommendation

Do not use the current `*_median_seconds` fields for final Milestone 5 aggregate
regret claims until M1 is resolved. Do not call candidate regret authenticated
until M2 and M3 have focused regressions. After those fixes, rerun the exact
host calibration matrix and repeat this review.
