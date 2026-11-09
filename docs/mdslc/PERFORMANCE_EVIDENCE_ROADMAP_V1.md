# Performance evidence roadmap v1

Direction update: 2026-11-10. The owner explicitly selects performance after the
chosen correctness boundary. Correctness remains mandatory for each measured
operation/realization; this does **not** mean implementing every future math
operation before timing the existing qualified GEMM route. Broader-math coverage
remains separately documented, not a dependency of this plan.

This document scopes evidence work only. It reports no new timing, speedup,
optimal tile, dispatch threshold, provider parity or target qualification.
[CURRENT_STATE.md](CURRENT_STATE.md) owns the current canonical engineering
checkpoint; the existing SPIR-V issuer correctness boundary is independent of the
CPU source-time route measured here and confers no Vulkan source/runtime authority.

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

Raw JSON/logs/manifests stay under ignored `benchmark_reports/`; commit only a
reviewed summary and digest inventory. Publish absolute call times and matched
ratios by shape, spread/order sensitivity, excluded cases and losses as well as
wins. Keep host/shape/interval scope explicit. A valid negative result passes the
evidence milestone and may motivate one separately scoped diagnosis.

There is no automatic winner selection, crossover threshold, default schedule
change, CPU/provider policy change, universal speedup, full BLAS parity, Issue #15
closure, new GPU/SPIR-V performance claim or public API/ABI freeze in P1. Any
subsequent tuning or policy proposal needs its own correctness and evidence gates.

## First issue

[Issue #98](https://github.com/onlyxItachi/MatcoreDSL/issues/98):
**P1: authenticated strict source-time CPU tile performance evidence**.

Acceptance gates:

- [ ] Build one bounded authenticated source-region timing harness over the stock
  route and at least two distinct explicit tile requests from one installation;
  preserve actual artifact consumption/ownership and all strict/effect guards.
- [ ] Freeze and review the matrix, exact numerical/timing/resource/lifetime
  contract, correctness corpus, host limits and campaign budgets before timings.
- [ ] Pass all selected correctness, refusal and artifact controls without changing
  source semantics, candidates or provider policy to improve timing results.
- [ ] Collect the bounded matching forward/reverse campaign only on an exclusive
  host; authenticate raw identities, actual thread/affinity and outputs, keeping
  every failure/noise refusal instead of fabricating complete results.
- [ ] Independently review the source-to-measurement binding and a digest-backed
  per-shape summary, including negative/inconclusive outcomes; make no automatic
  threshold, broad speedup or Issue #15 closure claim.

M1/M2 GEMV admission/execution, general tensors, sparse math and additional
targets are non-goals, not prerequisites. No benchmark was run while writing this
plan, and this document does not manufacture a passing performance result.
