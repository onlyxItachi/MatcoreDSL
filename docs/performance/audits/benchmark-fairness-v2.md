# Benchmark fairness audit v2

Date: 2026-07-26

Status: independent Milestone 6 review; **not accepted yet**

This review covers `matcore-bench`, the schema-v5 deep-audit runner, the
frozen Milestone 6 methodology, and the first attempted full collection. It
does not change a kernel, planner rule, runtime ABI, or public API.

## Claim vocabulary

- **[measured]**: observed by an identified executable, command, or generated
  report on the declared host.
- **[derived]**: calculated from measured values or reviewed control flow.
- **[source-backed]**: established directly by the identified repository
  source.
- **[hypothesis]**: plausible, but not established by the available evidence.
- **[proposed]**: a future change or experiment, not current behavior.

## Evidence identity

- **[source-backed]** The benchmark timing implementation was inspected
  through commit `c5031e111ae1640fb847fc8a4998f329e93e2198`.
- **[source-backed]** The deep-audit runner was inspected through commit
  `13f9c14303b21cec1784211385b4dfbf8f9890c1`.
- **[source-backed]** The planner runner's last relevant source commit was
  `dc99a8e3579181742bb50840e1225248bf3f49d9`.
- **[source-backed]** The frozen methodology was inspected at
  `c74f24fec09aa8730d94b3d401dc3d6609c02492`.
- **[measured]** The first attempted full collection used a clean benchmark
  binary reporting source commit
  `8c606d6de2e09429f85a24ff6d081a9c610ecd8e`. Its external manifest SHA-256
  was `aaf8f4816e6550be80fea038c67be50f76f5cf9870674f084e7493f486093ea5`
  when reviewed.
- **[measured]** That manifest contains 568 attempted records: 273 passed,
  118 reused, 176 expected-rejection-classified, and one failed. It also
  contains 49 predeclared runtime-bound skips. Collection stopped at case 567,
  `511x513x515`, forced four-thread native parallel AVX2, after a guarded hot
  timing fell below the one-millisecond aggregate floor. It is incomplete and
  is not Milestone 6 acceptance evidence.
- **[measured]** Repository hygiene passed after this read-only review.

The external bundle and local reproduction directories are intentionally not
part of Git.

## Executive verdict

The benchmark core has several unusually strong properties: explicit timing
scopes, caller-visible workspace, provider thread control, distinct left-input
sequences, deterministic inputs, a double-precision oracle, source
provenance, balanced planner-regret passes, and fail-closed forced variants.
Those properties support bounded diagnostic conclusions.

The attempted full matrix is not yet independently acceptable. Three
contract gaps and one incomplete run are high-severity Milestone 6 blockers:

1. **[measured]** `--resume` can reuse a report produced with a different
   benchmark commit and different measurement parameters.
2. **[source-backed]** The deep-audit matrix does not schedule the declared
   allocation-included one-shot mode.
3. **[source-backed]** prepacked-B mode excludes B preparation but does not
   measure its one-time preparation cost, so it cannot report the declared
   amortized total.
4. **[measured]** the first full collection stopped before completing the
   matrix.

Multi-thread native/OpenBLAS comparisons also use different placement
contracts. They are diagnostics, not fair parity results, until matched
placement is available or the asymmetry is made an explicit exclusion.

## Timing-scope audit

### Reused workspace, packing included

- **[source-backed]** The runner's `complete-hot` and `complete-cold` cases
  pass `--reuse-workspace --include-packing`.
- **[source-backed]** Output and workspace allocations occur before timing.
  Each native packed invocation performs the implementation's transient
  packing inside `execute`; OpenBLAS's opaque provider packing remains inside
  `cblas_sgemm`.
- **[source-backed]** Planning and persistent-context creation are outside the
  interval. Synchronization is inside because every invocation calls
  `runner.synchronize`, and parallel execution itself waits for workers.
- **[derived]** These results are fair complete-call, reused-workspace
  comparisons when thread placement is also matched. They are not end-to-end
  one-shot results.

### Allocation-included one-shot

- **[source-backed]** `matcore-bench --include-allocation` can re-plan and
  allocate output/workspace for every invocation while retaining inputs
  outside the interval.
- **[source-backed]** `run_cpu_deep_audit.py` always emits
  `--reuse-workspace`; it has no allocation-included suite.
- **[derived]** The full runner therefore cannot support the methodology's
  declared one-shot result or any claim about end-to-end allocation cost.
- **[proposed]** Add a separately named allocation-included suite. Keep input
  generation and the independent oracle outside timing, but state exactly
  which output/workspace allocation, planning, provider initialization, and
  context costs are included.

### Packing-excluded diagnostic

- **[source-backed]** The diagnostic prepares full-K A and B layouts before
  timing, then invokes the internal packed AVX2 microkernel traversal.
- **[source-backed]** The runner schedules both single-thread packed AVX2 and
  AVX-512 IDs, but the planner runner implements `--exclude-packing` only for
  AVX2. AVX-512 cases are rejected.
- **[derived]** This is a useful AVX2 microkernel diagnostic. It is neither
  complete native execution nor directly comparable to complete CBLAS.
- **[proposed]** Either schedule only the implemented diagnostic or add an
  independently authenticated AVX-512 equivalent. Expected rejection should
  not be mistaken for AVX-512 compute evidence.

### Prepacked B and distinct A inputs

- **[source-backed]** B is prepared once outside timing. Timed execution still
  includes transient A packing, compute, stores, and synchronization.
- **[source-backed]** `--lhs-sequence N` allocates N distinct, alignment-padded
  A matrices and cycles them while holding B fixed. The same cardinality is
  replayed untimed through the oracle.
- **[source-backed]** Sequence lengths 1, 4, 16, and 64 are predeclared.
- **[source-backed]** Neither JSON nor the runner records B preparation time.
  Sequence length therefore changes the A working set, but does not amortize a
  recorded B-preparation duration.
- **[derived]** Prepacked execution throughput is valid. A one-time prepack
  cost or amortized end-to-end total is not currently reportable.
- **[proposed]** Record a separately timed, guarded B-preparation phase and
  report `prepare + repetitions * execute` without folding it into the
  steady-state execution interval.
- **[source-backed]** Parallel packed variants do not implement prepacked-B
  execution and are rejected. The rejection is evidence of the missing
  feature, not a repeated parallel performance point.

## Thread-count, affinity, and provider audit

### Native variants

- **[source-backed]** Serial variants execute with one implementation thread.
  Under the runner's physical-core/compact policy they are dispatched to one
  authenticated bound worker.
- **[source-backed]** Native parallel variants use persistent workers with
  strict per-worker affinity. The requested worker count can be clamped to the
  number of 128-row macro tiles.
- **[source-backed]** Schema v5 records requested threads, plan
  `actual_threads`, SMT policy, affinity policy, whether worker affinity was
  applied, and a detailed placement diagnostic.
- **[measured]** The attempted bundle includes many clamped cases, such as a
  requested 12-thread `768^3` call using six workers and a requested
  12-thread `1024^3` call using eight.
- **[derived]** Cross-variant aggregation must join on `actual_threads`, not
  only the requested count. Clamped points remain useful planner behavior but
  are not equal-thread comparisons to an unclamped provider result.

### OpenBLAS

- **[source-backed]** The collection process sets
  `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
- **[source-backed]** Every OpenBLAS call uses
  `openblas_set_num_threads_local(requested)`, checks
  `openblas_get_num_threads()`, calls typed row-major `cblas_sgemm`, and
  restores the previous local count.
- **[source-backed]** Native and provider pools are not nested. A
  multi-thread provider call is rejected inside a bound native worker.
- **[derived]** These controls are sufficient to authenticate the configured
  OpenBLAS team size for this pthread provider. They do not instrument how
  many provider workers were simultaneously active for a particular small
  call; `actual_threads` is configured-provider count, not sampled
  concurrency.

### Placement mismatch

- **[source-backed]** Native parallel cases use physical-cores-only plus
  compact authenticated worker affinity.
- **[source-backed]** Multi-thread OpenBLAS cases deliberately use
  allow-SMT plus `affinity=none`, because provider-worker affinity is not
  authenticated.
- **[derived]** Equal numeric thread counts under those two policies are not
  equal placement. On this heterogeneous-frequency host, the difference can
  materially affect throughput.
- **[proposed]** Report multi-thread provider/native results in separate
  placement strata. Do not compute a parity ratio unless both records have
  equal actual thread count and an equivalent authenticated placement policy.
  If OpenBLAS affinity cannot be authenticated, label the comparison
  unbound-provider diagnostic.
- **[source-backed]** Planner-regret requests use the bound native policy.
  Multi-thread OpenBLAS is therefore illegal in those regret sweeps. Regret is
  valid for that policy's legal candidate set, not for an unconstrained
  provider candidate set.

## Warmup, aggregation, and statistics

- **[source-backed]** Warmups invoke the exact selected plan and timing mode.
  A probe and bounded calibration select an aggregate repetition count for hot
  measurements.
- **[source-backed]** Current hot aggregation targets twice the requested
  floor and applies another factor-of-two margin after calibration. Every
  retained aggregate sample must still clear the requested floor.
- **[source-backed]** Cold-cache and planner-regret cases deliberately use a
  one-microsecond floor; ordinary complete hot calls use the caller's
  one-millisecond floor.
- **[derived]** The one-millisecond floor is defensible for ordinary hot
  timing. One-microsecond points are timer diagnostics and must be excluded
  from aggregate performance or regret claims when timer noise is material.
- **[source-backed]** Seven retained samples produce minimum, median, and
  nearest-rank p95. The JSON does not retain the individual sample vector.
- **[derived]** Without individual samples, an independent reviewer cannot
  reconstruct the median/p95, inspect multimodality, apply a predeclared
  outlier rule, or correlate samples with host telemetry.
- **[proposed]** Keep the full per-sample vector only in external raw JSON.
  Sanitized summaries should continue to contain aggregates.

## Cache-state audit

- **[source-backed]** Hot-cache measurements perform warmup, probe, and
  calibration before retained samples. This is intentionally steady-state,
  not first-call latency.
- **[source-backed]** Cold mode touches one byte per cache line of a 64 MiB
  buffer before each retained sample and does not aggregate several GEMMs
  behind one eviction.
- **[derived]** Sixty-four MiB exceeds the host's reported aggregate LLC
  capacity, but userspace touching cannot prove complete eviction of every
  operand, provider-internal buffer, page-table entry, or TLB entry.
- **[derived]** The methodology correctly calls cold mode “best effort.”
  Cold results must not be labeled a specific cache hierarchy state.
- **[source-backed]** The environment records governor, min/max frequency
  policy, boost state, and a timestamp, but not measured core frequency during
  samples.
- **[derived]** AVX2/AVX-512 frequency drift or thermal effects cannot be
  corrected or asserted from this metadata.

## Correctness and output authentication

- **[source-backed]** Inputs are deterministic from a recorded seed and shape.
  Every output element is checked for finiteness and participates in a
  checksum.
- **[source-backed]** Small results use a full elementwise double-precision
  oracle. Larger results use 64 deterministic element samples plus an
  independent double-precision checksum with an error bound.
- **[source-backed]** The final timed output is authenticated immediately
  after timing. A separate untimed phase repeats the exact timed execution
  cardinality and authenticates every replayed invocation.
- **[derived]** This design prevents the oracle from contaminating the timed
  interval and covers every distinct A in sequence mode.
- **[derived]** For large results it is a sampled-plus-checksum oracle, not a
  full elementwise oracle. Sparse checksum-preserving corruption outside the
  64 sampled locations is theoretically possible. Performance reports must
  use the emitted oracle name rather than claim full-element validation.

## Planner-regret fairness

- **[source-backed]** Every legal complete-call candidate is measured in
  stable registry order and then exact reverse order.
- **[source-backed]** The selected candidate is not allowed to reuse the
  primary timing. Each candidate receives the same two-pass treatment.
- **[source-backed]** Untimed correctness replay is placed after forward
  timing and before reverse timing, reducing validation-order bias.
- **[source-backed]** Each balanced estimate is the arithmetic midpoint of the
  forward and reverse pass medians; plan fields are reauthenticated against
  preflight.
- **[derived]** This is materially fairer than a one-direction candidate
  sweep and is acceptable for the explicitly bound policy's legal set.
- **[derived]** The one-microsecond floor remains too weak for tiny candidates.
  Timer-noise cases must be excluded visibly, not silently included in regret
  aggregates.

## Process order, frequency drift, and outliers

- **[source-backed]** Full-matrix cases are sorted deterministically by a
  textual key. Each case launches a new process. Variant order is not
  randomized or alternated across independent replicates.
- **[derived]** Fixed order leaves complete-call comparisons exposed to
  temperature, frequency, desktop load, and process-order drift. The balanced
  order inside one planner-regret command does not repair the separate
  complete-call matrix.
- **[measured]** Other Milestone 6 probes already observed large independent
  run-median swings on this interactive host. Sub-percent differences are not
  defensible from one fixed-order seven-sample process per cell.
- **[proposed]** Use at least two reversed or Latin-square process orders for
  calibration and parity claims. Retain the declared shape partitions and
  report run-to-run ranges.

## Provenance and resume integrity

### What is strong

- **[source-backed]** Each benchmark report carries an exact Git object ID,
  clean/dirty state, provenance origin, compiler, flags, build type, host,
  provider, capability, topology, complete command parameters, and result
  metadata.
- **[source-backed]** The runner refuses in-repository raw output except under
  ignored `benchmark_reports/`, hashes each accepted raw report, and writes
  every command and explicit skip to the manifest.
- **[source-backed]** The diagnostic/calibration/holdout shape partition is
  frozen before optimization. Slow scalar variants have explicit,
  deterministic work bounds rather than disappearing from the manifest.

### Confirmed resume defect

- **[source-backed]** `authenticate_report()` verifies schema/version, clean
  provenance state, LHS sequence length, shape, requested variant, timing,
  correctness, and an upper bound on actual threads.
- **[source-backed]** It does not require the expected source commit, a
  benchmark executable digest, runner digest, requested thread count,
  warmup/iteration counts, timer floor, cache mode, packing mode, allocation
  mode, alignment, seed, memory bound, SMT policy, affinity policy, provider,
  topology, or compiler flags.
- **[measured]** An isolated reproduction invoked the current runner with
  `--warmup 0 --iterations 1 --timer-floor-us 1 --resume`. It reused a clean
  old report containing warmup 2, iterations 7, a one-millisecond floor, and
  source commit `8c606d6...`, then wrote a manifest claiming the new
  parameters.
- **[derived]** A resumed bundle can therefore be internally heterogeneous
  while appearing to follow one manifest configuration. Rebuilding after the
  interrupted run and using `--resume` would silently mix benchmark commits.
- **[proposed]** Freeze and authenticate a run identity containing at least:
  benchmark binary SHA-256, runner SHA-256 or source commit, benchmark source
  commit, schema, all CLI-affecting parameters, environment overrides,
  provider identity/config, and normalized case plan. Refuse resume on any
  mismatch.

### Rejection classification

- **[source-backed]** A nonzero command is classified as an expected rejection
  when stderr contains broad text such as `variant planning failed`.
- **[derived]** This can classify an unexpected planner defect as an expected
  capability/legality exclusion.
- **[proposed]** Authenticate structured rejection categories or exact
  expected reason families per case. Every rejection in the final bundle
  still requires manual review until then.

## Selective-shape and summary policy

- **[source-backed]** The runner enumerates every declared shape family,
  predeclares scalar runtime bounds, records rejected cases, and creates stable
  case keys.
- **[derived]** This prevents silent omission at collection time.
- **[proposed]** The sanitized summary must report every planned cell as
  passed, rejected, failed, or predeclared skip; include all holdout rows and
  all weak regions. Do not summarize only cells with equal thread counts or
  good ratios without also tabulating the excluded cells and reasons.

## Findings by severity

### High — Milestone 6 acceptance blockers

1. **[measured] Resume identity is not authenticated.** A rebuilt/resumed run
   can mix binaries, commits, and measurement settings.
2. **[source-backed] The declared one-shot allocation mode is absent from the
   deep-audit matrix.**
3. **[source-backed] Prepacked-B preparation time and amortized total are not
   measured.**
4. **[measured] The reviewed full collection is incomplete after one guarded
   timing failure.**

### Medium — exclusions or fixes required before strong claims

1. **[source-backed]** Multi-thread OpenBLAS is unbound/allow-SMT while native
   workers are bound/physical-only.
2. **[source-backed]** Provider `actual_threads` authenticates configuration,
   not observed active concurrency.
3. **[source-backed]** Fixed top-level case order is not balanced against
   frequency and thermal drift.
4. **[source-backed]** Individual timing samples and measured frequency are
   absent from raw JSON.
5. **[source-backed]** Expected-rejection classification is textually broad.
6. **[source-backed]** The compute-only matrix schedules an unsupported
   AVX-512 diagnostic, and the parallel prepacked matrix schedules variants
   known not to support that mode.
7. **[source-backed]** The generic environment `cpu_affinity` is captured
   before case planning may bind the caller. Case plan diagnostics, not that
   generic field, are the authoritative post-plan placement evidence.

## Required acceptance actions

1. **[proposed]** Start a fresh full collection from one rebuilt, clean
   benchmark binary, or harden resume before reusing any file. Never combine
   the reviewed `8c606d6...` bundle with a `c5031e1...` binary.
2. **[proposed]** Add and execute a bounded allocation-included suite.
3. **[proposed]** Measure one-time B preparation and explicitly derive
   amortized repeated execution totals.
4. **[proposed]** Compare only equal-actual-thread, equivalent-placement
   records. Keep unbound multi-thread OpenBLAS separate if provider affinity
   remains unauthenticated.
5. **[proposed]** Balance complete-call process order across replicates and
   retain external per-sample durations.
6. **[proposed]** Review every rejection reason and every exclusion; publish
   the complete cell-status table and raw-bundle digest.
7. **[proposed]** Re-run this independent review against the completed,
   homogeneous bundle.

## Independent conclusion

**[derived]** The current benchmark is suitable for guarded, explicitly scoped
diagnostics and has a strong basis for a defensible contract. The first full
Milestone 6 evidence bundle is not yet fair or complete enough to accept
cross-implementation parity, precise crossover, one-shot, amortized prepack,
or multi-thread native/OpenBLAS claims.

Benchmark fairness v2 verdict: **rejected pending the high-severity acceptance
actions above**.
