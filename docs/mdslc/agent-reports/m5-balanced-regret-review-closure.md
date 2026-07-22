# Milestone 5 balanced-regret review closure pass

## Scope and baseline

This independent pass reviewed integration commit
`24bd5afcb319b6b6c89e258724b0fea16d6101db` against the three medium findings
in `m5-balanced-regret-independent-review.md`. The reviewed fixes are:

- `d9bf6ee` (`fix(bench): authenticate planner regret contract`);
- `24bd5af` (`fix(bench): isolate correctness validation from timing`).

No production file was edited in this lane.

## Verdict

**The original M1, M2, and M3 findings are closed. One new medium timing-order
finding remains; final planner-regret calibration is not yet accepted.**

There are no high findings. Schema naming and versioning, forced-candidate plan
authentication, and repeated correctness validation are now materially stronger
and fail closed. The new correctness phase is outside every reported clock
pair. Its current placement after each candidate's timed block, however,
reintroduces candidate-dependent order bias because validation cardinality and
cost vary by candidate.

## Original finding closure

### M1 — balanced estimator mislabeled as a median: closed

- Benchmark JSON now emits schema version 3.
- `forward_pass_median_seconds` and `reverse_pass_median_seconds` retain their
  literal statistical meaning.
- Their arithmetic midpoint is emitted separately as
  `balanced_estimate_seconds`.
- The enclosing fields are accurately named
  `selected_balanced_estimate_seconds` and
  `fastest_legal_balanced_estimate_seconds`.
- The typed aggregation method is
  `arithmetic-mean-of-forward-and-reverse-pass-medians`.
- Neither production v3 output nor its schema contains the old misleading
  selected/fastest median names.
- `matcore-bench-v2.schema.json` is byte-unchanged across the fix range. Its
  reviewed SHA-256 is
  `1f7f7b34fa1a8c92958528cb00eca397513e80e372fe83b64b472ab59195e1a9`.

The ordinary top-level `median_seconds` remains the actual median of that
result's measured aggregate samples, so primary throughput and balanced regret
retain distinct meanings.

### M2 — recursive forced plan not authenticated: closed

Preflight plans are retained. Each forward and reverse forced measurement is
rejected unless it preserves:

- requested and selected stable IDs;
- legal/comparable status;
- selection reason and complete diagnostic;
- timing scope and planner version;
- actual threads;
- total/shared/per-worker workspace and alignment;
- prepacked storage and packing flags;
- persistent-context state;
- SMT and affinity policy;
- all worker-affinity flags and the affinity diagnostic.

Forced top-level planning also rejects a selected ID different from the
request. Authenticated comparison-critical fields are emitted per candidate,
and `plan_authenticated` becomes true only after both measured passes complete
the fail-closed checks.

The recording runner injects drift into every fingerprint field independently,
plus illegal and misattributed selections. All cases must fail with the
specific mismatched field or forced-selection error. This directly covers the
previously missing fallback/misattribution regression.

### M3 — intermediate corruption overwritten before final check: closed

The harness now:

1. measures each aggregate sample with one clock pair around all repetitions;
2. checks the final timed output immediately after timing;
3. executes an equal-cardinality untimed validation phase;
4. checks every validation invocation with the independent double-precision
   oracle;
5. reports the validation count, scope, and timed-final authentication.

The fake runner corrupts an intermediate validation invocation and would
restore correct output on the next invocation. The harness rejects the first
bad validation rather than accepting the later output. Oracle work does not
occur between a measured block's start and end clock reads. This satisfies the
review's requested untimed repeated-validation closure without contaminating
the reported interval.

This is a deterministic validation companion, not proof that an arbitrary
one-time hardware fault in a timed invocation must recur. The report accurately
states what was checked: final timed output plus every invocation in a separate
equal-cardinality validation phase.

## New finding

### N1 — candidate-dependent post-timing validation breaks exact order balance

Severity: **medium**

Each candidate is currently executed as a composite block:

```text
timed candidate work T_i -> untimed validation work V_i
```

Both registry passes use that same internal orientation:

```text
forward: T_0,V_0  T_1,V_1 ... T_n,V_n
reverse: T_n,V_n ... T_1,V_1  T_0,V_0
```

The forward/reverse order therefore no longer places every candidate's two
timed centers at the same mean wall-clock position when `V_i` differs. Let
`D_i = T_i + V_i` and `S = sum(D_i)` for one pass. Under a linear drift, the
mean temporal center of candidate `i` is:

```text
S - V_i / 2
```

rather than the common center `S`. Validation cost is candidate-dependent
because the number of validation executions equals
`measured_iterations * aggregate_repetitions`; faster candidates commonly use
more aggregate repetitions. Every validation execution also runs the
independent oracle, whose cost is not the candidate's timed compute cost.

This is not merely theoretical on the review host. A guarded compact-affinity
`64x64x64` probe with three measured samples and a 1 ms timer floor emitted per
pass validation counts ranging from 84 for reference to 1,212 for packed
AVX-512. Forward/reverse counts for the same candidate can also differ because
each pass independently calibrates aggregate repetitions. The complete probe
took 1.34 seconds while the reported candidate medians were measured in
microseconds, so post-block validation dominates the wall-clock environment
between candidate intervals.

Required resolution: preserve equal-cardinality validation while separating it
from candidate timing order, or mirror the internal orientation so the reverse
pass validates before timing (`T,V` forward and `V,T` reverse). The latter must
also account for warmup/probe/calibration placement so it does not introduce a
new candidate-specific preconditioning asymmetry. A global validation phase
before or after all candidate timing blocks is preferable when practical.

Add a deterministic scheduling regression that records timing-block and
validation-block events, not only plan calls. Then rerun the exact host
calibration matrix; prior balanced-regret values produced with candidate-local
post-timing validation are not final evidence.

## Fairness and evidence assessment

The following properties remain sound:

- packing and complete-call scope are inside candidate timing;
- allocation follows the explicitly requested benchmark mode;
- persistent context creation and warmups remain outside timed intervals;
- bound-worker mode rejects unauthenticated multi-thread OpenBLAS affinity;
- selected and alternative regret estimates use the same estimator;
- invalid timing, incorrect output, execution failure, or plan drift fails
  closed;
- candidate JSON order and member order remain deterministic;
- v3 explicitly exposes pass medians, plan metadata, validation cardinality,
  and estimator method.

Provider/native intervals are structurally comparable when both candidates are
legal: OpenBLAS internal packing remains inside its complete CBLAS call, native
packing remains inside native complete calls, and actual thread/affinity policy
is now emitted per candidate. N1 concerns the environment between those
intervals, not hidden work inside a measured interval.

## Verification performed

Focused Release tests at the reviewed source state:

```text
benchmark.cpu.contract  passed
benchmark.cpu.cli_json  passed
2/2 passed
```

Focused ASan/UBSan tests:

```text
benchmark.cpu.contract  passed
benchmark.cpu.cli_json  passed
2/2 passed
```

Additional checks:

- broader benchmark/planner Release selection from the prior pass: 8/8 passed;
- v3 schema parsed successfully with Python's standard JSON parser;
- an emitted real report had exactly the required/no-extra key sets for root,
  environment, configuration, result, scaling, regret, and every candidate;
- third-party Python `jsonschema` is unavailable, so no claim is made for that
  validator;
- `git diff --check` passed;
- raw methodology probes remained under `/tmp`.

## Acceptance recommendation

Accept the original three fixes as closed. Do not accept final planner-regret
or planner-calibration evidence until N1 is fixed and a final independent pass
confirms mirrored or globally separated timing/validation scheduling.
