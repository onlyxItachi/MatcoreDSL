# Milestone 5 planner-regret contract hardening

Date: 2026-07-22

Ownership was limited to `matcore-bench`'s benchmark contract, JSON schema,
focused benchmark tests, and this report. Planner costs and selection rules were
not changed.

## Closed review findings

1. Benchmark JSON now emits schema v3. Each comparable candidate reports its
   forward-pass median, reverse-pass median, and separately named balanced
   estimate. The typed aggregation method is
   `arithmetic-mean-of-forward-and-reverse-pass-medians`. No field named
   `median` contains that derived arithmetic mean. The v2 schema remains
   unchanged as a historical contract.
2. Each recursive forced-candidate measurement is authenticated against its
   preflight semantic plan before its timing can contribute to regret. The
   binding covers the requested and selected variant, legal/comparable state,
   planner version, timing scope, thread count, all workspace and packing
   metadata, persistent-context state, SMT/affinity policies, and all worker
   affinity flags and diagnostics. Drift fails closed with the pass, variant,
   and mismatched field in the diagnostic. Those authenticated preflight fields
   are emitted per candidate so provider/native comparability can be audited
   without borrowing metadata from the automatic plan.
3. Every pass has a separate untimed phase that executes the same number of
   invocations as the timed phase and checks every invocation with the
   independent double-precision oracle. Forward candidate passes place replay
   after timing; reverse candidate passes place replay before timing. This
   mirrored placement prevents candidate-dependent replay cost from appearing
   on only one side of the forward/reverse schedule. The final timed output is
   always authenticated immediately after timing. Normal primary benchmarks
   retain after-timing placement.

Correctness verification remains outside timed intervals. Each timed sample
uses exactly one clock pair around its full aggregate repetition block; no
oracle or cache-scanning work occurs between timed repetitions. Schema v3 emits
that boundary as `one-clock-pair-per-aggregate-repetition-block`. The timer-floor
contract applies to that aggregate duration.

## Adversarial coverage

The benchmark test double injects recursive-plan drift independently into the
selection reason, diagnostic, timing scope, comparability, planner version,
threads, workspace sizes/alignment, prepacked storage, packing flags,
persistent-context state, SMT/affinity policy, and worker-affinity fields. It
also injects illegal and misattributed forced selections. Every case is rejected.

Separate tests corrupt the second invocation of forward/after and
reverse/before untimed validation. A following invocation would restore the
correct output, but both placements reject the intermediate corruption. The
reverse diagnostic proves that its replay ran before timing.

## Validation

Release focused build and tests:

```text
cmake --build /tmp/matcore-m5-final-release \
  --target matcore_benchmark_core_test matcore-bench -- -j2
ctest --test-dir /tmp/matcore-m5-final-release --output-on-failure \
  -R '^benchmark\.cpu\.(contract|cli_json)$' -j1
2/2 passed
```

ASan/UBSan focused build and tests:

```text
cmake --build /tmp/matcore-m5-final-asan \
  --target matcore_benchmark_core_test matcore-bench -- -j2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir /tmp/matcore-m5-final-asan --output-on-failure \
  -R '^benchmark\.cpu\.(contract|cli_json)$' -j1
2/2 passed
```

The v3 schema parses as JSON, the CLI test consumes its required-field contract,
and an emitted planner-regret report contains only the new unambiguous estimate
names. The optional Python `jsonschema` package was not installed on this host,
so no claim is made for validation through that third-party implementation.

## Compatibility

`matcore-bench` now emits report version 3 by default. Consumers expecting v2
must migrate the planner-regret and untimed-validation names; the historical v2
schema remains available for interpreting stored v2 reports. The aggregate
timing boundary retains the pre-fairness one-clock-pair semantics. There is no
silent dual-schema output and no change to the installed runtime C ABI.
