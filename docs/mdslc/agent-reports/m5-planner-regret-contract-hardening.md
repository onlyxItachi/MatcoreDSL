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
3. Every measured execution is checked by the independent double-precision
   oracle immediately after that execution and outside its timed interval.
   The report records the checked-execution count and validation scope. This
   prevents a later correct execution from overwriting evidence of an earlier
   corrupt measured execution.

Correctness verification remains outside timed intervals. To authenticate each
execution without charging oracle work to the candidate, aggregate timing now
sums individually delimited execution intervals. This adds clock-boundary
measurement overhead relative to a single clock pair around an aggregate, so
pre-v3 raw timings are not directly interchangeable with v3 calibration runs.
The timer-floor contract still applies to the summed aggregate duration.

## Adversarial coverage

The benchmark test double injects recursive-plan drift independently into the
selection reason, diagnostic, timing scope, comparability, planner version,
threads, workspace sizes/alignment, prepacked storage, packing flags,
persistent-context state, SMT/affinity policy, and worker-affinity fields. It
also injects illegal and misattributed forced selections. Every case is rejected.

A separate test corrupts the second measured execution and would restore the
correct output on the following execution. The benchmark now rejects the
intermediate corruption at its original measured iteration rather than accepting
the eventual final output.

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
must migrate the planner-regret names and account for per-execution timing
boundaries; the historical v2 schema remains available for interpreting stored
v2 reports. There is no silent dual-schema output and no change to the installed
runtime C ABI.
