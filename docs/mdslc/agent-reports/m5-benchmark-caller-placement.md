# Milestone 5 benchmark caller-placement hardening

## Scope and finding

This lane owns the benchmark-only caller placement in
`compiler/tools/matcore-bench/planner_runner.cpp`, focused benchmark contract
assertions, and this report. It does not change the public runtime, planner,
kernel, or C ABI.

With a process mask of CPUs 0-11 and a two-worker compact context pinned to
CPUs 0 and 1, the submitting benchmark thread previously remained eligible
for every CPU in the process mask. A scheduler collision with a pinned worker
produced a measured AVX-512 block near one millisecond while adjacent repeats
were approximately 0.14-0.15 milliseconds. That interval was scheduler noise,
not a kernel cost.

## Resolution

For a bound-worker benchmark plan, the runner now:

1. Computes the worker placement from the process-restricted topology.
2. Selects a deterministic spare logical CPU outside the worker set. It
   prefers the highest-numbered logical CPU on an otherwise unused physical
   core in the selected NUMA node, preserving low-numbered compact workers.
3. Applies the one-CPU scheduler mask with the existing platform affinity API.
4. Removes the caller's complete physical core from subsequent worker
   placement when a dedicated core was available; a logical-only fallback is
   explicitly labeled when it was not.
5. Reauthenticates the caller affinity for subsequent bound plans and fails
   closed if an explicitly selected reservation cannot be applied.

If every allowed logical CPU is already a worker, execution remains legal but
the diagnostics explicitly state that caller isolation is unavailable. The
runner never claims isolation in that case. A true unbound
`--allow-smt --affinity none` plan does not request or apply caller affinity,
so provider-managed OpenBLAS execution remains unbound.

The selected caller CPU, reserved CPU set, dedicated-core status, NUMA node,
and application status appear in both per-plan affinity diagnostics and the
runner environment's affinity/topology diagnostics. No NUMA memory-placement
claim is added.

## Verification

Focused Release build and tests:

```sh
cmake --build /tmp/matcore-m5-final-release \
  --target matcore_benchmark_core_test matcore-bench -- -j2
ctest --test-dir /tmp/matcore-m5-final-release \
  -R '^benchmark\.cpu\.(contract|cli_json)$' --output-on-failure -j1
```

Result: **2/2 passed**.

Focused ASan/UBSan build and tests:

```sh
cmake --build /tmp/matcore-m5-final-asan \
  --target matcore_benchmark_core_test matcore-bench -- -j2
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ctest --test-dir /tmp/matcore-m5-final-asan \
    -R '^benchmark\.cpu\.(contract|cli_json)$' --output-on-failure -j1
```

Result: **2/2 passed**, with no sanitizer finding.

A real balanced-regret run used:

```sh
taskset -c 0-11 /tmp/matcore-m5-final-release/bin/matcore-bench \
  --m 16 --n 16 --k 16 --threads 2 --variant auto \
  --warmup 5 --iterations 9 --hot-cache --include-packing \
  --reuse-workspace --alignment 64 --affinity compact \
  --planner-regret --timer-floor-us 1 --json-out <external-path>
```

It reported workers `[0,1]`, caller CPU `11`,
`dedicated_physical_core=true`, valid correctness, and regret `1.0`. Legal
complete-call forward/reverse candidate medians ranged from 0.00554 to 0.00716
milliseconds; the approximately one-millisecond collision did not recur.

Five independent 256x256x256 runs with the same two-worker placement,
three warmups, and seven measured iterations all reported workers `[0,1]` and
caller CPU `11`. Regret values were:

- 1.0480
- 1.1721
- 1.0000
- 1.2329
- 1.2556

The maximum was **1.2556**, below the requested catastrophic-regret guard of
2.0. AVX-512 forward/reverse pass medians ranged from 0.16918 to 0.367369
milliseconds; no pass approached the prior scheduler-quantum outlier.

A one-CPU `taskset -c 0` run reported
`caller_scheduler_affinity_applied=false` and `no spare logical CPU`. A
separate two-thread unbound OpenBLAS run retained
`worker_affinity_applied=false`, used two provider threads, and reported that
caller affinity was not requested.

Raw JSON remains outside Git under
`/home/hamza-usta/archives/MatcoreDSL-M5-perf-20260722/caller-affinity-lane/`.
`git diff --check` passed. Focused assertions were integrated in the adjacent
benchmark contract commit because the shared integration worktree committed
them while the planner implementation was still under validation; no history
was rewritten.
