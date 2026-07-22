# Milestone 5 benchmark affinity follow-up

## Finding addressed

The first placement integration correctly pinned native parallel workers, but
two benchmark claims remained incomplete:

1. `--compare-one-thread` preserved explicit affinity on the same-family
   single-packed baseline, which was rejected because serial code still ran on
   the caller.
2. `physical-cores-only` with `--affinity none` capped the planner by physical
   core count but left workers unbound, so the scheduler could still place
   active workers on SMT siblings.

## Resolution

- Affinity-aware serial variants now submit their complete timed call to worker
  zero of the same strict persistent context. This covers reference, tiled,
  compiler-vectorized, native packed AVX2, native packed AVX-512, and
  single-thread OpenBLAS.
- Parallel AVX2 and AVX-512 one-thread comparisons now execute their
  same-family packed baseline on pinned worker zero. Worker submission and
  synchronization are explicitly included in the serial timing scope.
- Multi-thread OpenBLAS is rejected whenever native worker affinity is active,
  because provider-thread affinity is not authenticated. Single-thread
  OpenBLAS is legal: its complete call executes on pinned worker zero and the
  provider adapter must report one actual thread.
- The default `physical-cores-only` policy deterministically selects and pins
  one logical CPU per physical core through `plan_cpu_placement_v1`. This is
  recorded as policy-induced affinity, not user-requested affinity.
- `--allow-smt --affinity none` retains the complete single-node unbound path.
- Benchmark schema v2 now reports both
  `worker_affinity_user_requested` and
  `worker_affinity_policy_induced` at result and aggregate-environment scope.
  Diagnostics include exact CPU IDs and continue to state
  `numa_memory_placement=false`.

## Focused validation

Release with Clang 21.1.8 and OpenBLAS 0.3.32:

```text
benchmark.cpu.contract  passed
benchmark.cpu.cli_json  passed
2/2 tests passed
```

The CLI suite executed:

- all six serial families with strict compact affinity where available;
- compact, scatter, and local-first native parallel AVX2;
- AVX2 and AVX-512 parallel `--compare-one-thread --affinity compact`, each
  with a valid same-family packed baseline on this host;
- multi-thread OpenBLAS affinity rejection;
- unbound `--allow-smt --affinity none` metadata;
- induced physical-core-only binding with `user_requested=false`;
- the existing all-eight-variant and one-CPU `taskset` checks.

ASan/UBSan used `-fsanitize=address,undefined -fno-omit-frame-pointer`, leak
detection, and halt-on-error. Both focused tests passed again (`2/2`) without
an instrumented diagnostic.

## Claim boundary

Worker affinity is scheduler placement only. No NUMA page allocation,
first-touch placement, migration, interleave, or provider-worker affinity is
claimed. The focused tests establish correctness and measurement-contract
behavior, not universal performance results.
