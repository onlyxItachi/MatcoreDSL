# Milestone 5 benchmark/planner integration

## Scope and ownership

This lane started from `f964f6c5fbba8ec1ec5fdabe5d9d5ac432a7810e` and
changed only `compiler/tools/matcore-bench/`, `compiler/tests/benchmark/`, and
this report. It reconciles benchmark schema v2 with planner v3's placement
evidence and the strict persistent-worker affinity runtime.

## Implemented behavior

- The benchmark obtains the real scheduler mask through the platform affinity
  backend and projects the discovered topology through
  `restrict_cpu_topology_v1`. A count-only override is no longer passed to the
  planner.
- An unbound context supplies complete placement evidence only when the
  process-restricted topology is complete and contains exactly one NUMA node.
  Multi-node or incomplete discovery therefore rejects parallel planning.
- `compact`, `scatter`, and `local-first` use `plan_cpu_placement_v1`, pass the
  exact selected CPU IDs to a persistent context, and require complete worker
  affinity application. Cross-NUMA placement is refused.
- Worker affinity is scheduler affinity only. Diagnostics explicitly record
  `numa_memory_placement=false`; no allocation, binding, migration, interleave,
  or physical NUMA-performance claim is made.
- Explicit affinity is accepted only for native parallel variants. Scalar and
  external-provider selections fail rather than attributing an idle native
  pool's affinity to code that does not execute on that pool.
- Planner calls receive placement evidence directly, per-plan state retains
  the exact persistent context used for execution, and final environment
  capture reports post-run worker/submission counters.
- The process-local AVX2/FMA and AVX-512/FMA numerical self-tests remain the
  source of runtime-validation evidence.

## Validation evidence

Release configure/build used Clang 21.1.8, Ninja, the Clang/LLVM 21 CMake
packages, native frontend enabled, and OpenBLAS 0.3.32. Focused results:

```text
benchmark.cpu.contract  passed
benchmark.cpu.cli_json  passed
2/2 tests passed
```

The CLI test exercised all eight forced stable IDs. On the validation host all
eight executed successfully and passed the independent double-precision
oracle, including both persistent parallel variants. It also executed strict
two-worker `compact`, `scatter`, and `local-first` cases, checked that the
reported CPU IDs were inside the inherited mask, and confirmed post-run
submission counters.

A real `taskset -c 0` child reported one logical processor, one physical core,
and one available processor from the restricted topology. Reference GEMM
passed; forced two-thread parallel AVX2 failed before execution with the
actionable reason that at least two output macro-tiles and workers are needed.

The focused ASan/UBSan build used:

```text
-fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
```

Both focused tests passed again (`2/2`) with no sanitizer diagnostic.

## Limitations

- The local host exposes one physical NUMA node. Cross-node behavior is
  deliberately rejected here; this lane does not claim physical multi-node
  validation or NUMA memory placement.
- Explicit affinity does not bind OpenBLAS provider threads or the scalar
  caller. Such combinations are rejected instead of silently overstating
  placement.
- Performance values from the one-iteration forced-variant smoke commands are
  correctness evidence only and are not calibration claims.
