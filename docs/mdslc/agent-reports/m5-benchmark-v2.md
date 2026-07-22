# Milestone 5 benchmark-v2 lane report

## Scope and ownership

- Branch: `mdslc/m5-benchmark-v2`
- Base: `59f1869703eea6302ac0515b0ff7aca799b758aa`
- Implementation commit: `0cc4ff2`
- Owned paths: `compiler/tools/matcore-bench/**`,
  `compiler/tests/benchmark/**`, and this report.
- No runtime/public-header/planner/global-CMake source was modified.

## Implemented contract

`matcore-bench` now consumes CPU planner v3 and exposes the complete stable
eight-variant registry:

1. `cpu.reference.f32.v1`
2. `cpu.tiled.f32.v1`
3. `cpu.compiler-vectorized.avx2-fma.f32.v1`
4. `cpu.external.openblas.f32.v1`
5. `cpu.native-packed.avx2-fma.f32.v1`
6. `cpu.native-packed.avx512-fma.f32.v1`
7. `cpu.native-parallel.avx2-fma.f32.v1`
8. `cpu.native-parallel.avx512-fma.f32.v1`

The benchmark runner owns one persistent execution context for its lifetime.
Parallel measurements include shared B packing, per-worker A packing, worker
submission, compute, and synchronization. Context construction remains outside
the timed region and is reported explicitly. Post-run environment metadata
records workers started and completed context submissions, so reuse does not
appear permanently as zero.

The version-2 JSON schema adds:

- capability-v2 and topology-v1 diagnostics;
- the source and scope of runtime-validation evidence;
- planner version and all eight candidate diagnostics;
- total, shared, and per-worker workspace;
- requested versus actual threads;
- persistent-context state and submission counters;
- explicit SMT and affinity policy metadata;
- optional same-family one-thread speedup and parallel efficiency;
- optional per-candidate timings and deterministic planner regret.

The old v1 schema remains present. `--planner-regret` is deliberately limited
to automatic planning with complete include-packing calls. The AVX2-only
`--exclude-packing` microkernel diagnostic remains non-comparable and cannot be
used for regret.

## Capability and placement honesty

An ISA becomes runtime-validated in the benchmark capability-v2 record only
after a process-local, numerically checked tiny packed GEMM succeeds. The
`runtime_usable` predicate is only a prerequisite and is not itself reported as
numerical validation. Every emitted benchmark result is independently checked
against the double-precision oracle.

On Linux, `sched_getaffinity` constrains the planner's available-processor
count and the persistent context's worker capacity. A one-CPU `taskset` run
reported one available processor and one context worker.

Per-worker binding is not implemented by the current execution-context API.
The default affinity policy is `none`, metadata says that only the inherited
process mask applies, and `compact`, `scatter`, or `local-first` requests fail
actionably. Consequently this lane does not claim pinned-worker or physical
NUMA performance. Integration needs a runtime worker-placement interface before
those policies can be accepted. Explicit `--allow-smt` is required to permit
SMT; requesting more threads no longer silently enables it.

## Validation

Release build directory and all raw JSON/stdout were under `/tmp`, outside Git.

```text
cmake -S compiler -B /tmp/matcore-bench-v2-build.yB8eQo -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-bench-v2-build.yB8eQo \
  --target matcore-bench matcore_benchmark_core_test -- -j2
ctest --test-dir /tmp/matcore-bench-v2-build.yB8eQo \
  --output-on-failure -R '^benchmark\.cpu\.' -j1
```

Result: 2/2 benchmark tests passed. The CLI test enumerated all eight IDs,
forced every legal implementation, exercised AVX2/AVX-512 single and parallel
dispatch on this host, verified split workspace and post-run submissions, and
checked actionably unavailable alternatives.

ASan/UBSan Debug build:

```text
cmake -S compiler -B /tmp/matcore-bench-v2-asan.r5Pq9N -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=address,undefined'
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
  /tmp/matcore-bench-v2-asan.r5Pq9N/bin/matcore_benchmark_core_test
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
  python3 compiler/tests/benchmark/run_benchmark_cli_tests.py \
    --bench /tmp/matcore-bench-v2-asan.r5Pq9N/bin/matcore-bench \
    --schema compiler/tests/benchmark/matcore-bench-v2.schema.json
```

Result: C++ contract PASS and CLI/JSON contract PASS with no reported ASan,
UBSan, or leak failure.

## Measurement observation

A zero-warmup, two-iteration OpenBLAS run produced a misleading first-use
regret spike. Repeating `256x128x128`, two threads, with three warmups, nine
iterations, and a 100 microsecond timer floor selected OpenBLAS at a 0.0465 ms
median and regret 1.00 for that run. This is host/run-specific evidence only.
Performance claims must use the declared warmup contract; short initialization
runs remain useful correctness smoke tests but not planner-calibration data.

## Integration notes

- The lane-local CMake target must link `MatcoreDSL::CpuPlannerV3`; no global
  CMake change is required.
- Keep both benchmark schema files installed or available to tests if schema
  consumers require historical validation.
- Integrate a worker-placement backend before enabling non-`none` affinity or
  describing speedup data as pinned-core evidence.
- Python `jsonschema` was unavailable. The schema was syntax-checked with
  `python3 -m json.tool`, while the CLI contract test validates all required
  object fields and semantic values.
