# ADR 0006: MDSLC CPU benchmark contract v1

- Status: Accepted for Milestone 4 implementation
- Date: 2026-07-22
- Scope: Standalone CPU GEMM measurement and calibration

## Context

Milestones 1 and 2 proved the native compiler/runtime path and a deterministic
three-variant CPU planner. The earlier opt-in runtime benchmark was useful as a
regression probe, but it timed only the selected compute body, used fixed
iteration counts, compared against the planner's float reference
implementation, and emitted no stable machine-readable record. It could not
support fair comparisons between a library call, a packed native kernel, and a
reused/prepacked execution context.

Milestone 4 introduces external OpenBLAS and native packed variants. Their
allocation, packing, provider-thread, and workspace behavior must be visible;
otherwise a fast-looking result can describe a different workload.

## Decision

`matcore-bench` owns the versioned CPU GEMM benchmark contract. Matrix tuples
are always encoded and reported in **M, N, K** order. Its v1 JSON schema is
`matcore.benchmark.cpu.gemm`, version 1. The executable uses stable variant IDs
and a runner hook that exposes legality, selected implementation, actual thread
count, workspace, workspace alignment, packing, prepacked-B support, provider
metadata, execution, and synchronization. The hook is internal C++; it is not
part of the installed runtime C ABI.

The four distinct measurement interpretations are:

1. **End-to-end one-shot**: planning plus output/workspace allocation,
   preparation/packing, compute, and synchronization. Input generation and the
   correctness oracle remain outside the interval.
2. **Reused workspace**: planning and allocation occur before measurement;
   preparation/packing, compute, and synchronization are timed.
3. **Prepacked B**: B preparation occurs once outside measurement; repeated
   compute and synchronization are timed. A variant must explicitly advertise
   this support.
4. **Compute diagnostic**: preparation/packing occurs before measurement. This
   is labeled `exclude-packing` and is not compared with a complete BLAS call as
   an end-to-end result.

Hot-cache operations shorter than the declared timer floor are aggregated.
Each recorded sample is the aggregate duration divided by its exact repetition
count, and every aggregate must reach the floor. Cold-cache mode performs a
best-effort 64 MiB eviction before each non-aggregated sample. A cold sample
below the timer floor remains a correctness result but is explicitly rejected
for performance claims. The v1 default floor is 1 ms.

Every run records minimum, median, and nearest-rank p95 time. SGEMM throughput
uses:

```text
operations = 2 * M * N * K
GFLOP/s = operations / median_seconds / 1e9
```

The correctness implementation is independent of the selected float kernel.
It accumulates expected values in double precision. Bounded problems use a
complete element oracle. Larger problems use deterministic element samples,
an all-output finite-value scan, and an independently derived double-precision
global checksum. Error limits use a conservative floating-point accumulation
bound derived from K and the absolute products. NaN and infinity always fail.

Before allocating, checked integer arithmetic accounts for A, B, C, alignment
slack, declared workspace, declared prepacked storage, and the cold-cache
buffer. The default hard cap is 2 GiB and is user-configurable. Exceeding the
cap fails before output or benchmark storage is allocated. The 8192 square is
present only in the explicit full profile.

Raw output belongs under the ignored `benchmark_reports/` tree. Only manually
reviewed, sanitized summaries belong under `docs/performance/cpu/`.

## Declared profiles

The quick profile contains:

```text
1x1x1, 2x3x2, 16x16x16, 33x35x37,
64x7x19, 127x129x131, 128x128x128, 256x256x256
```

The standard profile contains square sizes 4 through 2048 from the Milestone 4
matrix, eight declared rectangular shapes, and five tail-sensitive shapes.
The full profile adds only 4096 and 8192 square problems. Alignment sweeps are
separate runs at 4, 16, 32, and 64 bytes so the requested minimum alignment is
never hidden in one aggregate record.

## Calibration and claims

Single-thread calibration must set and report provider thread count, process
affinity, governor, boost state, compiler, flags, build type, capability
record, provider version/configuration, workspace, cache, allocation, and
packing modes. On a heterogeneous CPU, the calibration summary must name the
logical CPU and cache group used. Automatic-planner regret is:

```text
selected_variant_time / fastest_legal_variant_time
```

Rejected timer-noise results are listed but excluded from regret aggregates.
Raw measurements do not change the planner at runtime. Any crossover committed
to the planner is a deterministic, reviewed static rule with its calibration
matrix and host limitations documented.

No result is a universal performance claim. Kernel-only diagnostics, different
thread counts, different packing treatment, cold/hot cache runs, and one-shot
versus reused execution are not interchangeable.

## Consequences

The accepted Milestone 4 implementation connects all five planner-v2
candidates to this contract: reference, tiled, compiler-vectorized AVX2/FMA,
optional OpenBLAS, and native packed AVX2/FMA. The packed runner exposes
caller-owned workspace and both transient and caller-owned prepacked-B modes;
the benchmark allocates those resources outside reused-workspace intervals.
OpenBLAS uses authenticated LP64 CBLAS and process-local thread control. The
reference, tiled, and compiler-vectorized candidates report their actual
single-thread execution even when the request permits more threads; packed v1
requires exactly one requested thread. None pretends to provide parallel
execution.

The validation-host calibration and its static planner rules are recorded in
`docs/performance/cpu/milestone-4-single-thread-calibration-2026-07-22.md`.
That evidence is deliberately host/provider-specific and does not authorize a
general OpenBLAS-parity claim.

The benchmark does not define runtime autotuning and does not authorize GPU,
AVX-512, BF16, INT8, AMX, topology, NUMA, or parallel-native support by itself.
