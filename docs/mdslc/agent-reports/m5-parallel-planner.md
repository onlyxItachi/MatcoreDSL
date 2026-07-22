# Milestone 5 persistent parallel runtime and planner lane

- Date: 2026-07-22
- Base: `e4dc0affff6c540a65435ba25c5cefa4d69cb562`
- Branch: `mdslc/m5-parallel-planner`
- Ownership: new persistent execution-context, parallel packed-GEMM, planner-v3,
  resource-projection, focused tests, and this report only

## Implemented contract

`CpuExecutionContextV1` owns a fixed worker set created once at context
construction. Submissions are serialized per context, while independent
contexts may execute concurrently. Task `t` is assigned to worker
`t % actual_threads`; per-worker result records occupy distinct 64-byte cache
lines. Shutdown is idempotent, waits for an active submission, and then joins
the fixed workers. Callback failure is reported deterministically by lowest
failed task index. The external-provider nesting policy rejects a native pool
larger than one worker.

The parallel packed engine partitions output into independent 128-row macro
tiles. Its caller-owned workspace is explicit:

```text
[one immutable packed-B image]
[alignment padding]
[worker 0 transient packed-A]
[worker 1 transient packed-A]
...
```

B is packed once before dispatch and remains inside the end-to-end execution
interval. Workers call the existing prepacked-B AVX2 or AVX-512 backend on
disjoint row bands. The complete tensor, alias, alignment, ISA, context,
nesting, and workspace contract is checked before B packing or output
mutation. No packing buffer is allocated by the engine.

Planner v3 has eight deterministic F32 candidates: the five Milestone 4
variants, packed AVX-512, parallel AVX2, and parallel AVX-512. Parallel thread
count is bounded by the explicit request, optional maximum, context capacity,
available processors, physical cores unless SMT is explicitly allowed, and
the number of output macro tiles. A parallel candidate needs at least two
workers/tiles and at least 1,048,576 multiply-accumulate units per selected
worker. Shared and per-worker workspace bytes are reported separately and
combined with overflow checks.

The planner consumes injected versioned capability-v2 and topology-v1 records
through explicit loss-checked projections. AVX-512 legality retains distinct
hardware, OS-state, compiler, implementation, and runtime-validation facts.
Optimistic implementation resources cannot override a missing capability-v2
runtime-validation fact. Planner/resource discovery performs no subprocess or
hardware probing in its hot path.

## Focused commits

Lane-native commits, in dependency order:

1. `58322e5` — persistent CPU execution context
2. `9fac9a7` — deterministic parallel packed AVX2 row bands
3. `e03b1d0` — one shared packed-B image plus private A slices
4. `f48857e` — deterministic planner v3 and synthetic policy tests
5. `901cca2` — exact shared packed-B workspace test
6. `2ff36a7` — capability-v2/topology-v1 planner projections
7. `de58ac9` — persistent parallel AVX-512 execution
8. `051e8b1` — planner-v3 runtime-resource projection

The lane temporarily cherry-picked dependency commits `4213a95`, `f11f7b3`,
and `90068d7` only for compilation. The integration owner must not cherry-pick
those duplicates from this lane.

## Validation evidence

All compilations used Clang 21.1.8, C++20, strict
`-Wall -Wextra -Wpedantic -Werror`, and portable C++ threads/synchronization.

- Release focused binaries: 4/4 passed.
- Debug (`-O2 -g`, matching the supported vector-debug policy): 4/4 passed.
- ASan+UBSan: 4/4 passed with leak detection and halt-on-error.
- TSan: execution-context and parallel AVX2/AVX-512 tests 2/2 passed.
- Repetition stress: context plus parallel GEMM, 40/40 iterations passed.
- Persistent workers: 51 repeated generic submissions plus repeated GEMMs did
  not increase `workers_started`.
- Correctness: AVX2 `385x67x65` and AVX-512 `257x35x33`, including M/N/K
  tails, matched independent double-precision oracles on the physical host.
- Preflight rejection: insufficient workspace, request above context ceiling,
  external-provider nesting, and output alias submitted zero worker jobs and
  did not mutate the sentinel output where applicable.
- Synthetic planner tests covered physical-core versus SMT ceilings, explicit
  maximums, provider thread ceilings, tiny/small-work rejection, incomplete
  topology, unavailable contexts, malformed policies, AVX-512 fail-closed
  behavior, deterministic diagnostics, and affinity-limited availability.
- `git diff --check` and `git fsck --full --strict` passed; worktree was clean
  before this report.

Manual focused binaries were placed under `/tmp`, never the repository.

## Local scaling diagnostic

This is a narrow, non-universal diagnostic, not a committed benchmark database.
The production parallel path was pinned with `taskset -c 0-3`; B packing and
all compute were timed, allocation was outside the interval, median of five
measured iterations after one warmup:

| Variant | Shape | 1 thread | 4 threads | Speedup | 4-thread GFLOP/s |
|---|---:|---:|---:|---:|---:|
| packed AVX2 | 2048 cubed | 0.133980 s | 0.041489 s | 3.229x | 414.081 |
| packed AVX-512 | 2048 cubed | 0.129629 s | 0.041180 s | 3.148x | 417.195 |

The 1024-cubed AVX2 diagnostic reached only 2.123x because serial packing and
coordination were a larger fraction. That smaller result is intentionally
retained as crossover evidence; it is not hidden or generalized. The project
benchmark harness and full planner-regret calibration remain integration-owner
acceptance work.

## Limitations and integration notes

- The executor does not apply OS affinity itself. The versioned topology and
  placement layer selects CPU IDs; platform-specific affinity application must
  remain an isolated integration backend.
- Row-band v1 parallelism requires at least two MC-row output tiles. Skinny-M
  problems remain single-threaded even when N/K are large.
- Shared B is repacked once per complete call. A future explicit persistent
  prepacked-weight context can amortize it, but no hidden cache was added.
- The execution context is internal C++; no STL type crosses the stable C ABI.
  Public additive context handles and ABI wiring belong to integration-owned
  `runtime_c.h` and `cpu_runtime.cpp`.
- CMake registration is intentionally left to the integration owner because
  this lane was prohibited from editing shared CMake files.
- No physical multi-node NUMA, AMX, Windows runtime, or GPU claim is made.
