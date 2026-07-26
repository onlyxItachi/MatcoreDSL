# GEMM parallel runtime and topology audit

Date: 2026-07-26

Scope: MDSLC F32 packed GEMM, persistent CPU execution contexts, planner-v3
thread selection, worker affinity, and the current topology/NUMA contract.

## Evidence boundary

- **[measured]** The bounded probes in this report used a clean Release build
  at source commit `951239f1bee5541a4cf5ad72fab2192de07cf89d`,
  Clang 21.1.8, and the physical AMD Ryzen AI 9 HX 370 host described below.
  Every retained `matcore-bench` result passed the benchmark's double-precision
  oracle and timing guard.
- **[measured]** Raw JSON, stdout, probe source, and probe binaries are outside
  Git under `/home/hamza-usta/.tmp/mdslc-m6-parallel-audit/`. This document is
  the reviewed summary; the raw directory is not a durable project artifact.
- **[measured]** The desktop was not an isolated performance laboratory during
  these bounded probes: the observed load average was approximately 3--4,
  frequency was dynamic, and unrelated interactive processes were active.
  Absolute timings below are diagnostics, not planner-calibration inputs.
- **[source-backed]** Source conclusions refer to
  `cpu_execution_context.cpp`, `cpu_parallel_gemm.cpp`,
  `cpu_planner_v3.cpp`, `cpu_runtime.cpp`, and `cpu_topology_v1.cpp` at the
  commit above.
- **[source-backed]** The 4096-cube scaling figures cited below are the retained
  guarded measurements already reviewed in
  `docs/performance/cpu/milestone-5-advanced-cpu-2026-07-22.md`; that report
  records its own measured source commit and limitations.

**[source-backed]** Principal source anchors at the audited commit:

| Contract | Source anchor |
| --- | --- |
| worker epoch, active-worker participation, cyclic assignment, completion | `compiler/lib/runtime/cpu_execution_context.cpp:77-144` |
| submission validation, publish/wake/wait path | `compiler/lib/runtime/cpu_execution_context.cpp:308-370` |
| row-band task and per-worker workspace selection | `compiler/lib/runtime/cpu_parallel_gemm.cpp:135-167` |
| shared-B/per-worker-A workspace layout | `compiler/lib/runtime/cpu_parallel_gemm.cpp:169-218` |
| serial B pack followed by worker dispatch | `compiler/lib/runtime/cpu_parallel_gemm.cpp:226-306` |
| thread ceiling, parallel cost, and legality threshold | `compiler/lib/planner/cpu_planner_v3.cpp:264-376,594-864` |
| context placement and fail-closed NUMA policy | `compiler/lib/runtime/cpu_runtime.cpp:1354-1453` |
| compact/scatter/SMT candidate ordering | `compiler/lib/platform/cpu_topology_v1.cpp:166-220,692-789` |

## Executive findings

1. **[source-backed]** The worker pool is genuinely persistent, deterministic,
   allocation-free per submission, and fail-closed. A context creates its
   fixed workers once, serializes submissions, and joins them only at shutdown.
2. **[source-backed]** Parallel GEMM decomposes only the `M` dimension into
   fixed 128-row bands. `N` and `K` never add parallel tasks. This makes
   `M <= 128` inherently single-worker even when `N*K` is enormous.
3. **[measured]** A forced four-thread native AVX2 request for
   `64x1024x1024` and `64x4096x4096` was rejected because each problem has
   only one 128-row macro tile. The rejection was actionable and did not
   silently fall back.
4. **[derived]** Tail-band imbalance is severe immediately after a 128-row
   boundary: `M=129` creates row loads `[128,1]`, for only 50.4% ideal static
   utilization before any dispatch or packing cost.
5. **[measured]** In a balanced forward/reverse candidate sweep at
   `129x512x512`, automatic planning selected two-thread parallel AVX-512 at
   0.635 ms while packed AVX-512 was fastest at 0.553 ms; regret was 1.148.
   The parallel AVX2 candidate was also slower than packed AVX2
   (0.644 ms versus 0.559 ms).
6. **[source-backed]** Every complete parallel call serially packs all of `B`
   on the submitting thread before worker dispatch. Workers share that packed
   image read-only and have isolated transient-`A` workspaces.
7. **[source-backed]** Packed `B` is safely shared within one call, but the
   current parallel API always repacks it on every submission. It cannot
   consume the existing persistent prepacked-`B` contract.
8. **[measured]** Empty persistent-pool submission cost was about
   1.94--2.73 us for an exact one-worker context, but 10.57--12.41 us when a
   12-worker context executed only one active task. All 12 workers observe
   every epoch because submission uses `notify_all`.
9. **[source-backed]** Static cyclic task assignment has no stealing and no
   performance weighting. This is deterministic and synchronization-light,
   but it cannot compensate for tail bands or this host's heterogeneous
   5.16 GHz and 3.29 GHz maximum-frequency groups.
10. **[measured]** With compact affinity, an explicit two-thread
    `allow-smt` plan selected sibling CPUs `[0,12]`. A bounded
    `1024x1024x1024` AVX2 run delivered 138.4 GFLOP/s versus 242.8 GFLOP/s
    with physical-core CPUs, 57.0% of the physical-core throughput.
11. **[source-backed]** On this one-socket, one-node host, `compact` and
    `scatter` both selected `[0,1,2,3]` for four physical workers and
    `[0..7]` for eight. Scatter is socket/NUMA-bucket aware, not LLC-group or
    core-class aware.
12. **[source-backed]** The public runtime currently forbids cross-NUMA worker
    placement and performs no page binding, migration, or interleaving.
    Multi-node policy is therefore fail-closed/synthetic, not physically
    validated.

## Current execution model

### Persistent context

- **[source-backed]** `CpuExecutionContextV1::create` allocates a fixed
  `std::thread` vector, waits for every worker to initialize, authenticates
  strict requested affinity, and tears the whole context down if any worker
  cannot bind.
- **[source-backed]** Repeated submissions do not recreate workers.
  `workers_started` and `completed_submissions` are monotonic diagnostics, and
  the runtime tests assert 50 repeated submissions with an unchanged worker
  count.
- **[source-backed]** One `submission_mutex` is held for a whole submission.
  Concurrent calls on the same context are intentionally serialized; separate
  contexts can execute concurrently.
- **[source-backed]** Reentrant submission from one of the context's workers
  fails immediately. Provider/native nesting with more than one native worker
  is also rejected before execution.
- **[source-backed]** Workers sleep on a condition variable. A submitter
  publishes a borrowed stack `SubmissionV1`, increments an epoch, calls
  `notify_all`, and waits until every active worker increments the common
  completion count.
- **[source-backed]** Inactive workers still wake and take `state_mutex` to
  observe the new epoch. They do not retain the borrowed submission pointer,
  so the lifetime is safe, but a large context imposes wake cost on a
  one-worker plan.

### GEMM task graph

- **[source-backed]** The task count is exactly
  `ceil(M / MC)` with `MC=128`; actual threads are
  `min(requested_threads, task_count)`.
- **[source-backed]** Task `t` owns rows
  `[128*t, min(128*(t+1), M))` and all columns. Worker `w` executes tasks
  `w, w+p, w+2p, ...`, where `p` is the actual thread count.
- **[source-backed]** No `N`-tile or `K`-tile is independently schedulable.
  There is no dynamic queue, stealing, task splitting, or performance-weighted
  assignment.
- **[source-backed]** Output ownership is race-free: every task writes a
  disjoint row band of `C`; `K` is not split, so no cross-worker reduction is
  needed.
- **[source-backed]** The only worker barrier is at complete-call completion.
  There is no barrier between `NC` or `KC` panels.
- **[hypothesis]** The absence of panel barriers avoids frequent
  synchronization, but workers may drift onto different packed-`B` panels.
  When the packed image exceeds shared cache, loss of phase alignment may
  reduce shared-cache reuse. Hardware-counter evidence is required before
  promoting this to a measured cause.

## Workspace, packing, and ownership

- **[source-backed]** Workspace is one caller-owned 64-byte-aligned arena:
  a full immutable packed-`B` image followed by one rounded, non-overlapping
  transient-`A` slice per active worker.
- **[source-backed]** Per-worker strides are rounded to 64 bytes, and
  `WorkerResultV1` is cache-line aligned and padded. These choices prevent
  false sharing in status and workspace metadata.
- **[source-backed]** The submitter calls `prepare_b` before `run_tasks`.
  Packing is serial, occurs inside the complete-call interval, and first
  touches packed-`B` pages on the submitting thread.
- **[source-backed]** The packed view authenticates source address, dimensions,
  `KC=256`, `NC=256`, `NR=16`, storage, and provenance. Worker callbacks only
  read the packed image.
- **[source-backed]** Each worker traverses every `NC` and `KC` panel for its
  row band. Its transient `A` block is repacked for every `NC` panel, so wide
  problems repeat `A` packing even though `A` values are unchanged.
- **[source-backed]** The context itself owns no matrix workspace. Allocation
  and first touch remain explicit caller responsibilities.
- **[source-backed]** Parallel reports expose total workspace, shared packed-B
  bytes, per-worker bytes, actual threads, macro-tile count, and context
  submission number.
- **[source-backed]** The parallel entry point accepts raw `rhs` and always
  prepares a new packed view. The single-thread packed backend has an explicit
  prepacked-`B` execution path, but the parallel backend does not.
- **[proposed]** Add a context-borrowed, caller-owned parallel prepacked-`B`
  path that reuses the existing identity/provenance rules. It must retain
  explicit lifetime, invalidation, ISA/layout identity, size bounds, and
  thread-safe read-only sharing; it must not become a global mutable cache.

## Load balance

### Deterministic row-band utilization

**[derived]** The table below assumes equal-speed workers and equal cost per
output row. `ideal utilization = M / (workers * max worker rows)`. It is an
upper bound from task granularity, not a measured hardware efficiency.

| M | 128-row tasks | Active workers | Static rows per worker | Ideal utilization |
| ---: | ---: | ---: | --- | ---: |
| 129 | 2 | 2 | `[128,1]` | 0.504 |
| 192 | 2 | 2 | `[128,64]` | 0.750 |
| 257 | 3 | 3 | `[128,128,1]` | 0.669 |
| 4096 | 32 | 4 | `[1024,1024,1024,1024]` | 1.000 |
| 4096 | 32 | 6 | `[768,768,640,640,640,640]` | 0.889 |
| 4096 | 32 | 12 | eight `384`, four `256` | 0.889 |
| 4096 | 32 | 24 | eight `256`, sixteen `128` | 0.667 |

- **[derived]** A work-per-thread legality threshold cannot correct these
  waves. `129x512x512` has ample arithmetic work by the current threshold, yet
  its second task contains only one row.
- **[source-backed]** Planner-v3 computes `actual_threads` from macro-tile
  count, but its cost divides total work evenly by that count. It does not
  model `ceil(tasks/threads)`, tail-band rows, or heterogeneous worker speed.
- **[measured]** The current clean-binary `129x512x512` regret probe confirms
  this boundary matters: packed AVX-512 was fastest while automatic planning
  selected the two-thread parallel AVX-512 candidate.
- **[proposed]** Cost parallel work by predicted makespan:
  explicit per-band row counts assigned to concrete worker classes, plus
  serial packing and measured dispatch. Avoid activating a worker whose tail
  band does not reduce predicted makespan.

### Rectangular decomposition

- **[measured]** For equal 67.1-million-MAC proxy problems in balanced
  forward/reverse sweeps, four-thread parallel AVX2 took 0.331 ms for
  `1024x64x1024` versus 1.094 ms packed single-thread (3.31x), and 0.310 ms for
  `1024x1024x64` versus 0.992 ms (3.20x).
- **[measured]** `64x1024x1024` exposed only one row task; its parallel
  variant was illegal, while packed AVX-512 completed in a balanced 1.146 ms.
- **[derived]** The current decomposition is usable for tall problems because
  large `M` supplies bands. It cannot exploit short-wide problems, even when
  total work matches a well-scaling tall problem.
- **[proposed]** Add an `N`-band decomposition for small-`M`/large-`N`
  problems, and select between `M`, `N`, and two-dimensional output tiling by
  shape. Keep `K` unsplit initially to avoid reduction buffers and altered
  floating-point order.
- **[proposed]** If `N` is split, use cache-line-aligned column boundaries and
  explicit tail ownership. A row-major column split can otherwise create
  adjacent-worker false sharing within every output row.

## Dispatch and synchronization cost

### Probe method

- **[measured]** A temporary Release probe compiled the exact execution-context
  sources with `clang++-21 -O3 -DNDEBUG -pthread`, pinned workers to physical
  CPUs 0--11, performed 200 warmups, then timed 5,000 submissions per point
  with `std::chrono::steady_clock`.
- **[measured]** Each task updated a distinct 64-byte-padded worker slot, so the
  callback was observable without introducing inter-worker atomics.
- **[measured]** Three independent process runs produced these median ranges:

| Context workers | Active workers | Tasks | Median submission range |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 1.94--2.73 us |
| 2 | 2 | 2 | 3.02--3.75 us |
| 4 | 4 | 4 | 3.45--3.95 us |
| 6 | 6 | 6 | 6.49--9.50 us |
| 12 | 12 | 12 | 13.75--16.03 us |
| 12 | 1 | 1 | 10.57--12.41 us |
| 12 | 2 | 2 | 11.00--12.18 us |
| 12 | 4 | 4 | 11.60--13.38 us |

- **[measured]** Overprovisioning therefore increased the one-active-worker
  median by roughly 4--6x relative to an exact one-worker context in this
  probe.
- **[source-backed]** The cause is structurally plausible: `notify_all` wakes
  every context worker, and every worker serially inspects the shared epoch
  under `state_mutex`, even when only the first worker participates.
- **[source-backed]** Every active worker also takes `state_mutex` once at
  completion to increment one shared counter; the last active worker wakes the
  submitter.
- **[measured]** Previously retained tiny calls through a bound four-worker
  context took 5.90 us for `1x1x1` and 5.22 us for `2x3x2`, dominated by call
  and pool submission rather than arithmetic.
- **[proposed]** Preserve persistent workers but add targeted wake-up or a
  generation mechanism that does not make inactive workers contend on every
  submission. Measure futex and condition-variable behavior before selecting
  an implementation.
- **[proposed]** Permit an explicitly documented caller-thread execution path
  for one-thread plans when affinity semantics allow it. A bound-context plan
  must not silently migrate work away from its selected worker policy.

## Affinity, SMT, and heterogeneous cores

- **[measured]** Host topology is one socket, one NUMA node, 12 physical cores,
  and 24 logical CPUs. CPUs 0--3 report a 5,157,895 kHz maximum and share a
  16 MiB L3; CPUs 4--11 report about 3,289,474 kHz and share an 8 MiB L3.
- **[source-backed]** `CpuTopologyV1` records logical/core/socket/node identity
  and cache-sharing groups, but records no core performance class or maximum
  frequency.
- **[source-backed]** Physical compact placement sorts by node, socket, core
  ID, thread index, then logical CPU. This host therefore fills CPUs 0--3
  before CPUs 4--11.
- **[source-backed]** Equal row-band assignment is not weighted for the two
  observed performance groups. The slowest assigned worker determines the
  completion barrier.
- **[measured]** Retained 4096-cube AVX2 scaling improved from 482.8 GFLOP/s at
  four threads to only 524.0 at six and 590.1 at twelve. This is consistent
  with, but does not by itself prove, the combined effects of slower added
  cores, serial packing, cache pressure, and static imbalance.
- **[source-backed]** On a one-node/one-socket topology, the current scatter
  routine has one bucket, so its result is identical to compact. It does not
  scatter across the two LLC groups.
- **[source-backed]** With `allow_smt`, compact order places both siblings of a
  core before moving to the next core. The current plan tool selected
  `[0,12]` for two workers.
- **[measured]** In sequential bounded 21-sample AVX2 runs at
  `1024x1024x1024`, physical-core compact placement measured 8.844 ms
  (242.8 GFLOP/s) and sibling compact placement measured 15.517 ms
  (138.4 GFLOP/s).
- **[hypothesis]** That 43% throughput loss is primarily sibling contention,
  but process-order and dynamic-frequency effects were not neutralized.
  Replicate with alternating order and frequency telemetry before using the
  ratio as calibration.
- **[proposed]** Replace the exposed binary `allow SMT` scheduling behavior
  with “physical cores first, then siblings” unless the user explicitly
  requests sibling-first placement. Reuse the already modeled
  `prefer_physical_cores` ordering.
- **[proposed]** Make placement cache-group and core-class aware. Preserve a
  portable default when a platform cannot authenticate these properties.

## NUMA behavior

- **[measured]** The validation host has one physical NUMA node. No real
  cross-node timing is available.
- **[source-backed]** Context creation always sets `allow_cross_numa=false`.
  A requested worker count that cannot fit the selected node is rejected.
- **[source-backed]** `local-first` authenticates the creator CPU, selects its
  NUMA node, and pins workers within that node. It does not bind or migrate
  tensor/workspace pages.
- **[source-backed]** Reports explicitly set
  `numa_memory_placement_applied=false`. Caller first-touch remains a visible
  requirement, not a runtime-enforced placement operation.
- **[source-backed]** Planner-v3 has a deterministic cross-node cost penalty
  and synthetic placement evidence, but the current public context cannot
  execute that path.
- **[proposed]** Keep multi-node execution disabled until allocation/first-touch
  ownership, shared packed-B placement, per-worker A placement, and output
  ownership are executable and physically validated. Synthetic planner tests
  must remain labeled synthetic.

## Current crossover policy

- **[source-backed]** A parallel candidate requires at least two 128-row bands
  and at least `2^20` multiply-accumulates per active thread.
- **[source-backed]** Its static cost is approximately
  `(M*N*K*isa_factor)/threads + 6*M*K + 6*K*N + 200000 + 20000*threads`;
  AVX-512 uses factor 1 and AVX2 factor 2.
- **[source-backed]** The model includes serial-equivalent `A` and `B` packing
  terms and a fixed thread overhead, but not row-wave makespan, tail-band
  utilization, inactive-worker wakeups, heterogeneous cores, LLC groups,
  repeated/prepacked `B`, or caller-core packing speed.
- **[measured]** Retained Milestone 5 evidence found its worst compact-plan
  point at `192x192x192`: automatic two-thread AVX-512 had regret 1.363 versus
  packed AVX2.
- **[measured]** The clean-current `129x512x512` probe independently found
  another crossover weakness with regret 1.148.
- **[proposed]** Calibrate separate thresholds for:
  exact context size, active worker count, row-band utilization, aspect-ratio
  family, transient versus prepacked `B`, affinity/core class, and ISA.
  Keep the resulting rule deterministic; do not add runtime autotuning.

## Ranked causes and candidate changes

| Rank | Finding | Confidence | Expected affected region | Candidate change |
| ---: | --- | --- | --- | --- |
| 1 | **[source-backed]** M-only 128-row tasking | High | Short-wide and vector-like `M<=128`; row-boundary tails | Add `N`/2D output decomposition and shape-specific task geometry |
| 2 | **[source-backed]** Serial full-B packing every call | High | Large `K*N`, repeated weights, higher thread counts | Add explicit parallel prepacked-B execution; investigate parallel/overlapped packing |
| 3 | **[source-backed]** Cost model ignores task-wave imbalance | High | `M mod 128`, especially 129/192/257 | Predict per-worker row load and makespan |
| 4 | **[measured]** All-worker wake on partial submission | High | Tiny/small calls and oversized contexts | Targeted wake mechanism or legal caller-thread fast path |
| 5 | **[source-backed]** SMT sibling-first ordering | High | Explicit compact `allow-smt` at sub-core-count threads | Prefer all physical cores before siblings |
| 6 | **[source-backed]** No core-class/LLC-aware assignment | High | Hybrid core groups and >4 threads on this host | Add authenticated core classes/cache groups and weighted placement |
| 7 | **[source-backed]** Static cyclic, no stealing | Medium | Heterogeneous cores and uneven tail bands | Deterministic weighted static schedule first; bounded stealing only if determinism is preserved |
| 8 | **[hypothesis]** Panel-phase drift reduces shared-B locality | Medium | Large packed B and many workers | Panel-coordinated teams or double-buffered shared B after counter validation |
| 9 | **[source-backed]** Same-context submissions serialize | High | Independent concurrent GEMMs in one context | Retain safety now; later consider explicit streams/subcontexts, not implicit overlap |

## Required internal contract changes before performance work

- **[proposed]** A planner-visible task-geometry record: partition axis,
  macro-tile sizes, task count, per-task rows/columns, waves, and predicted
  active worker utilization.
- **[proposed]** A topology-performance projection distinct from raw discovery:
  core class, LLC group, maximum authenticated frequency class, and placement
  weights, with unknown values represented explicitly.
- **[proposed]** A versioned parallel packed-operand view reusable across
  submissions and workers, with caller ownership and exact invalidation.
- **[proposed]** A submission-cost record parameterized by context capacity and
  active workers. Static defaults remain mandatory when no host calibration is
  authorized.
- **[proposed]** A packing execution plan that states serial/parallel ownership,
  first-touch policy, panel order, and whether packing can overlap compute.
- **[proposed]** Diagnostics must expose selected partition axis, task count,
  waves, row/column utilization, selected CPU IDs/core classes, and whether
  packed `B` is transient or reused.

## Direct answers to the milestone questions

- **[source-backed]** **Is packed B shared safely?** Yes within one call:
  it is prepared before dispatch, immutable during workers, and protected by
  source/storage provenance. It is not reusable by the parallel public path
  across calls.
- **[source-backed]** **Is B repacked unnecessarily?** For repeated execution
  with unchanged `B`, yes: every parallel call repacks the full matrix because
  no parallel prepacked entry point exists.
- **[source-backed]** **Is rectangular decomposition suitable?** Only when
  `M` supplies enough 128-row bands. It is structurally unsuitable for
  short-wide/vector-like problems with small `M`.
- **[measured]** **What is persistent dispatch cost?** On this host and probe,
  roughly 2--4 us for exact 1--4-worker contexts and roughly 11--13 us when a
  12-worker context activates only 1--4 workers.
- **[source-backed]** **Where are barriers?** One wake/complete barrier per
  submission; no GEMM panel barrier. Every active worker contends once on the
  completion mutex.
- **[source-backed]** **Is false sharing controlled?** Worker result and
  workspace slices are cache-line separated. With an output base not aligned
  to 64 bytes, a row-band boundary can still bisect one cache line even though
  its relative byte offset is a multiple of 64.
- **[measured]** **When does threading cross over?** No universal threshold is
  established. Current evidence rejects `M<=128`, shows bad automatic
  selection at `M=129` and historical `M=192`, and shows good four-thread
  balanced-candidate speedups for two `M=1024` rectangular proxies.
- **[source-backed]** **Does affinity handle this host well?** It binds
  strictly and reports honestly, but scatter ignores LLC groups, SMT compact
  is sibling-first, and no core-performance weighting exists.
- **[source-backed]** **What is physically validated for NUMA?** Single-node
  placement only. Multi-node logic is synthetic/fail-closed; no page placement
  or physical multi-node performance is claimed.

## Validation commands

**[measured]** The principal bounded commands were:

```sh
# Exact planner and placement decisions.
matcore-plan --m M --n N --k K --threads T \
  --smt physical --affinity compact --numa single-node --variant auto

# Balanced forward/reverse candidate estimates.
matcore-bench --m M --n N --k K --variant auto --threads 4 \
  --warmup 5 --iterations 11 --hot-cache --include-packing \
  --reuse-workspace --alignment 64 --physical-cores-only \
  --affinity compact --timer-floor-us 100 --planner-regret --guard \
  --json-out /home/hamza-usta/.tmp/mdslc-m6-parallel-audit/RESULT.json

# Physical-core versus sibling-first compact placement.
matcore-bench --m 1024 --n 1024 --k 1024 \
  --variant cpu.native-parallel.avx2-fma.f32.v1 --threads 2 \
  --warmup 7 --iterations 21 --hot-cache --include-packing \
  --reuse-workspace --alignment 64 --affinity compact \
  --physical-cores-only ...  # repeated with --allow-smt
```

## Audit verdict

- **[source-backed]** The persistent runtime is correct, explicit, and
  conservatively synchronized, but its task geometry and topology policy are
  materially less sophisticated than its worker-lifetime machinery.
- **[measured]** The current architecture can scale well when `M` provides
  balanced row bands: retained 4096-cube AVX2 reached 3.863x at four threads,
  and balanced current rectangular proxies reached about 3.2--3.3x.
- **[measured]** It also has reproducible weak boundaries: small `M` cannot
  parallelize, `M=129` is mis-costed, oversized contexts amplify wake latency,
  and sibling-first compact placement loses substantial throughput in the
  bounded host probe.
- **[proposed]** Milestone 7 should first implement task-geometry-aware
  costing, physical-core-first SMT placement, parallel prepacked-B reuse, and
  shape-dependent `M`/`N` partitioning. These changes have higher confidence
  than adding more worker threads or weakening legality gates.
