# MDSLC CPU performance deep-audit findings v1

Milestone: MDSLC Milestone 6 — CPU Performance Deep Audit

Date: 2026-07-26

## Evidence boundary

- **measured —** The authenticated aggregate evidence is the homogeneous schema-v6
  collection in
  [CPU performance deep-audit evidence v1](cpu-performance-deep-audit-v1.md):
  source commit `509ef2b775e501783dfa7f2c4aa21e91f513bd6a`,
  benchmark SHA-256
  `a5a07cf06b6274aeba50a66c20713847f2d65a28ab28021e6d27e64a941c31f5`,
  runner SHA-256
  `be1db49ce5e82d34fc8b455d86c2fe2ad46ea5363a71b0af43b31104f1fd010d`,
  forward manifest
  `b3f872bd0085b15a8cd0cfcc7663af2a41f445355a3e3237c979dc52618362c0`,
  and reverse manifest
  `4939c0c77586e4115dfe5c1aab1ff044d716e9a5d060c9f2ef52f265634df7f8`.
- **measured —** The collection contains 711 executable cases, 583 accepted
  forward reports, 429 accepted reverse controls, 128 authenticated expected
  legality rejections, 58 predeclared runtime-bound skips, zero reused reports,
  and no failed or incomplete state.
- **source-backed —** Complete-hot and one-shot results pair stable-forward and
  stable-reverse order. Diagnostic, prepacked-B, and regret results remain
  forward-only and are reported only within their declared scope.
- **source-backed —** The incomplete schema-v5 collection and every earlier
  mixed or weakly authenticated resume bundle are excluded. They are historical
  fairness findings, not inputs to the aggregates below.
- **measured limitation —** `kernel.perf_event_paranoid=4` blocked hardware
  counters. Hardware IPC, cache/TLB misses, stalled cycles, physical DRAM
  traffic, and workload frequency were not measured.
- **source-backed limitation —** Every multi-thread OpenBLAS result uses an
  unbound `allow-smt/none` provider placement. It is diagnostic-only and is
  never compared as parity or regret evidence with compact, physical-core
  native workers.

Claim labels follow the
[frozen methodology](../audits/cpu-performance-deep-audit-methodology.md):
**measured**, **derived**, **source-backed**, **hypothesis**, and **proposed**.
Supporting evidence is separated into the
[data-movement](../audits/gemm-data-movement.md),
[microkernel](../audits/gemm-microkernel-analysis.md),
[parallel-runtime](../audits/gemm-parallel-runtime.md),
[roofline](../audits/cpu-roofline-and-counters.md),
[BLAS-architecture](../audits/blas-architecture-comparison.md),
[GEMV/GEVM](../audits/gemv-gevm-kernel-design.md), and
[benchmark-fairness](../audits/benchmark-fairness-v2.md) audits. The
[HPC kernel engineering handbook](../HPC_KERNEL_ENGINEERING_HANDBOOK.md)
provides the longer cross-topic synthesis.

## Direct answers to the twenty audit questions

### 1. Why does OpenBLAS win most large regular GEMM shapes?

- **measured —** For complete-hot, equal-placement single-thread cells, the
  fastest native/OpenBLAS median throughput ratios are 0.868 for medium square
  and 0.849 for large square.
- **source-backed —** OpenBLAS combines architecture-selected packing and
  compute kernels, target-specific block/unroll parameters, direct and small
  paths, and more flexible thread decomposition. MDSLC currently uses one
  register/cache geometry for AVX2 and AVX-512, one packed shape path, and a
  one-step K loop. See the
  [mature BLAS comparison](../audits/blas-architecture-comparison.md#why-openblas-wins-most-large-regular-shapes).
- **hypothesis —** The remaining regular-shape gap is the aggregate of
  microkernel scheduling, cache-level supply, copy-kernel quality, checked
  full-tile overhead, and static blocking. Available evidence cannot assign a
  percentage to each cause.

- **derived — recommended bounded conclusion:** OpenBLAS wins the audited
  medium/large single-thread envelope while its mature implementation space is
  broader; MDSLC's precise causal shares remain unresolved.

### 2. Is MDSLC limited primarily by packing, microkernel throughput, cache blocking, scheduling, or threading?

- **measured/derived —** The primary limit is region-dependent:

  - small-M wide transient execution is B-preparation-sensitive;
  - medium/large serial square execution is an execution/cache-level problem,
    not a DRAM-bandwidth problem;
  - tiny and tail-heavy calls are latency/edge-overhead-sensitive;
  - native parallel execution has structural tasking, serial-packing, and
    placement limits.

- **measured —** Across comparable cells, compute-only time is a median 0.903
  of complete-hot time. The diagnostic duration is therefore 9.7% lower, but
  its full-K packed layout and narrower execution scope prevent attributing
  that difference to packing alone or comparing it with complete BLAS.
- **measured limitation —** Counter access is unavailable, so cache,
  execution-port, and synchronization shares cannot be separated numerically.

### 3. Which current tile parameters are demonstrably wrong or suboptimal?

- **measured —** No `MC`, `NC`, or `KC` value is demonstrated universally
  wrong. Controlled `NC=512` and `KC=128` probes produced mixed, noisy results
  rather than a portable improvement.
- **source-backed/derived —** The shared AVX2/AVX-512 4x16 register tile leaves
  the AVX-512 loop with only four accumulator chains and most ZMM registers
  unused. That is a high-confidence architectural limitation, not proof that a
  particular replacement tile wins.
- **proposed —** Treat alternative 8x16/4x32 AVX-512 tiles and
  shape/ISA-specific macro blocks as experiments, not corrected constants.

### 4. Does one static blocking configuration cause shape-family regressions?

- **measured —** Single-thread native/OpenBLAS medians vary materially by
  family: vector-like 1.903, small-square 0.942, short-wide 0.884,
  medium-square 0.868, large-square 0.849, tail-heavy 0.843, and tall-skinny
  0.795.
- **derived —** One tuple creates conflicting A-repack, B-residency, and C-pass
  tradeoffs, but the ratio variation alone does not prove blocking is the
  cause.
- **hypothesis —** A small deterministic family of shape/ISA configurations
  may reduce regressions. It requires calibration/holdout evidence; replacing
  one universal tuple with another is unsupported.

### 5. Is B repacked unnecessarily?

- **source-backed —** B is not repacked for each M block. Single-thread
  transient execution packs each B panel once per call and reuses it across M;
  parallel execution packs the full B image once per submission.
- **source-backed —** Repeated parallel submissions repack unchanged B because
  the parallel path cannot consume the existing caller-owned prepacked-B
  descriptor.
- **derived —** The cross-call parallel repack is removable only through an
  explicit authenticated lifetime/identity contract.

### 6. Can packed B be shared safely across workers?

- **source-backed —** Yes, within one call: B is completely prepared before
  dispatch, immutable while workers run, and read by workers that own disjoint
  C rows and separate A workspaces.
- **source-backed —** Cross-call sharing is not implemented in the parallel
  path. It is safe only with caller-owned storage, source identity, format,
  mutation generation, extent, lifetime, and concurrent-read semantics.

### 7. Are AVX2 and AVX-512 using enough accumulators?

- **source-backed/static model —** AVX2 has eight independent YMM accumulator
  chains; LLVM's static model says that covers its modeled four-cycle FMA
  recurrence at two YMM FMAs per cycle.
- **source-backed/static model —** AVX-512 has four ZMM chains and the same
  modeled 32 FLOP/cycle throughput as AVX2.
- **measured limitation —** This establishes static dependency structure, not
  physical port utilization. A larger AVX-512 tile's performance benefit is
  unmeasured.

### 8. Are loads, address generation, or dependency chains limiting FMA throughput?

- **source-backed/static model —** The AVX2 loop has six loads and eight
  register-register FMAs per K step; LLVM models FP-resource pressure rather
  than AGU saturation and finds no hot-loop spill.
- **source-backed/static model —** AVX-512 uses one vector load and four
  memory-broadcast FMAs with only four recurrence chains.
- **measured limitation —** Hardware load-port, AGU, IPC, and stall data are
  unavailable. “Execution/cache-level limited” is the strongest defensible
  physical classification.

### 9. Is software prefetch useful on this host?

- **source-backed —** Neither current microkernel emits explicit prefetch.
- **hypothesis —** Prefetch is unlikely to repair the sequential 20 KiB
  in-L1 micro-panel loop; outer packed-panel or source-matrix prefetch may help
  cold or rectangular cases.
- **measured limitation —** No controlled prefetch/no-prefetch physical result
  exists. No production prefetch rule is justified.

### 10. Where does AVX-512 downclocking offset wider vectors?

- **measured limitation —** Nowhere is established. APERF/MPERF, MSR-backed
  frequency, and per-sample effective clocks were unavailable.
- **derived —** Current AVX-512's small measured advantage can already be
  explained by a kernel that performs the same work at the same static modeled
  throughput; attributing any crossover to downclocking would be fabrication.

### 11. Are tail paths disproportionately expensive?

- **measured —** Tail-heavy native/OpenBLAS median ratio is 0.843, below
  small-square 0.942. A 3x15 edge microtile takes nearly the same time as a
  full 4x16 microtile and therefore delivers about `45/64` useful efficiency.
- **measured/derived —** Tiny tails pay padded FMAs, edge-buffer work, and call
  overhead. Large single-axis tails near 512 did not show a stable large
  penalty, while crossing an NC boundary can add an extra macro-panel.
- **derived —** Tail cost is disproportionate for small/low-occupancy tiles,
  but “tails always dominate” is unsupported.

### 12. Is the current parallel decomposition suitable for rectangular matrices?

- **source-backed —** Only M is decomposed into 128-row tasks; actual workers
  are capped at `ceil(M/128)` regardless of N and K.
- **measured —** 128 parallel-output-macro-tile-count rejections and 49
  requested-to-actual clamps were retained in the authenticated matrix.
- **derived —** The design is suitable when tall shapes expose balanced M
  bands and structurally unsuitable for M-small/N-large shapes.
- **source-backed limitation —** Multi-thread OpenBLAS throughput at 2, 4, and
  12 configured threads remains unbound diagnostic-only, so no native/provider
  parallel parity ratio is claimed.

### 13. What is the cost of persistent-worker dispatch?

- **measured —** A bounded exact-source probe measured approximately
  1.94–2.73 microseconds for a right-sized one-worker context,
  3.45–3.95 microseconds for four workers, and 10.57–12.41 microseconds when a
  12-worker context activated only one worker. See
  [parallel runtime audit](../audits/gemm-parallel-runtime.md#dispatch-and-synchronization-cost).
- **source-backed —** `notify_all` wakes every context worker each epoch,
  including inactive workers.
- **measured limitation —** These are host- and probe-specific dispatch costs,
  not universal planner constants.

### 14. At which sizes should packing be avoided?

- **measured —** In the authenticated repeated-B aggregate, including one-time
  preparation makes one prepacked execution 1.245 times complete-call time;
  four, 16, and 64 executions reduce the amortized ratios to 0.781, 0.599, and
  0.547.
- **measured —** Earlier bounded data-movement evidence found per-call B
  preparation dominant for M=1/M=8 with N=K=4096, while it removed only 1.8%
  at `1024^3`.
- **derived —** B preparation should be amortized for the measured repeated
  small-M/wide cases. The aggregate break-even lies between one and four
  repeated inputs for the audited set.
- **measured limitation —** No universal direct no-pack dimension cutoff is
  established; prepacking and no-pack execution are different mechanisms.

### 15. Which regions need specialized small, tall, or wide kernels?

- **measured —** Tall-skinny is the weakest single-thread family at median
  ratio 0.795; `4096x64x4096` and `8192x32x1024` are the only reported cells
  below 0.75.
- **measured —** Vector-like native execution is already strong relative to
  this OpenBLAS baseline at median 1.903, despite its measured internal
  B-preparation sensitivity.
- **proposed —** Prioritize tall-skinny register/block profiles, low-occupancy
  tail/full-tile separation, and N/2-D parallel tasking. Test 1/2/4-row no-pack
  paths only as internal latency/packing experiments; do not assume they
  improve the already favorable provider ratio.

### 16. What performance ceiling does roofline analysis predict?

- **measured —** The operational beyond-LLC triad payload ceiling is
  38.90 GB/s.
- **derived/static model —** The current inner loops model at 32 FLOP/cycle,
  giving a nominal 165.05 GFLOP/s CPU0 reference at its sysfs policy maximum
  and an optimistic 1502.3 GFLOP/s all-core reference at all policy maxima.
- **derived limitation —** These are operational/model ceilings, not measured
  silicon peak, effective-frequency throughput, or DRAM-controller bandwidth.
  Shape-specific arithmetic intensity determines the lower roof.

### 17. What fraction of the predicted ceiling is achieved?

- **derived —** Retained representative `1024^3` results place native AVX2 at
  80.0% and OpenBLAS at 90.5% of the nominal CPU0 model ceiling.
- **derived —** At `4096^3`, native AVX2 reaches 39.3% of the optimistic
  all-core arithmetic reference and 48.1% of the operational payload roof;
  OpenBLAS reaches 72.3% and 88.6%, respectively.
- **source-backed limitation —** The parallel percentages are diagnostic
  bounds: provider traffic differs and OpenBLAS placement is unbound, so they
  are neither physical utilization nor parity evidence.

### 18. Which improvements are generic and which are host-specific?

- **source-backed/derived — generic structural findings:**

  - remove unnecessary full-tile edge-buffer work;
  - represent ISA-specific kernel configurations;
  - make transformed-B identity/lifetime explicit;
  - expose M/N task geometry and task-wave makespan;
  - preserve caller-owned workspace and fail-closed legality;
  - separate timing scopes and authenticate provenance.

- **hypothesis/proposed — host-specific choices:**

  - exact MR/NR/MC/NC/KC values and crossovers;
  - AVX2 versus AVX-512 selection;
  - prefetch distance and K-unroll depth;
  - dispatch constants, worker count, core-class weights, and affinity;
  - no-pack/prepack threshold.

- **measured limitation — unmeasured questions:**

  - AVX-512 frequency response;
  - cache/TLB event shares and physical memory traffic;
  - OpenBLAS worker placement/active concurrency;
  - real multi-node NUMA behavior;
  - physical port and frontend/backend stall shares.

### 19. What internal backend contracts are missing?

- **proposed —** The audits identify these private contracts:

  1. `CpuGemmKernelConfigV1` for ISA, block/register geometry, pack schema,
     full/edge functions, and alignment;
  2. planner-visible shape-path classification;
  3. packed-panel identity, ownership, readiness, and invalidation;
  4. an internal microkernel call v2 with separate full/edge entry points;
  5. deterministic M/N/2-D task geometry and shared-panel lifecycle;
  6. phase instrumentation for pack A, pack B, kernel, edge copy, dispatch,
     and wait;
  7. topology performance classes and task-wave-aware cost.

- **source-backed —** These remain private candidates and must not leak
  implementation headers or C++ types through the stable C ABI. See
  [BLAS comparison](../audits/blas-architecture-comparison.md#missing-internal-backend-contracts)
  and the
  [pre-freeze decision log](../../api/PRE_FREEZE_DECISIONS.md#missing-internal-backend-abstractions).

### 20. Which public-API assumptions may need reconsideration before freeze?

- **source-backed —** The later freeze must reconsider fixed candidate arrays,
  public forced-variant enum growth, device-neutral versus CPU-specific
  execution hints, diagnostic-string ownership, packed-content identity,
  parallel prepacking, topology introspection, alpha/beta effects, dynamic
  shapes, transformed-storage lifetime, and numerical-order guarantees.
- **source-backed —** Existing explicit output/effects, typed descriptors,
  fail-closed policy, versioned C records, query-before-execute workspace,
  opaque contexts, capability dimensions, and trusted frontend declarations
  remain structurally sound but are not frozen.
- **proposed —** No public API change follows directly from this audit. The
  complete decision set remains in
  [PRE_FREEZE_DECISIONS.md](../../api/PRE_FREEZE_DECISIONS.md).

## Root causes ranked by confidence and expected impact

| Rank | Root cause | Confidence | Expected impact | Expected affected region | Evidence boundary |
|---:|---|---|---|---|---|
| 1 | Narrow regular-GEMM implementation space and current microkernel/cache-level supply | High for the gap; medium for causal split | High | Medium/large single-thread square and tall-skinny | 0.868/0.849 medium/large ratios; 0.795 tall-skinny; static artifact/roofline evidence; no hardware counters |
| 2 | Per-call B preparation and padded M work | High for B preparation; structural for padding | High within repeated small-M/wide calls | Repeated small-M/wide and low-occupancy tails | Measured prepack amortization and bounded 1/8-row probes; padding contribution not separately timed |
| 3 | M-only 128-row parallel tasking and serial full-B preparation | High structurally; medium for throughput share | High within parallel rectangular/large calls | Short-wide, M-boundary tails, large parallel calls | 128 legality rejections, 49 clamps, source task graph; no fair provider-placement ratio |
| 4 | Four-chain AVX-512 4x16 register tile | High architectural confidence; performance impact unmeasured | Medium to high if a wider tile validates | AVX-512 regular GEMM | Exact ZMM artifact and static model; replacement tile not benchmarked |
| 5 | Checked full/edge path and padded edge execution | High structurally; medium measured impact | Medium | Tiny, small-K, tail-heavy | Full-tile edge-buffer initialization and 3x15 diagnostic |
| 6 | One common blocking tuple for both ISAs and all shapes | Medium | Medium | Tall-skinny, wide and NC/KC boundaries | Conflicting derived tradeoffs; simple block pilots did not win consistently |
| 7 | Planner omits task-wave and placement detail | High structurally; currently bounded impact | Medium | M just above 128-row boundaries and clamped thread plans | Comparable-placement regret median 1.000, p95 1.159, max 1.213 |
| 8 | Oversized-context wake and topology-insensitive placement | High for dispatch mechanism; host-specific magnitude | Medium for small/heterogeneous-core calls | Tiny/small parallel calls and heterogeneous cores | Exact-source dispatch probe and physical/sibling placement probe |

## Evidence-gated Milestone 7 experiments

None of these items is a claimed win or an accepted production change.

1. **proposed — full-tile hot path.** In
   `compiler/lib/runtime/cpu_packed_avx2.cpp` and
   `cpu_packed_avx512.cpp`, separate a prevalidated full-tile symbol from the
   checked edge wrapper. Require exact-symbol proof of no edge-buffer
   initialization/stack traffic, unchanged tail legality, randomized
   correctness, ASan/UBSan, and paired complete-call holdout improvement.
2. **proposed — ISA-specific kernel configuration.** Add a private
   `CpuGemmKernelConfigV1` in the packed backend and prototype AVX-512 8x16 and
   4x32 kernels separately. Require at least eight independent accumulators,
   no spills, exact ZMM FMA inspection, generic-binary ISA gating, all tails,
   and paired schema-v6 calibration/holdout evidence before registration.
3. **proposed — tall-skinny and blocking profiles.** Test a small versioned
   profile set, not ad hoc global constants. Use the frozen calibration and
   holdout families, alternate process order, and reject gains below run
   variation or accompanied by another family's material regression.
4. **proposed — small-M direct paths.** Prototype private 1/2/4-row no-pack or
   lightly packed kernels. Require complete-call latency improvement over the
   best current native variant, exact tail/alignment safety, and no regression
   to the already favorable vector-like provider ratio.
5. **proposed — parallel prepacked B.** Add an internal/additive execution path
   that borrows the authenticated caller-owned descriptor. Require source
   generation, format, extent, lifetime, invalidation, concurrent-read tests,
   output-unchanged failure behavior, and TSan stress; do not add a global
   cache.
6. **proposed — M/N/2-D task graph.** Keep K unsplit. Record M/N ways, tasks,
   waves, disjoint C ownership, shared-panel readiness, barriers, affinity, and
   per-worker workspace. Require short-wide/tail correctness, deterministic
   output, TSan, shutdown/error stress, and native scaling before planner use.
7. **proposed — task-wave-aware planner.** Estimate concrete worker makespan,
   serial packing, active/inactive dispatch, core class, and actual thread
   clamps. Preserve deterministic integer costs and explicit rejections.
   Re-evaluate comparable-placement median/p95/max regret on the frozen matrix.
8. **proposed — targeted wake and placement.** Test targeted worker wakeups and
   physical-core-first SMT ordering. Accept only if repeated execution,
   affinity diagnostics, context shutdown, error propagation, and Linux/Windows
   compatibility remain green.
9. **proposed — lower-confidence scheduling experiments.** Only after the
   structural work, compare K-unroll, register-broadcast, software-pipeline,
   and outer-panel prefetch variants against explicit controls. Do not promote
   based on `llvm-mca` alone.
10. **proposed — phase and counter evidence.** Add diagnostic-only native phase
    timing for pack A/B, kernel, edge copy, dispatch, and wait. Collect
    privileged counters only in a separately authorized environment; keep
    counter absence explicit otherwise.

## Bounded conclusion

- **measured —** The authenticated single-thread evidence shows native median
  ratios from 0.795 to 1.903 across shape families, with tall-skinny the
  weakest and vector-like the strongest relative to OpenBLAS.
- **measured —** One-shot allocation/preparation has median 1.063, p95 2.425,
  and maximum 2.621 time ratio versus reused-workspace complete calls across
  46 comparable cells.
- **measured —** Prepacked-B amortization is unfavorable for one execution and
  favorable by four executions in the audited aggregate.
- **measured —** Comparable-placement planner regret is median 1.000,
  p95 1.159, and maximum 1.213 across 24 cells.
- **source-backed limitation —** Multi-thread OpenBLAS remains unbound
  diagnostic-only; no parallel native/BLAS parity conclusion is supported.
- **derived —** Milestone 6 identifies evidence-backed experiments for
  Milestone 7 but does not establish that any proposed kernel, blocking,
  threading, or planner change will win.
