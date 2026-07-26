# MDSLC HPC kernel engineering handbook

Milestone: MDSLC Milestone 6 — CPU Performance Deep Audit

Scope: native CPU linear-algebra engineering, with F32 row-major GEMM as the
implemented operation

Audit date: 2026-07-26

Production source checkpoint:
`951239f1bee5541a4cf5ad72fab2192de07cf89d`

**source-backed —** This handbook is the engineering synthesis of the
Milestone 6 audit. It is not a benchmark-result replacement, a Milestone 7
parity claim, or a public API freeze.

## Evidence language and boundaries

- **source-backed —** A source-backed statement follows from an identified
  repository source revision or an immutable upstream reference.
- **measured —** A measured statement comes from an identified executable,
  artifact, host, command, and guarded timing or inspection method.
- **derived —** A derived statement is arithmetic or control/data-flow
  reasoning from stated measured or source-backed inputs.
- **hypothesis —** A hypothesis is plausible but is not established by the
  available observations.
- **proposed —** A proposed statement is a future experiment or internal
  change; it is not accepted production behavior.

- **measured —** Physical measurements summarized here used an AMD Ryzen AI 9
  HX 370 host running Ubuntu 26.04 and Linux `7.0.0-27-generic`, with Clang/LLVM
  21.1.8 and the authenticated OpenBLAS 0.3.32 pthread LP64 provider.
- **measured —** Linux reported 12 physical cores, 24 logical CPUs, one NUMA
  node, two LLC groups, 48 KiB L1D and 1 MiB L2 per core, a `performance`
  governor, and enabled boost.
- **measured limitation —** `kernel.perf_event_paranoid=4` rejected
  unprivileged `perf stat`; cycles, hardware IPC, cache/TLB events, branch
  misses, stalled cycles, APERF/MPERF, and memory-controller traffic were not
  measured. Static `llvm-mca` output is never treated as a hardware counter.
- **measured historical exclusion —** The first schema-v5 collection stopped
  at case 567 and the independent fairness-v2 review rejected its resume,
  one-shot, prepack-preparation, placement, and completeness contracts. That
  mixed/incomplete bundle remains excluded from every final aggregate.
- **measured —** The final homogeneous schema-v6 collection identifies source
  commit `509ef2b775e501783dfa7f2c4aa21e91f513bd6a`, benchmark SHA-256
  `a5a07cf06b6274aeba50a66c20713847f2d65a28ab28021e6d27e64a941c31f5`,
  runner SHA-256
  `be1db49ce5e82d34fc8b455d86c2fe2ad46ea5363a71b0af43b31104f1fd010d`,
  forward manifest
  `b3f872bd0085b15a8cd0cfcc7663af2a41f445355a3e3237c979dc52618362c0`,
  and reverse manifest
  `4939c0c77586e4115dfe5c1aab1ff044d716e9a5d060c9f2ef52f265634df7f8`.
- **measured —** It contains 711 executable cases, 583 accepted forward
  reports, 429 accepted reverse controls, 128 authenticated expected legality
  rejections, 58 predeclared runtime-bound skips, zero reused reports, and no
  failed or incomplete state.
- **derived limitation —** Multi-thread native and OpenBLAS numbers are not
  parity evidence when native workers use authenticated compact physical-core
  placement while OpenBLAS uses an unbound provider team, even if requested
  and configured thread counts are numerically equal.

Primary repository evidence:

- [Authenticated CPU deep-audit evidence v1](cpu/cpu-performance-deep-audit-v1.md)
- [CPU deep-audit findings v1](cpu/cpu-performance-deep-audit-findings-v1.md)
- [CPU deep-audit methodology](audits/cpu-performance-deep-audit-methodology.md)
- [GEMM data movement and packing](audits/gemm-data-movement.md)
- [GEMM microkernel and instruction scheduling](audits/gemm-microkernel-analysis.md)
- [GEMM parallel runtime and topology](audits/gemm-parallel-runtime.md)
- [CPU roofline and counter audit](audits/cpu-roofline-and-counters.md)
- [GEMV and GEVM kernel design](audits/gemv-gevm-kernel-design.md)
- [Mature BLAS architecture comparison](audits/blas-architecture-comparison.md)
- [Benchmark fairness audit v2](audits/benchmark-fairness-v2.md)
- [Public API pre-freeze decision log](../api/PRE_FREEZE_DECISIONS.md)

### Final authenticated aggregate

- **measured —** For complete-hot, equal-placement single-thread comparisons,
  the median fastest-native/OpenBLAS throughput ratios are 0.849 large-square,
  0.868 medium-square, 0.884 short-wide, 0.942 small-square, 0.843 tail-heavy,
  0.795 tall-skinny, and 1.903 vector-like.
- **measured —** Allocation-inclusive one-shot time divided by equivalent
  reused-workspace complete-call time is median 1.063, p95 2.425, and maximum
  2.621 across 46 comparable cells.
- **measured —** The prepacked-B aggregate includes one authenticated
  preparation call. Its median amortized/complete ratios are 1.245, 0.781,
  0.599, and 0.547 for 1, 4, 16, and 64 repeated A inputs, respectively.
- **measured —** Comparable-placement planner regret across 24 cells is median
  1.000, p95 1.159, and maximum 1.213; 45 candidate timings with mismatched
  thread or placement contracts are excluded.
- **measured limitation —** Multi-thread OpenBLAS reports configure 2, 4, or
  12 provider threads but do not authenticate provider CPU placement. Their
  166.482, 252.490, and 374.638 median GFLOP/s are retained only as unbound
  diagnostics and have no native/OpenBLAS ratio.

## 1. MDSLC CPU execution architecture

- **source-backed —** Valid C++ `.mdsl` source is authenticated by the native
  Clang frontend, converted through typed Matcore IR, planned against a fixed
  CPU variant registry, rewritten through the stable C ABI, and linked as an
  ordinary native artifact.
- **source-backed —** The current F32 row-major contiguous GEMM registry
  separates reference, tiled, compiler-vectorized, native packed AVX2,
  native packed AVX-512, native parallel AVX2, native parallel AVX-512, and
  optional OpenBLAS implementations. Forced illegal variants fail instead of
  silently falling back.
- **source-backed —** The runtime keeps matrix storage and large packing
  workspace caller-owned. The additive workspace and prepacked-B contracts
  query required size and alignment before execution; insufficient storage
  fails before output mutation.
- **source-backed —** CPU capability records distinguish hardware support,
  OS-enabled state, compiler support, implementation availability, and
  physical runtime validation. ISA-specific functions are isolated instead of
  compiling the whole runtime with AVX2 or AVX-512.
- **source-backed —** The persistent execution context owns workers but not
  matrix workspace. Repeated submissions reuse workers, serialize submissions
  within one context, and preserve disjoint output ownership.
- **derived —** The architecture already contains the essential semantic,
  legality, ownership, and dispatch layers of a native GEMM system. Milestone
  7 needs a broader internal implementation space, not a replacement compiler
  or a hidden-memory BLAS wrapper.

- **source-backed —** Principal source anchors are the
  [planner registry](../../compiler/lib/planner/cpu_planner.h),
  [planner v3](../../compiler/lib/planner/cpu_planner_v3.cpp),
  [runtime ABI](../../compiler/include/matcore/runtime_c.h), and
  [execution context](../../compiler/lib/runtime/cpu_execution_context.cpp).

## 2. GEMM arithmetic and data movement

- **derived —** Row-major F32 GEMM performs
  `F = 2*M*N*K` FLOPs under the conventional multiply-add count.
- **derived —** An optimistic unique-payload lower bound is
  `4*(M*K + K*N + M*N)` bytes: read A and B once and write C once. It is not a
  measured DRAM-traffic value.
- **source-backed —** The native packed loop order is `NC -> KC -> MC`. Each
  transient `(KC,NC)` B panel is packed once and reused across M blocks; each
  `(MC,KC)` A block is packed once for every N macro-panel.
- **derived —** Approximate transient packed writes are
  `ceil(N/NC)*M*K*4` bytes for A, with M-tail padding, and
  `round_up(N,NR)*K*4` bytes for B. B is not repacked for every M block.
- **source-backed —** Every `KC` panel after the first reloads C before
  accumulation and stores it afterwards.
- **derived —** With `P=ceil(K/KC)`, that is `2*P-1` logical C
  pass-equivalents. This is a code-visible traffic model, not evidence that
  every pass reaches main memory.
- **measured —** Prepacked B removed 83.5% of transient time for
  `1x4096x4096`, 72.8% for `8x4096x4096`, 15.3% for `64x1024x1024`, 1.8% for
  `1024^3`, and 0.5% for `1024x64x1024` in the bounded data-movement sweep.
- **derived —** Per-call B preparation is the dominant removable cost for
  small-M/wide repeated execution, while square and tall-narrow cases are
  primarily constrained elsewhere.

## 3. Roofline analysis

- **source-backed —** The roofline model bounds attainable throughput by the
  lesser of an arithmetic ceiling and bandwidth times operational arithmetic
  intensity; this handbook follows Williams, Waterman, and Patterson,
  [Roofline](https://doi.org/10.1145/1498765.1498785).
- **measured —** A guarded, aligned, vectorized payload probe measured a
  retained beyond-LLC triad ceiling of 38.90 GB/s on the validation host.
  This is application payload, not memory-controller bytes.
- **measured —** The same probe measured footprint-sensitive triad rates of
  445.55 GB/s at L1D scale, 224.55 GB/s at L2 scale, and 136.97 GB/s at local
  L3 scale.
- **derived —** The exact current AVX2 and AVX-512 inner loops both model at
  32 FLOP/cycle under `llvm-mca-21 -mcpu=znver5`; at CPU 0's sysfs policy
  maximum this gives a nominal 165.05 GFLOP/s single-core reference ceiling.
  The frequency and scheduling-model assumptions make this a bound, not a
  physical peak measurement.
- **derived —** At `1024^3`, native AVX2 achieved 80.0% and OpenBLAS 90.5% of
  that nominal CPU0 ceiling in the retained sample. DRAM bandwidth is not the
  primary serial-square constraint because even the pessimistic native-outer
  intensity gives a memory roof far above the arithmetic bound.
- **measured —** Prepacking raised `1x4096x4096` from 4.17 to 22.58 GFLOP/s
  and `8x4096x4096` from 21.44 to 99.29 GFLOP/s in the retained roofline
  sample.
- **derived —** Vector-like GEMM is first packing-sensitive and then
  cache/bandwidth-sensitive; medium and large serial square GEMM is primarily
  an execution/cache-level problem; tiny and heavily tailed calls are
  latency/edge-overhead sensitive.
- **hypothesis —** Cache-level traffic, topology, and synchronization jointly
  contribute to the large parallel gap, but blocked hardware events prevent a
  defensible numerical apportionment.

## 4. Cache and TLB behavior

- **derived —** With `KC=256`, one full 4-row packed A micro-panel is 4 KiB
  and one 16-column packed B micro-panel is 16 KiB. Their 20 KiB total fits
  within the host's 48 KiB L1D.
- **derived —** The largest transient A and B macro buffers are 128 KiB and
  256 KiB, respectively. Their 384 KiB total fits within the host's 1 MiB
  private L2, while source matrices, C, code, and runtime state compete for
  the remaining capacity.
- **derived —** The transient packed workspace spans at most 96 ordinary
  4 KiB pages under the current block sizes. Full prepacked B spans 1,024 pages
  at `1024^2` and 16,384 pages at `4096^2`.
- **hypothesis —** Large-stride source panels and full-prepacked images can
  pressure data TLBs, but no DTLB event was available because
  `perf_event_paranoid=4`.
- **hypothesis —** The existing block tuple is cache-conservative on CPU 0
  rather than obviously oversized. Address-controlled or counter-enabled work
  is required before attributing a regression to associativity or replacement.
- **source-backed —** Goto and van de Geijn explain why useful blocking is not
  merely “fit everything in L1”: packed panels may stream from L2, and TLB
  reach may constrain blocking. See
  [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf).
- **proposed —** Preserve page footprint, alignment, and source stride in
  diagnostic records; do not introduce implicit huge pages, page migration,
  or host-specific prefetch rules.

## 5. Macro blocking

- **source-backed —** Current native AVX2 and AVX-512 share
  `MR=4`, `NR=16`, `MC=128`, `NC=256`, and `KC=256`, defined in
  [cpu_gemm_backend.h](../../compiler/lib/runtime/cpu_gemm_backend.h).
- **measured —** Controlled `NC=512` and `KC=128` sweeps did not establish a
  stable universal improvement. `NC=512` was about 1% faster at `1024^3` but
  about 6% slower at `8x4096x4096`; `KC=128` was about 2% slower at `1024^3`.
- **derived —** Increasing NC may reduce A repacking while increasing packed-B
  residency; reducing KC shrinks the panel working set while increasing C
  accumulation passes.
- **source-backed —** OpenBLAS uses architecture-selected block/unroll
  parameters and several shape paths. BLIS stores regular and small/skinny
  block sizes and kernels in an architecture context.
- **hypothesis —** MDSLC needs a small deterministic family of
  ISA/shape-specific configurations rather than one replacement global tuple.
- **proposed —** Any profile family must be calibrated on the frozen diagnostic
  and calibration partitions, then accepted or rejected on the unchanged
  holdout matrix; sub-percent changes below run-to-run variability do not
  justify production selection.

## 6. Packing layouts and ownership

- **source-backed —** A is packed as `[4-row micro-panel][k][row lane]`, and B
  as `[16-column micro-panel][k][column lane]`; missing M/N lanes are
  zero-filled.
- **source-backed —** The 64-byte workspace alignment makes aligned packed-B
  vector loads legal at each K step. Output accesses remain unaligned-safe.
- **source-backed —** Transient execution uses one A block and one B block in
  caller-owned workspace. Parallel execution uses one full read-only packed-B
  image plus one 64-byte-separated A workspace per active worker.
- **source-backed —** Parallel B packing is serial on the submitting thread
  and completes before dispatch. Workers do not duplicate packed B.
- **source-backed —** The single-thread prepacked-B descriptor authenticates
  source address, dimensions, block geometry, storage extent, and provenance.
  The parallel path cannot yet consume it and therefore repacks B every call.
- **derived —** Safe within-call B sharing is already solved; safe cross-call
  sharing requires an explicit identity/lifetime contract rather than a
  global cache.
- **proposed —** A private packed-panel descriptor should state schema,
  logical/padded extents, panel strides, alignment, ownership, readiness,
  source identity, and invalidation.
- **proposed —** Parallel prepacked-B execution may borrow caller-owned
  immutable storage after full authentication; hidden allocation or ambiguous
  content caching remains prohibited.

## 7. Register blocking

- **source-backed —** The current AVX2 4x16 kernel holds eight YMM accumulator
  chains, two packed-B vectors, and four broadcast A vectors.
- **source-backed —** The current AVX-512 4x16 kernel holds only four ZMM
  accumulator chains and one packed-B vector; 27 of 32 architectural ZMM
  registers are unused in its hot loop.
- **derived —** Both kernels perform 128 FLOPs per K step, so the wider
  AVX-512 instructions reduce instruction count without increasing register
  tile work or reuse.
- **measured —** No hot-loop register spill or stack access was present in
  either exact Release symbol.
- **derived —** AVX2's eight independent accumulators cover the LLVM model's
  four-cycle FMA latency at two modeled YMM FMAs per cycle. AVX-512's four
  recurrences provide a shorter latency-hiding distance.
- **hypothesis —** An AVX-512 8x16 or 4x32 kernel could use eight accumulators
  and expose more independent work. The better geometry remains host- and
  edge-behavior-dependent.
- **proposed —** Record register tile, pack schema, full/edge entry points, and
  ISA requirements in a private versioned kernel configuration rather than
  forcing AVX2 and AVX-512 to share one geometry.

## 8. AVX2 microkernel design

- **measured —** The exact Release symbol
  `matcore_cpu_packed_avx2_4x16_microkernel_f32_v1` was 927 bytes and its K
  loop contained two aligned YMM B loads, four scalar-to-YMM broadcasts, and
  eight `vfmadd231ps` instructions.
- **source-backed —** The loop advances one K element at a time; Clang emitted
  no K unrolling or explicit software prefetch.
- **derived —** A full K step performs 128 FLOPs and reads 80 packed-panel
  bytes, for an L1-facing input intensity of 1.6 FLOP/byte before amortized C
  traffic.
- **measured —** `llvm-mca` predicted four cycles per loop and 32 FLOP/cycle
  with FP-resource pressure rather than AGU saturation. This is a static
  scheduling result.
- **measured —** Bounded complete-call AVX2 throughput was 110.21, 127.72,
  130.53, and 123.16 GFLOP/s for square 128, 256, 512, and 1024,
  respectively, in the microkernel audit.
- **hypothesis —** K unrolling may reduce branch/address overhead but cannot
  raise the current LLVM-model FP ceiling without another scheduling or tile
  change.
- **proposed —** First remove avoidable full-tile wrapper overhead, then test
  alternate AVX2 MR/NR or K-unroll choices with exact-symbol inspection and
  complete-call holdout timing.

## 9. AVX-512 microkernel design

- **measured —** The exact Release symbol
  `matcore_cpu_packed_avx512_4x16_microkernel_f32_v1` was 745 bytes and its K
  loop contained one ZMM B load plus four `vfmadd231ps` operations using
  embedded scalar broadcasts.
- **source-backed —** The AVX-512 path deliberately reuses AVX2's 4x16
  packing and blocking contract and contains no K unrolling or explicit
  prefetch.
- **measured —** `llvm-mca` predicted the same four-cycle, 32-FLOP/cycle block
  throughput as AVX2 for the current body.
- **measured —** In the bounded complete-call sample, AVX-512 exceeded AVX2 by
  only 1.1% to 5.1% across 128–1024 square and `511x513x515`.
- **derived —** Wider vectors do not imply a 2x result when useful work and
  modeled loop throughput remain unchanged.
- **measured limitation —** APERF/MPERF and workload frequency were
  unavailable. The audit does not claim that AVX-512 downclocking caused any
  crossover.
- **proposed —** Prototype 8x16 and 4x32 kernels separately, require at least
  eight accumulator chains, no spills, exact ZMM FMA artifact evidence,
  fail-closed ISA gating, complete tails, and fair complete-call improvement.
- **proposed —** Only after selecting a register tile should K unroll factors
  and memory-broadcast versus register-broadcast forms be compared.

## 10. Tail and alignment handling

- **source-backed —** Current M and N tails are zero-padded in packed panels.
  Incomplete output tiles use a 4x16, 64-float stack buffer; K uses the exact
  logical trip count and does not require padding.
- **measured —** Both current exact microkernel symbols initialize the
  256-byte edge buffer before deciding whether the tile is complete, so full
  tiles pay the zero-fill tax.
- **measured —** A 3x15 edge microtile took nearly the same time as a full 4x16
  tile at K 8, 64, and 256, yielding approximately the useful-lane ratio
  `45/64 = 0.703`.
- **derived —** `31x33` output executes a 32x48 register-tile envelope, about
  50.1% more FMAs than useful output requires.
- **measured —** A 513-column case was about 5% below nearby full dimensions,
  but it also crosses an NC boundary and creates a third N macro-panel for one
  useful column.
- **derived —** Microkernel edge cost and macro-panel boundary repacking must
  be measured separately before assigning causality.
- **source-backed —** Packed B uses aligned loads; C uses unaligned-safe loads
  and stores. No out-of-range speculative vector load was found in the audited
  symbols.
- **proposed —** Split a prevalidated full-tile hot function from a checked
  edge wrapper, and evaluate narrow or masked edge kernels only with complete
  correctness, sanitizer, alignment, and NC-boundary tests.

## 11. Small-GEMM strategies

- **derived —** M=1 still packs four A rows and executes four accumulator rows,
  producing four times the useful FMAs before packing overhead.
- **measured —** Tiny microkernel calls show an approximately 2 μs floor, and
  retained tiny bound-context calls were dominated by dispatch rather than
  arithmetic.
- **source-backed —** Current native GEMM has no no-pack small family; all
  packed paths pay panel preparation, checked-call, and padded-edge costs.
- **source-backed —** OpenBLAS has direct, small-kernel, and GEMM-to-GEMV
  dispatch paths; BLIS provides alternate small/skinny “sup” paths. Their
  exact thresholds are upstream implementation examples, not MDSLC rules.
- **proposed —** Test private 1xNR, 2xNR, and 4xNR direct or lightly packed
  kernels, keep exact M/N/K tails, and use complete-call latency rather than
  compute-only throughput to choose crossovers.
- **proposed —** A one-thread plan in an oversized persistent context may use
  a documented caller-thread fast path only if selected affinity semantics are
  preserved; otherwise use a right-sized or targeted-wake execution path.

## 12. Tall-skinny and short-wide strategies

- **source-backed —** Native parallel tasks are 128-row M bands. Task count is
  `ceil(M/128)` and N/K do not create tasks.
- **measured —** Four-thread native AVX2 was illegal for
  `64x1024x1024` and `64x4096x4096` because each has only one M task.
- **measured —** Equal-work tall proxies with M=1024 achieved about 3.2–3.3x
  over packed single-thread at four threads in the bounded parallel audit.
- **derived —** The current decomposition suits tall shapes that expose enough
  balanced row bands and structurally cannot scale short-wide/vector-like
  shapes with `M<=128`.
- **source-backed —** A is repacked once per NC panel, so wide N repeats
  A preparation even though A values are unchanged.
- **proposed —** Select among M, N, and 2-D output-tile decomposition by
  declared shape class. Keep K unsplit until a versioned deterministic
  reduction contract exists.
- **proposed —** N partitions must preserve disjoint output ownership and
  cache-line-aware boundaries. A shared-B panel lifecycle must make readiness,
  reuse, workspace, and barriers explicit.

## 13. Threading and persistent execution

- **source-backed —** A context creates a fixed worker team once, reuses it for
  submissions, and joins workers only at shutdown. It serializes submissions
  within one context and rejects worker reentrancy and unsafe provider/native
  nesting.
- **source-backed —** Task assignment is deterministic cyclic assignment over
  M bands, with no stealing or performance weighting.
- **measured —** Empty submission was about 1.94–2.73 μs for an exact
  one-worker context, 3.45–3.95 μs for four workers, and 10.57–12.41 μs when a
  12-worker context activated only one task.
- **source-backed —** `notify_all` wakes every worker on every epoch; inactive
  workers still inspect the shared epoch under the state mutex.
- **derived —** At `M=129`, two row bands hold 128 and one rows, giving only
  50.4% ideal equal-speed utilization before packing and dispatch.
- **measured —** Automatic planning selected two-thread parallel AVX-512 for
  `129x512x512` at 0.635 ms while single-thread packed AVX-512 was fastest at
  0.553 ms, regret 1.148.
- **measured —** In the final authenticated native-only scaling diagnostic,
  large-square four-thread execution has median 3.279x speedup and 0.820
  parallel efficiency over its corresponding one-thread packed-ISA baseline.
  Twelve-thread large-square execution has median 6.108x speedup and 0.509
  efficiency. These are native placement-matched diagnostics, not OpenBLAS
  parity.
- **measured —** Vector-like native scaling remains weak: median speedup is
  1.257x at four threads and 1.256x at twelve, consistent with M-only task
  underexposure.
- **proposed —** Cost predicted makespan from concrete task waves and worker
  classes, and do not activate a worker whose tail band does not reduce it.
- **proposed —** Evaluate targeted wakeups, physical-core-first placement,
  N/2-D task geometry, and explicit parallel prepacked-B before adding more
  threads or weakening legality.

## 14. NUMA and affinity

- **measured —** The validation host has one physical NUMA node; no
  cross-node runtime performance was measured.
- **source-backed —** Current public context creation sets
  `allow_cross_numa=false`, authenticates requested worker binding, and
  reports that no page placement was applied.
- **source-backed —** Compact/scatter placement is socket/node aware but does
  not weight core classes or explicitly distribute over the host's two LLC
  groups.
- **source-backed —** Compact `allow-smt` ordering is sibling-first. A
  two-thread plan selected CPUs `[0,12]`.
- **measured —** In a bounded `1024^3` AVX2 comparison, sibling placement
  reached 138.4 GFLOP/s versus 242.8 GFLOP/s on physical cores. Process-order
  and dynamic-frequency effects were not fully neutralized.
- **hypothesis —** Sibling contention is the primary reason for that bounded
  difference, but the exact ratio is not a calibration constant.
- **proposed —** Prefer all authenticated physical cores before SMT siblings,
  record LLC/core-class information when reliable, and preserve an explicit
  unknown state on platforms that cannot discover it.
- **proposed —** Keep multi-node execution fail-closed until page
  ownership/first-touch, packed-B placement, output partitioning, and
  cross-node policy are executable and physically validated. Synthetic NUMA
  tests do not establish physical performance.

## 15. GEMV and GEVM design

- **source-backed —** No public GEMV/GEVM declaration, typed operation, C ABI
  entry point, planner variant, or executable kernel was added in Milestone 6.
- **derived —** Large GEMV and GEVM have an ideal arithmetic-intensity limit
  approaching 0.5 FLOP/byte, so they are bandwidth-oriented Level-2
  operations rather than GEMM kernels that should inherit packed-GEMM rules.
- **proposed —** Private GEMV semantics should initially overwrite contiguous
  F32 `y[M]` with row-major `A[M,K] * x[K]`; private GEVM semantics should
  overwrite `y[N]` with `x[K] * A[K,N]`. Both should reject overlap, hidden
  allocation, implicit migration, and unsupported layout.
- **derived —** Row-major GEMV favors several independent row-dot accumulators
  that share x loads but require horizontal reductions.
- **derived —** Row-major GEVM favors contiguous N-block accumulation or a
  row-stream AXPY form; naïve column-wise traversal wastes cache-line payload.
- **proposed —** Direct no-pack execution should be the initial private path.
  Persistent transformed matrices require explicit caller ownership, format
  identity, source generation/invalidation, size bounds, and concurrent-read
  semantics.
- **proposed —** GEMV and GEVM may share a private orientation-aware matvec
  problem while retaining distinct source/IR identities. Public names,
  vector-view shape, alpha/beta, batching, reduction, and lifetime remain
  decisions for the later API freeze.

## 16. Prepacking and repeated execution

- **source-backed —** Single-thread prepacked-B execution removes B
  preparation from each timed call but retains A packing and full compute.
- **source-backed —** Benchmark sequence mode cycles distinct, aligned A inputs
  against fixed B and independently authenticates every invocation.
- **measured historical exclusion —** The rejected schema-v5 runner recorded
  steady-state prepacked execution without one-time B preparation. Its
  purported amortization evidence is excluded.
- **source-backed —** Schema v6 records one separately timed, authenticated B
  preparation call, steady-state execution, and the derived amortized total
  `prepare_time + repetitions*execute_time`.
- **measured —** Across eight cells per sequence length, the median preparation
  time is 7.785–8.354 ms. The amortized/complete ratio is 1.245 for one
  execution, then 0.781, 0.599, and 0.547 for 4, 16, and 64 executions.
- **derived —** The authenticated aggregate crosses break-even between one and
  four executions. That is evidence for the audited fixed-B set, not a
  universal prepacking threshold or a direct no-pack crossover.
- **proposed —** Do not create a process-global cache. Any persistent packed
  object must be owned by an explicit context or caller, authenticate source
  identity and content generation, define invalidation and concurrency, and
  have a bounded size.
- **proposed —** Repeated vectors against one matrix may eventually lower to a
  batched matrix operation when semantics allow; a loop of hidden independent
  matvec calls is not automatically optimal.

## 17. External BLAS integration

- **source-backed —** MDSLC's optional adapter calls typed row-major CBLAS
  SGEMM, controls the OpenBLAS local thread count, checks configured provider
  threads, and restores the previous policy.
- **source-backed —** OpenBLAS v0.3.32 combines architecture dispatch,
  direct/small/GEMV-forwarded paths, target copy kernels, target blocking,
  regular packed drivers, and M/N-partitioned threading.
- **source-backed —** BLIS 2.1 separates control trees from architecture
  context, exposes pack schemas and microkernel contracts, and supplies
  separate small/skinny paths and loop-level threading choices.
- **measured/derived —** In the final complete comparable single-thread table,
  OpenBLAS is faster on 30 of 35 cells. The family-median native/OpenBLAS ratio
  is 0.849 for large square and 0.795 for tall-skinny, while native leads the
  vector-like family at median 1.903.
- **measured —** At `1024^3`, the fastest native result is packed AVX2 at
  131.217 GFLOP/s versus OpenBLAS at 147.119 GFLOP/s, ratio 0.892. The two
  weakest cells are tall-skinny `4096x64x4096` at 0.749 and
  `8192x32x1024` at 0.750.
- **derived —** OpenBLAS does not win because it alone reuses a packed B panel;
  MDSLC already does that. Its broader architecture/shape-specific
  implementation space is the important comparison.
- **derived limitation —** Provider thread-count control does not authenticate
  which provider CPUs ran. Unbound OpenBLAS multi-thread results must remain a
  separate placement stratum from bound native workers.
- **proposed —** Keep OpenBLAS as an honest legal candidate and external
  baseline. Do not copy its code, thresholds, synchronization, or hidden
  memory model into MDSLC.

Pinned upstream evidence:

- [OpenBLAS v0.3.32 GEMM dispatch](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/interface/gemm.c)
- [OpenBLAS v0.3.32 packed driver](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/driver/level3/level3.c)
- [OpenBLAS v0.3.32 threaded driver](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/driver/level3/level3_thread.c)
- [BLIS 2.1 microkernel contract](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/docs/KernelsHowTo.md)
- [BLIS 2.1 small/skinny architecture](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/docs/PerformanceSmall.md)
- [BLIS 2.1 threading guidance](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/docs/Multithreading.md)

## 18. Planner cost modelling

- **source-backed —** Planner-v3 requires at least two 128-row tasks and
  approximately `2^20` multiply-accumulates per active thread before admitting
  a parallel candidate.
- **source-backed —** Its static parallel cost includes divided arithmetic
  work, A/B packing terms, fixed overhead, and per-thread overhead, but omits
  exact row-wave makespan, inactive-worker wakeups, core classes, LLC groups,
  prepacked-B mode, and provider placement.
- **derived —** Dividing total work evenly by thread count mis-costs
  `M mod 128` tails such as 129, 192, and 257 rows.
- **source-backed —** The deterministic registry records legality, rejection
  reasons, estimated cost, priority, selected variant, workspace, and
  capabilities.
- **measured —** The final sanitized recomputation compares only candidates
  with the selected candidate's actual thread count and placement. Across 24
  cells it reports regret median 1.000, p95 1.159, and maximum 1.213, while
  visibly excluding 45 mismatched candidate timings.
- **proposed —** Add private planner inputs for shape path, kernel
  configuration, packing mode, task geometry, waves, actual worker classes,
  placement, packed-object identity, and measured/default dispatch cost.
- **proposed —** Compute a deterministic predicted makespan rather than
  assuming equal work. Preserve saturating integer arithmetic, stable
  candidate order, explicit rejection, and no runtime autotuning.
- **proposed —** Calibrate on a frozen matrix and report holdout regret. Keep
  OpenBLAS selectable when it is demonstrably faster; planner-selected
  OpenBLAS is never native parity.

## 19. Fair benchmark methodology

- **source-backed —** Retained reports must distinguish one-shot
  allocation-included, reused-workspace packing-included, prepacked-B, and
  compute-only diagnostic scopes. Compute-only native timing is not directly
  comparable to complete CBLAS SGEMM.
- **source-backed —** Complete hot calls use a declared timer floor, warmups,
  repeated aggregate samples, deterministic inputs, and output authentication
  after timing. Cold-cache mode is a best-effort 64 MiB eviction diagnostic,
  not proof of a named cache state.
- **source-backed —** Fair native/provider comparisons require equal operation
  semantics, requested and actual threads, placement class, cache mode,
  packing scope, allocation scope, and correctness.
- **source-backed —** The frozen matrix includes small, medium, large,
  tall-skinny, short-wide, vector-like, tail-heavy, and repeated fixed-B
  families, with diagnostic/calibration/holdout partitions fixed before
  optimization.
- **measured historical exclusion —** The first schema-v5 run stopped at case
  567 and fairness-v2 proved that its resume identity, one-shot, and
  prepack-preparation contracts were insufficient. It remains excluded from
  every final aggregate.
- **source-backed —** The final schema-v6 runner authenticates the benchmark
  binary, runner, source commit, seed, normalized case plan, environment,
  measurement options, provider identity, and every accepted raw-file digest;
  any identity mismatch rejects resume.
- **measured —** The final forward collection has 583 accepted reports and no
  reuse. Paired complete-hot/one-shot reverse control has 429 accepted reports
  and no reuse. All failures and incomplete states are fatal to sanitization.
- **source-backed —** Complete-hot and one-shot aggregates use arithmetic
  midpoints of stable-forward and stable-reverse process-order medians and
  retain their direction range. Diagnostic, prepacked-B, and regret entries
  remain explicitly forward-only.
- **source-backed —** Schema-v6 raw reports retain ordered normalized timing
  samples, one-time prepack duration, complete provenance, and structured
  rejection categories. The sanitized report publishes every planned cell as
  passed, rejected, or predeclared skip.
- **source-backed —** Raw JSON, logs, binaries, profiler databases, and
  disassemblies remain external or under ignored paths; only reviewed
  summaries and SHA-256 inventories belong in Git.

## 20. Kernel correctness and artifact validation

- **source-backed —** Every accelerated variant must pass forced execution;
  rank/shape/layout/dtype/memory-space/mutability/alias/workspace legality;
  null, overflow, alignment, M/N/K-tail, overlap, repeated-execution, and
  unavailable-ISA rejection tests.
- **source-backed —** Rejected execution must leave output unchanged, and no
  unsupported forced request may silently fall back.
- **source-backed —** Small benchmark outputs use a full independent
  double-precision element oracle. Larger outputs use deterministic sampled
  elements plus an independent double-precision checksum/error bound; reports
  must name that oracle rather than call it full-element validation.
- **source-backed —** Sequence mode replays every distinct A input outside the
  interval and authenticates every output.
- **measured —** Exact artifact tests found YMM packed FMA instructions in the
  AVX2 symbol and ZMM packed FMA instructions in the AVX-512 symbol; neither
  hot loop was scalarized.
- **source-backed —** Release/Debug, ASan/UBSan, TSan for shared state, strict
  C ABI, install/package/consumer, source-inaccessible installed-tree,
  repository hygiene, and Windows compatibility remain acceptance gates.
- **proposed —** New full-tile, edge, no-pack, parallel-prepack, and 2-D
  scheduling paths must add exact symbol and sanitizer/race tests before
  planner promotion.

## 21. Performance regression diagnosis

- **proposed —** Diagnose in layers: first authenticate semantics and timing
  scope; then compare plan/actual threads and placement; then separate
  allocation, B preparation, A packing, compute, output edges, dispatch, and
  waiting.
- **proposed —** Reproduce each regression with alternating process order and
  a nearby control shape. Preserve raw sample vectors, environment,
  frequency-policy metadata, executable digest, and source provenance.
- **derived —** A vector-like transient/prepacked gap points first to B
  preparation; a square complete/compute gap of only a few percent points
  instead to microkernel/cache-level supply.
- **derived —** A sharp regression at `N=NC*q+1` may be an extra macro-panel,
  not only an edge microkernel. A regression at `M=MC*q+1` may be task-wave
  imbalance, not only tail arithmetic.
- **derived —** Requested thread count is insufficient evidence; actual
  threads, task waves, bound CPUs, core classes, LLC groups, and provider
  placement determine whether comparisons are equivalent.
- **proposed —** Use exact disassembly to reject scalarization, spills, missing
  ISA, or unintended prologue work. Use `llvm-mca` to generate scheduling
  hypotheses, never to replace physical timing/counters.
- **proposed —** When counters are unavailable, state the missing observation
  and stop at the strongest supported classification. Do not infer cache/TLB
  misses, hardware IPC, downclocking, or memory-controller bytes from elapsed
  time alone.
- **proposed —** Reject planner-threshold changes whose gains do not survive
  the frozen holdout set and whose worst-case regret or correctness gate
  regresses.

## 22. Concrete MDSLC findings and next actions

### Ranked findings

1. **measured/source-backed — high confidence, high affected-region impact:**
   B preparation is measured dominant for repeated M=1/M=8 wide calls. Padded
   M work and M-only tasking are separate derived structural losses whose
   shares are not measured.
2. **source-backed/measured — high structural confidence, high expected
   parallel-region impact:** parallel execution serially prepares full B, uses
   M-only 128-row tasks, and exposes task-wave, wakeup, and placement limits.
   Multi-thread OpenBLAS is unbound, so no provider-relative share is claimed.
3. **source-backed/static model — high architectural confidence, unmeasured
   replacement impact:** the AVX-512 kernel retains a 4x16 tile and only four
   accumulators, giving the same static 32-FLOP/cycle model ceiling as AVX2.
   A larger tile's complete-call benefit is not established.
4. **measured — high confidence, medium impact:** every full microtile pays
   edge-buffer initialization and validation/prologue overhead; partial tiles
   compute all padded lanes.
5. **measured/derived — medium confidence, medium-to-high impact:** one shared
   blocking tuple exposes conflicting A-repack, B-residency, and C-pass
   tradeoffs, but a simple global NC/KC replacement did not win consistently.
6. **measured/derived — high confidence, medium impact:** current planner cost
   misses row-wave makespan and can choose parallel execution at pathological
   M boundaries.
7. **hypothesis — low confidence until privileged observation:** TLB pressure,
   software-prefetch benefit, AVX-512 frequency effects, and exact cache-stall
   shares remain unresolved.

### Evidence-gated Milestone 7 sequence

1. **proposed —** Add a private prevalidated full-tile entry point that avoids
   edge-buffer initialization; preserve the checked edge wrapper.
2. **proposed —** Add private ISA-specific kernel configurations and prototype
   AVX-512 8x16 and 4x32 plus one evidence-backed alternate AVX2 geometry.
3. **proposed —** Add 1/2/4-row no-pack or lightly packed small-M paths with
   complete tails and no hidden memory.
4. **proposed —** Add explicit parallel prepacked-B execution that reuses the
   existing caller-owned identity/provenance contract.
5. **proposed —** Add deterministic M/N/2-D task geometry, shared-panel
   ownership, physical-core-first placement, and task-wave-aware costing.
6. **proposed —** Add a small shape/ISA-specific blocking family only after
   interleaved calibration and frozen-holdout evidence.
7. **proposed —** Evaluate K unrolling, software-pipelined loads, panel
   prefetch, and panel double-buffering only after higher-confidence structural
   changes, with no-prefetch/no-pipeline controls.
8. **proposed —** Use the authenticated schema-v6 matrix and its frozen
   calibration/holdout partition as the before-state. Every Milestone 7 change
   requires a new homogeneous paired collection and independent fairness
   review before any native-BLAS parity claim.

### Explicitly unsupported conclusions

- **measured limitation —** Milestone 6 does not establish native BLAS parity.
- **measured limitation —** It does not establish hardware IPC, cache/TLB
  miss rates, branch behavior, stalled cycles, physical memory traffic, or
  workload frequency.
- **measured limitation —** It does not establish AVX-512 downclocking, an
  optimal MR/NR/MC/NC/KC tuple, universal prefetch benefit, or physical
  multi-node NUMA behavior.
- **source-backed limitation —** It does not add or freeze public GEMV/GEVM,
  packed-object, planner, microkernel, topology, or execution-context APIs.
- **proposed —** Public API/ABI/backend-contract decisions remain inputs to a
  later freeze milestone after native performance changes are validated.

## Upstream reference ledger

- **source-backed —** OpenBLAS evidence is pinned to tag `v0.3.32`, commit
  `8cecf899e21d99f9d8766ed34bfeeb3e2992c844`; relevant immutable links are
  listed in the [BLAS architecture audit](audits/blas-architecture-comparison.md#immutable-upstream-references).
- **source-backed —** BLIS evidence is pinned to tag `2.1`, commit
  `caf0db6be1202d9c83c79b51e75eceb96aa8b556`; relevant immutable links are
  listed in the same audit.
- **source-backed —** LLVM scheduling evidence is pinned to
  `llvmorg-21.1.8`, commit
  `2078da43e25a4623cab2d0d60decddf709aaea28`, including the
  [`llvm-mca` documentation](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/CommandGuide/llvm-mca.rst).
- **source-backed —** AMD host facts use the
  [Ryzen AI 9 HX 370 product specification](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html)
  and [Zen 5 software optimization guide, publication 58455 revision 1.00](https://docs.amd.com/v/u/en-US/58455_1.00).
- **source-backed —** GEMV/GEVM semantic comparison uses Netlib LAPACK 3.12.1
  [SGEMV](https://www.netlib.org/lapack/explore-html/d7/dda/group__gemv_ga0d35d880b663ad18204bb23bd186e380.html)
  and [CBLAS declarations](https://www.netlib.org/lapack/explore-html/de/da0/cblas_8h_source.html#l00201).
