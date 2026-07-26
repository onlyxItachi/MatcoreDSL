# GEMM data-movement and packing audit

Date: 2026-07-26

Status: Milestone 6 audit evidence. This report does not change production
blocking, packing, planner selection, or the public ABI.

## Claim vocabulary

- **measured**: observed by a guarded benchmark or direct machine query on the
  declared host.
- **derived**: calculated from the reviewed source, dimensions, and documented
  machine data.
- **source-backed**: directly established by the cited implementation.
- **hypothesis**: plausible but not established by the available measurements.
- **proposed**: a candidate experiment or implementation change, not an
  accepted production decision.

## Scope and evidence identity

- **source-backed** — The production implementation was audited from the clean
  post-Milestone-5 baseline `951239f1bee5541a4cf5ad72fab2192de07cf89d`.
  Subsequent Milestone 6 commits present while this report was written changed
  benchmark instrumentation and documentation, not the packed GEMM runtime
  reviewed here.
- **measured** — The physical host is an AMD Ryzen AI 9 HX 370 with 12 physical
  cores, 24 logical CPUs, one NUMA node, a 48 KiB 12-way L1D and 1 MiB 16-way
  L2 per core. CPU 0 belongs to a 16 MiB LLC group. The governor was
  `performance`, boost was enabled, and frequency was not fixed.
- **measured** — Exact-main packing-mode probes used the clean Release
  `matcore-bench` from `951239f`, Clang 21.1.8, OpenBLAS 0.3.32 pthread, and
  `taskset -c 0`.
- **measured** — Repeated blocking and packing probes used three clean,
  local-only experimental commits derived from `951239f`:
  `01d6b7af8a71b92ce38c5b43f5a65b9428346437` (restored production
  `MC=128,NC=256,KC=256`),
  `e7e3b0815fb4c6915c60e4195e78084d2a90810a` (`NC=512`), and
  `2923592190360332ff138336848200de6c380040` (`KC=128`). These commits and
  binaries are disposable experiments and are not candidates for publication.
- **measured** — The retained packing comparison contains 30 guarded schema-v4
  JSON files. The SHA-256 of the sorted content-digest list is
  `e3487827d2699b0bb8b3ec2acb3fd36ece32831912428fbb25a83ef161a73d8b`.
  The interleaved blocking sweep contains 60 guarded files with digest
  `1ae9ff48b1e2008aa2df79c82ccf639196bf00772f1420c6e80f53068a5ea31f`.
  Raw JSON remains outside Git.
- **measured** — Every retained result passed the benchmark's independent
  double-precision oracle and `--guard`. The repeated probes used one thread,
  CPU 0, hot cache, caller-reused workspace, 64-byte matrix alignment, three
  warmups where noted, and seven or nine measured samples per process.
- **measured** — Some interleaved runs had severe low-frequency or competing-load
  outliers (for example, 1024-cube baseline run medians ranged from 68.1 to
  134.7 GFLOP/s). Medians and full ranges are therefore reported; sub-percent
  blocking differences are not treated as decisions.
- **hypothesis** — Hardware counters would help separate cache, TLB, and memory
  effects, but this lane did not use them. Conclusions below do not infer cache
  misses or TLB events from elapsed time alone.

Representative complete-call command:

```sh
LD_LIBRARY_PATH=BUILD/lib taskset -c 0 BUILD/bin/matcore-bench \
  --m M --n N --k K \
  --variant cpu.native-packed.avx2-fma.f32.v1 \
  --threads 1 --allow-smt --affinity none \
  --warmup 3 --iterations 9 --hot-cache --include-packing \
  --reuse-workspace --alignment 64 --guard \
  --json-out RAW_EXTERNAL_PATH
```

## Current packed-kernel contract

| Item | Current value | Evidence |
| --- | ---: | --- |
| `MR` | 4 rows | **source-backed** — `compiler/lib/runtime/cpu_gemm_backend.h:21` |
| `NR` | 16 columns | **source-backed** — `compiler/lib/runtime/cpu_gemm_backend.h:22` |
| `MC` | 128 rows | **source-backed** — `compiler/lib/runtime/cpu_gemm_backend.h:23` |
| `NC` | 256 columns | **source-backed** — `compiler/lib/runtime/cpu_gemm_backend.h:24` |
| `KC` | 256 depth | **source-backed** — `compiler/lib/runtime/cpu_gemm_backend.h:25` |
| workspace alignment | 64 bytes | **source-backed** — `compiler/lib/runtime/cpu_gemm_backend.h:13` |
| AVX2 register tile | 4x16, eight YMM accumulators | **source-backed** — `compiler/lib/runtime/cpu_packed_avx2.cpp:339-365` |
| AVX-512 register tile | 4x16, four ZMM accumulators | **source-backed** — `compiler/lib/runtime/cpu_packed_avx512.cpp:358-370` |

- **source-backed** — A is packed as consecutive
  `[micro-row-of-4][k][row-lane]` panels and missing M lanes are zero-filled
  (`cpu_packed_avx2.cpp:178-197`).
- **source-backed** — B is packed as consecutive
  `[micro-column-of-16][k][column-lane]` panels and missing N lanes are
  zero-filled (`cpu_packed_avx2.cpp:199-219`).
- **source-backed** — The AVX-512 path deliberately reuses the AVX2 packing
  format, dimensions, and workspace contract
  (`cpu_packed_avx512.cpp:27-30`, `314-324`).
- **source-backed** — A packed B micro-panel advances by 16 floats per K step.
  Because the workspace begins at 64-byte alignment, each AVX2 YMM and AVX-512
  ZMM B load is aligned (`cpu_packed_avx2.cpp:350-352`,
  `cpu_packed_avx512.cpp:364-366`).
- **source-backed** — A uses scalar broadcasts, while output uses unaligned-safe
  loads and stores. There is no separate aligned-output kernel
  (`cpu_packed_avx2.cpp:339-375`).
- **source-backed** — The production loop order is `NC -> KC -> MC`. B is packed
  once for an `(NC,KC)` panel and reused over every M block; A is packed once
  per `(MC,KC)` block for every NC panel
  (`cpu_packed_avx2.cpp:496-506`).
- **source-backed** — No large buffer is allocated internally. Workspace size
  is computed with checked arithmetic and contains one padded `MC x KC` A block
  plus, for transient execution, one padded `KC x NC` B block
  (`cpu_gemm_backend.cpp:83-148`).

## Cache-scale working sets

- **derived** — One AVX2 microkernel step over a full `KC=256` panel consumes a
  4 KiB packed-A micro-panel and a 16 KiB packed-B micro-panel. Their 20 KiB
  total fits in the host's 48 KiB L1D.
- **derived** — The largest transient packed macro buffers are 128 KiB for A and
  256 KiB for B, or 384 KiB total. They fit in the host's 1 MiB private L2, but
  source A/B, output C, code, and other runtime data compete for the remainder.
- **derived** — Under ideal set distribution, the packed micro-panels occupy
  approximately five lines per L1 set (one A plus four B) in a 12-way cache.
  The two macro buffers occupy approximately six lines per L2 set in a 16-way
  cache. This arithmetic does not establish real replacement behavior.
- **hypothesis** — The baseline constants are cache-conservative rather than
  obviously oversized on CPU 0. Conflict behavior can still differ with
  allocator addresses and source/output strides; counters or address-controlled
  experiments are required to prove it.
- **derived** — The transient packed workspace spans at most 96 ordinary 4 KiB
  pages. A full prepacked 1024x1024 B spans 1,024 pages (4 MiB), and a
  4096x4096 B spans 16,384 pages (64 MiB).
- **derived** — Packing a `KC x NC` source-B block with a large row stride can
  touch as many as one source page per K row even though the destination is only
  64 contiguous pages. Analogous row-stride effects apply to source A and C.
- **hypothesis** — Source-panel page dispersion and full-prepacked-B scans may
  create DTLB pressure on large-stride rectangular problems. No DTLB counter
  evidence is available in this lane, so this remains a testable hypothesis.

## Packing-volume model

- **derived** — For dimensions divisible by `MR`, the transient A packed-write
  volume is approximately
  `ceil(N/NC) * M * K * sizeof(float)`. M tails replace each 128-row block's
  actual row count with `round_up(rows,MR)`.
- **derived** — B is read once per call and the packed-write volume is
  `round_up(N,NR) * K * sizeof(float)`. B is not repacked for each M block.
- **derived** — Single-thread prepacked-B execution removes per-call B packing
  but retains exactly the same A packing and compute loop
  (`cpu_packed_avx2.cpp:511-570`).
- **derived** — A is repacked `ceil(N/NC)` times. That is conventional for the
  chosen `NC -> KC -> MC` loop order, but it is increasingly expensive as N
  grows and M/K shrink.

| Shape | A packed writes per transient call | B packed writes per transient call | Padded FMA / useful FMA | Evidence |
| --- | ---: | ---: | ---: | --- |
| 31x33x35 | 4.375 KiB | 6.563 KiB | 1.501x | **derived** |
| 128x128x128 | 64 KiB | 64 KiB | 1.000x | **derived** |
| 512x512x512 | 2 MiB | 1 MiB | 1.000x | **derived** |
| 1024x1024x1024 | 16 MiB | 4 MiB | 1.000x | **derived** |
| 1x4096x4096 | 1 MiB | 64 MiB | 4.000x | **derived** |
| 8x4096x4096 | 2 MiB | 64 MiB | 1.000x | **derived** |
| 64x1024x1024 | 1 MiB | 4 MiB | 1.000x | **derived** |
| 1024x64x1024 | 4 MiB | 256 KiB | 1.000x | **derived** |

- **derived** — The M=1 path writes four packed A rows and executes all four
  accumulator rows, then commits only one. It performs four times the useful
  FMAs even before packing cost is considered.
- **derived** — A 31x33 output executes a 32x48 padded register-tile envelope,
  about 50.1% more FMAs than the mathematical output requires. The edge path
  also stages partial tiles through a 4x16 stack buffer
  (`cpu_packed_avx2.cpp:320-337`, `377-381`).
- **source-backed** — K is not padded, but every `KC` panel after the first
  reloads C and every panel stores it (`cpu_packed_avx2.cpp:241-244`,
  `339-347`, `368-375`).
- **derived** — With `q=ceil(K/KC)`, the microkernel issues `q` output writes and
  `q-1` output reads, or `2q-1` logical C pass-equivalents. This is not a DRAM
  traffic claim because C may remain in cache.

## Measured packing cost

- **measured** — The table reports the median of three independent process
  medians from the clean local production-equivalent build. Both modes include
  A packing and complete compute. `prepacked-B` prepares caller-owned B storage
  before timing, so the difference estimates repeated-execution B-pack cost,
  not one-shot end-to-end cost.

| Shape | Transient A+B | Prepacked B | Time removed | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 1x4096x4096 | 8.413 ms | 1.389 ms | 83.5% | **measured** — B packing dominates this repeated row-vector-like call |
| 8x4096x4096 | 9.009 ms | 2.447 ms | 72.8% | **measured** — B packing remains the largest removable component |
| 64x1024x1024 | 1.154 ms | 0.977 ms | 15.3% | **measured** — B packing is material but no longer dominant |
| 1024x1024x1024 | 15.929 ms | 15.635 ms | 1.8% | **measured** — compute and A-side work dominate repeated execution |
| 1024x64x1024 | 1.092 ms | 1.087 ms | 0.5% | **measured** — the small B panel is not a meaningful bottleneck |

- **measured** — The accepted Milestone 4 results independently showed the same
  ordering: prepacking accelerated 1x4096x4096 by 5.95x,
  8x4096x4096 by 4.11x, and 1024-cube by 1.02x
  (`docs/performance/cpu/milestone-4-single-thread-calibration-2026-07-22.md:146-158`).
- **measured** — Exact-main compute-only diagnostics showed large gaps for
  M=1/M=8 wide shapes, moderate gaps for tails, and small gaps for large
  squares. These values are deliberately not tabulated as packing time because
  compute-only also removes KC-panel C reload/stores and changes packed
  working-set residency (`compiler/tools/matcore-bench/planner_runner.cpp:1082-1120`).
- **derived** — The measured aspect-ratio trend matches the volume model:
  1x4096x4096 writes 64 MiB of packed B for only 1 MiB of padded packed A,
  whereas 1024x64x1024 writes 256 KiB of packed B and 4 MiB of packed A.

## Static blocking sweep

- **measured** — Baseline, `NC=512`, and `KC=128` executables were interleaved
  in rotating order. Each cell is the median of five independent run medians;
  parentheses show the full range. All values are useful-operation GFLOP/s.

| Shape | Baseline 128/256/256 | NC=512 | KC=128 |
| --- | ---: | ---: | ---: |
| 8x4096x4096 | 28.25 (14.25–30.01) | 26.54 (25.07–28.45) | 28.69 (11.18–30.66) |
| 64x1024x1024 | 115.97 (114.02–118.14) | 116.83 (115.69–117.39) | 117.11 (116.15–117.47) |
| 1024x64x1024 | 122.20 (107.03–123.59) | 122.21 (116.04–122.82) | 122.51 (105.70–123.12) |
| 1024x1024x1024 | 133.86 (68.08–134.66) | 135.20 (128.29–136.51) | 131.17 (63.48–132.69) |

- **measured** — Neither candidate produced a consistent improvement above
  measurement variation. `NC=512` was about 1% faster at 1024-cube but 6%
  slower at 8x4096x4096; `KC=128` was about 1% faster on the two 64-wide
  fixtures but 2% slower at 1024-cube.
- **measured** — Unpaired pilots with `MC=256`, `KC=384`, and combined
  `MC=256,NC=512` did not establish a portable improvement and were excluded
  from decision evidence.
- **derived** — One fixed blocking tuple exposes conflicting objectives:
  increasing NC can reduce A repacks but enlarges B residency, while reducing
  KC shrinks the panel working set but increases C accumulation passes.
- **source-backed** — The current AVX2 and AVX-512 variants share the same
  blocking tuple even though their register files, load widths, and accumulator
  counts differ.
- **proposed** — Do not change production constants from this sweep. Milestone 7
  should test a small, versioned family of shape/ISA-specific profiles against
  calibration and holdout matrices rather than replace one universal tuple with
  another.

## Parallel packing and ownership

- **source-backed** — The parallel workspace contains one full packed B shared
  by all workers plus one packed-A workspace per actual worker
  (`cpu_parallel_gemm.cpp:169-217`).
- **source-backed** — B is packed serially on the submitting thread before any
  worker tasks are dispatched (`cpu_parallel_gemm.cpp:268-293`).
- **source-backed** — Worker tasks are only 128-row M bands. Actual parallelism
  is capped by `ceil(M/128)` regardless of N or K
  (`cpu_parallel_gemm.cpp:146-166`, `256-260`).
- **source-backed** — Each worker reads the same immutable packed-B view and
  writes a disjoint output row band using a disjoint, 64-byte-separated
  per-worker A workspace (`cpu_parallel_gemm.cpp:158-163`, `194-215`).
- **derived** — Packed B is already safely shared across workers within one
  execution; it is not duplicated per worker.
- **derived** — For 4096x4096 B, the parallel path must serially prepare 64 MiB
  before dispatch. The existing Milestone 5 4096-cube workspace is therefore
  approximately 64 MiB shared B plus 128 KiB per worker, matching the retained
  64.25–67 MiB reports
  (`milestone-5-advanced-cpu-2026-07-22.md:232-252`).
- **derived** — M=1, M=8, and M=64 can use only one native worker even when N
  and K are large. This decomposition cannot accelerate the prompt's
  vector-like and short-wide families.
- **hypothesis** — Serial full-B preparation limits scaling for calls whose
  compute is short relative to KxN packing. A dedicated pack-only timing and
  parallel call timeline are still needed to quantify the exact fraction.
- **proposed** — Add a private parallel execution surface that accepts the
  existing authenticated caller-owned prepacked-B identity. Do not add a global
  cache or infer lifetime.
- **proposed** — Evaluate two-dimensional output-tile ownership for small-M,
  large-N shapes. Any shared panel protocol must make synchronization and
  workspace visible and preserve disjoint C ownership.
- **proposed** — Compare full-B prepacking with panel-at-a-time packing plus
  synchronized consumers. Full prepacking avoids barriers but has a large
  footprint; panel streaming reduces footprint but may add synchronization.

## Ranked findings

| Rank | Finding | Confidence | Expected affected region |
| ---: | --- | --- | --- |
| 1 | M/NR tail padding performs work for nonexistent output lanes | **source-backed + derived: high** | M=1 and general M/N tails; potentially very high impact |
| 2 | Per-call B packing dominates small-M, large-N/K transient execution | **measured: high** | row-vector-like and repeated-weight workloads; very high impact |
| 3 | Parallel execution serially packs full B and has no cross-call parallel-prepack path | **source-backed: high** | large B, repeated calls, higher thread counts; high expected impact |
| 4 | Parallelism partitions only M in 128-row bands | **source-backed: high** | small-M/large-N rectangular GEMM; high expected impact |
| 5 | A is repacked once per NC panel | **source-backed: high** | short-wide shapes; medium expected impact |
| 6 | KC panels cause repeated logical C loads/stores | **source-backed + derived: high** | large K; medium expected impact, physical traffic unmeasured |
| 7 | Full prepacked B and strided source panels create large page footprints | **derived; causal confidence low** | large KxN; TLB/cache impact remains a hypothesis |
| 8 | One common static block tuple serves all shapes and both ISAs | **source-backed: high; performance confidence medium** | cross-shape/ISA compromise; tested simple replacements did not win consistently |

## Direct audit answers

1. **measured** — OpenBLAS wins most large regular single-thread shapes in the
   accepted Milestone 4 matrix (26 of 30 overall) while native AVX2 reached
   135.33 versus 154.54 GFLOP/s at 1024-cube. Packing explains only about 1.8%
   of repeated 1024-cube time, so large-square loss is not primarily B packing
   (`milestone-4-single-thread-calibration-2026-07-22.md:82-87`, `160-164`).
2. **measured + source-backed** — Packing is primary for small-M wide calls;
   M-tail waste and M-only partitioning are structural rectangular limits.
   Microkernel throughput and scheduling remain separate audit lanes.
3. **measured** — No current MC/NC/KC value is demonstrated universally wrong.
   The controlled NC/KC sweep rejects a simplistic one-constant fix.
4. **derived** — One static tuple creates conflicting reuse and C-traffic
   tradeoffs, but shape-dependent profiles still require holdout validation.
5. **source-backed** — B is not redundantly repacked per M block. It is packed
   once per call (or once before all parallel tasks); only explicit
   single-thread prepacked-B avoids cross-call repacking.
6. **source-backed** — B is already immutable and shared across native workers.
   Cross-call sharing is safe only through explicit caller-owned identity and
   lifetime.
7. **derived** — Packing should be avoided or amortized at least for measured
   M=1/M=8, N=K=4096 calls. This evidence does not define a universal numeric
   threshold.
8. **proposed** — Prioritize private 1/2/4-row no-pack or lightly packed paths,
   explicit parallel prepacked-B reuse, and N-aware task decomposition before
   speculative prefetch or huge-page policy.

## Milestone 7 experiments

- **proposed** — Implement and force-test 1xNR, 2xNR, and 4xNR small-M kernels
  that avoid padded accumulator rows; retain complete tail and alias checks.
- **proposed** — Add benchmark phases that time B preparation alone, A packing
  alone, and complete execution without comparing those diagnostics to CBLAS
  complete calls.
- **proposed** — Add an explicit private parallel-prepacked-B execution path
  using caller-owned storage, provenance, size bound, and invalidation.
- **proposed** — Test 2-D M/N task decomposition with disjoint output tiles and
  compare dispatch/barrier cost on short-wide holdouts.
- **proposed** — Calibrate multiple blocking profiles with alternating AB/BA
  execution and separate calibration/holdout shapes. Reject profiles whose
  gains are below run-to-run variability.
- **proposed** — Collect DTLB/cache counters only when the host permits a
  documented, reproducible counter backend; until then, keep page-pressure
  conclusions labeled as hypotheses.
- **proposed** — Keep all optimizations internal or additive. Do not create a
  hidden packed-B cache, hidden allocation, or new public GEMM ownership
  assumption before the later API/ABI freeze milestone.
