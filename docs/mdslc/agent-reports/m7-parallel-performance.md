# Milestone 7 parallel-performance lane

Date: 2026-07-26

Ownership: `cpu_parallel_gemm.{h,cpp}`, its focused runtime test, and this
report. The lane was developed in an isolated worktree from `6a26994849aadf`
and did not modify the shared integration worktree.

## Finding

The retained Milestone 6 source audit was correct that complete native-parallel
execution prepared the full packed-B image on the submitting thread before
waking the persistent workers. A bounded probe made the shape dependence
concrete:

- For `1024x1024x1024`, single-thread transient and prepacked AVX2 medians were
  16.553 ms and 16.242 ms. Packing was not a dominant region.
- For short-wide `64x4096x4096`, the same medians were 24.088 ms and
  16.730 ms. Repacking B accounted for a large removable share of the
  complete-call interval.
- Parallelizing B preparation for every large B image was rejected. In an
  initial ABBA experiment it slowed the 1024-square AVX2 path from about
  8.0 ms to 11.0 ms. The cache/barrier cost exceeded the small serial packing
  share.

The integration review narrowed the accepted policy twice after a boundary
matrix exposed non-robust cells. Cooperative preparation now requires more
than one worker, `M <= 64`, `N >= 4096`, `K >= 1024`, a packed image of at
least 4 MiB, and more than one NC panel. It then admits only either
`K >= 4096`, or the very-wide boundary `M <= 32 && N >= 8192`. The rejected
`M=128` and `32x4096x1024` cells retain serial B preparation. This is a
measured short-wide rule, not a universal packing policy.

## Implementation

- Existing caller-owned workspace remains unchanged: one shared packed-B image
  followed by isolated per-worker A slices. No allocation or hidden copy was
  introduced.
- Persistent workers pack disjoint NC panels directly into their deterministic
  final offsets. A release/acquire phase barrier publishes the complete
  read-only image before any worker computes; an abort flag prevents a future
  packing invariant failure from stranding peers at the barrier.
- Packing and compute share one execution-context submission. The callback
  preserves the existing cyclic output-task assignment, K is not split, and C
  ownership remains disjoint.
- Serial AVX2, serial AVX-512, and cooperative preparation use one private
  `cpu_packed_b_format.h` contract for block layout, view construction,
  dimensions, extent, alignment, and provenance. Parallel execution validates
  that common metadata before submitting work.
- The private runtime report exposes `packed_b_threads`; zero means the
  submitting thread used the existing serial preparation path.
- Preflight workspace, alignment, overlap, ISA, thread-count, and nesting
  failures still happen before output mutation or worker submission.

## Correctness and race validation

The focused test activates both AVX2 and AVX-512 at `32x8192x1024`, compares
each with the independent double-precision oracle, checks output guards,
checks `packed_b_threads == 4`, and proves packing did not add a second pool
submission. Separate AVX2 `65x4096x1024` and AVX-512 `32x4096x1024`
boundaries authenticate `packed_b_threads == 0`. The existing packed-backend
tests continue to validate KC/NC tails and malformed prepacked views through
the same shared format helper.

| Configuration | Result |
| --- | --- |
| Release, Clang 21.1.8, OpenBLAS 0.3.32 | `runtime.cpu.parallel_packed.v1` passed |
| Debug, OpenBLAS disabled | 1/1 passed |
| ASan + UBSan Debug, OpenBLAS disabled | 1/1 passed; no sanitizer diagnostic |
| TSan Debug `-O1 -g -fsanitize=thread` | 1/1 passed; no race or deadlock diagnostic |

TSan used
`TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1`.
ASan/UBSan used `detect_leaks=1`, `halt_on_error=1`, and stack traces.

## Bounded ABBA evidence

Both binaries were clean Release builds. Baseline was `6a26994849aadf`; the
accepted implementation was `a008a57e84af`. Each process was constrained with
`taskset -c 0-3`; the native context used compact affinity, four physical
workers, 3 warmups, 7 measured samples, a 2 ms timer floor, hot cache, complete
packing, and the benchmark guard. The host was an AMD Ryzen AI 9 HX 370
(12 physical cores, 24 logical CPUs, one NUMA node). Frequency was dynamic, so
these are bounded host diagnostics rather than universal calibration data.

| Shape / ISA | Baseline median | New median | Baseline GFLOP/s | New GFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| `64x4096x4096`, AVX2 | 22.020 ms | 12.253 ms | 97.52 | 175.26 |
| `64x4096x4096`, AVX-512 | 20.803 ms | 12.366 ms | 103.23 | 173.66 |
| `1024x1024x1024`, AVX2 | 8.946 ms | 8.990 ms | 240.06 | 238.87 |
| `1024x1024x1024`, AVX-512 | 8.011 ms | 8.691 ms | 268.07 | 247.10 |

Every short-wide B sample was faster than every corresponding baseline sample.
The retained median improvements were 1.80x for AVX2 and 1.68x for AVX-512.
The square path does not activate cooperative packing; its noisier AVX-512
samples are not evidence of a code-path effect and no square performance claim
is made.

A separate current-commit S-P-P-S interleave for the short-wide shape measured:

| ISA | One-thread packed | Four-thread parallel | Speedup |
| --- | ---: | ---: | ---: |
| AVX2 | 25.654 ms, 83.71 GFLOP/s | 12.007 ms, 178.85 GFLOP/s | 2.14x |
| AVX-512 | 24.188 ms, 88.78 GFLOP/s | 13.305 ms, 161.40 GFLOP/s | 1.82x |

The AVX-512 samples had visibly larger spread. Neither result meets the
Milestone 7 four-thread 3.0x target. Cooperative B packing removes one
authenticated serial bottleneck; it does not establish native/OpenBLAS parity
or solve remaining microkernel, A-packing, placement, and scheduling limits.

## Integration review and final boundary calibration

An independent review rejected the initial broad activation rule and the
duplicated format/provenance implementation. Commits `648ef7d`, `4462677`, and
`4719528` introduced the shared format contract, both-ISA activation tests,
abortable phase barrier, and the final measured gate described above.

The final clean-commit ABBA confirmation used the same guarded complete-call
contract with 3 warmups and 7 samples. Four-thread cells were constrained to
logical CPUs 0-3 and 12-thread cells to logical CPUs 0-11. Baseline remained
the pre-optimization `6a26994849aadf` build. Ratios below are
`candidate_GFLOP/s / baseline_GFLOP/s`, each formed from the median of the two
outer or inner ABBA observations.

| Retained cell | 4 threads | 12 threads |
| --- | ---: | ---: |
| `32x8192x1024`, AVX2 | 1.118x | 1.438x |
| `64x4096x4096`, AVX2 | 1.715x | 1.879x |
| `32x8192x1024`, AVX-512 | 1.186x | 1.526x |
| `64x4096x4096`, AVX-512 | 1.720x | 1.809x |

The retained-cell median was 1.621x and the minimum was 1.118x on this host.
The rejected boundary matrix included `M=128` cells and
`32x4096x1024`; observed ratios ranged down to 0.816x, which is why those
regions do not activate the optimization. These are host-bounded calibration
results, not a claim for other processors.

Final raw JSON is untracked outside every Git worktree under
`/var/tmp/MatcoreDSL-m7-cooperative-b-final-clean.4VqD1X`.

## Handoff

Initial implementation commit: `a008a57e84af17bef7113b108d34141f8a7e3ed7`.
Integrated implementation commits: `d13d264`, `648ef7d`, `4462677`, and
`4719528`.

The integration owner should rerun the complete Release/Debug/sanitizer and
Windows matrix after resolving overlap with concurrent planner/runtime
diagnostic work. In particular, `CpuParallelGemmReportV1` is private to the
standalone runtime, but the new diagnostic field may require a straightforward
conflict resolution if that struct changed concurrently.
