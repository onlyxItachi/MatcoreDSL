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

The accepted policy is therefore deliberately narrow: cooperatively prepare B
only when execution uses more than one worker, `M <= MC` (`MC=128`), the packed
image is at least 4 MiB, and more than one NC panel exists. This is the audited
short-wide region, not a universal packing rule.

## Implementation

- Existing caller-owned workspace remains unchanged: one shared packed-B image
  followed by isolated per-worker A slices. No allocation or hidden copy was
  introduced.
- Persistent workers pack disjoint NC panels directly into their deterministic
  final offsets. A release/acquire phase barrier publishes the complete
  read-only image before any worker computes.
- Packing and compute share one execution-context submission. The callback
  preserves the existing cyclic output-task assignment, K is not split, and C
  ownership remains disjoint.
- The common v1 `CpuPackedBViewV1` identity, dimensions, packed extent, and
  provenance are reconstructed before the already-existing backend validator
  consumes the view.
- The private runtime report exposes `packed_b_threads`; zero means the
  submitting thread used the existing serial preparation path.
- Preflight workspace, alignment, overlap, ISA, thread-count, and nesting
  failures still happen before output mutation or worker submission.

## Correctness and race validation

The focused test adds a `64x1040x1009` case. It crosses NC and KC tails, exceeds
the 4 MiB gate, requires four workers, compares with the independent
double-precision oracle, checks output guards, checks `packed_b_threads == 4`,
and proves packing did not add a second pool submission.

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

Raw JSON remains untracked under
`/home/hamza-usta/.tmp/m7-parallel-performance/`.

## Handoff

Implementation commit: `a008a57e84af17bef7113b108d34141f8a7e3ed7`.

The integration owner should rerun the complete Release/Debug/sanitizer and
Windows matrix after resolving overlap with concurrent planner/runtime
diagnostic work. In particular, `CpuParallelGemmReportV1` is private to the
standalone runtime, but the new diagnostic field may require a straightforward
conflict resolution if that struct changed concurrently.
