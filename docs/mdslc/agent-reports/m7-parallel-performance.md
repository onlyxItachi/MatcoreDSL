# Milestone 7 parallel-performance lane

Date: 2026-07-26

> Historical lane report: the measurements and implementation narrative below
> describe candidate checkpoint
> `4719528354575f5aff74def97b534e763cb2033c`. Final independent review found
> that one retained shape did not justify the broad production activation
> region. The final branch keeps the infrastructure private but makes
> cooperative B preparation dormant; its focused test now authenticates
> `packed_b_threads == 0`.

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

The integration review narrowed the accepted policy after a boundary matrix
exposed non-robust cells. Completion review then found that the extra
`M <= 32 && N >= 8192` branch had been chosen from a shape labeled as holdout
by the later-frozen methodology. That branch was removed rather than
retrofitting the evidence split. At candidate checkpoint `4719528`,
cooperative preparation required more than one worker, `M <= 64`, `N >= 4096`,
`K >= 4096`, a packed image of at least 4 MiB, and more than one NC panel. The
declared calibration shape `64x4096x4096` supported only one point inside that
region, which is why final review disabled production selection.

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

At the candidate checkpoint, the focused test activated both AVX2 and AVX-512
at `8x4096x4096`, compared
each with the independent double-precision oracle, checks output guards,
checked `packed_b_threads == 4`, and proved packing did not add a second pool
submission. The former `32x8192x1024` experimental branch was an explicit
excluded boundary for both ISAs and authenticated `packed_b_threads == 0`.
The final branch instead requires `packed_b_threads == 0` for every production
case until a complete boundary matrix exists.
The existing packed-backend tests continue to validate KC/NC tails and
malformed prepacked views through the same shared format helper.

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

An intermediate clean checkpoint
`a008a57e84af17bef7113b108d34141f8a7e3ed7` produced this S-P-P-S
interleave for the short-wide shape:

| ISA | One-thread packed | Four-thread parallel | Speedup |
| --- | ---: | ---: | ---: |
| AVX2 | 25.654 ms, 83.71 GFLOP/s | 12.007 ms, 178.85 GFLOP/s | 2.14x |
| AVX-512 | 24.188 ms, 88.78 GFLOP/s | 13.305 ms, 161.40 GFLOP/s | 1.82x |

The AVX-512 samples had visibly larger spread. Neither result reaches 3.0x on
this diagnostic cell. Later runtime changes mean these numbers are not
final-checkpoint evidence and cannot decide the formal large-square
four-thread aggregate. Cooperative B packing removes one authenticated serial
bottleneck; it does not establish native/OpenBLAS parity or solve remaining
microkernel, A-packing, placement, and scheduling limits.

## Integration review and boundary evidence

An independent review rejected the initial broad activation rule and the
duplicated format/provenance implementation. Commits `648ef7d`, `4462677`, and
`4719528` introduced the shared format contract, both-ISA activation tests,
and abortable phase barrier. A later completion audit rejected the exact
holdout-derived threshold that those commits had retained.

The final clean-source ABBA confirmation used the same guarded complete-call
contract with 3 warmups and 7 samples. External benchmark receipts identify
candidate `4719528354575f5aff74def97b534e763cb2033c` and baseline
`6a26994849aadf738910e18a0cebb66ea9b238dc`; the raw JSON remains untracked.
Four-thread processes were constrained to logical CPUs 0-3 and 12-thread
processes to logical CPUs 0-11; individual native workers were not pinned.
Ratios below are
`candidate_GFLOP/s / baseline_GFLOP/s`, each formed from the median of the two
outer or inner ABBA observations.

| Diagnostic cell | 4 threads | 12 threads |
| --- | ---: | ---: |
| `32x8192x1024`, AVX2 | 1.118x | 1.438x |
| `64x4096x4096`, AVX2 | 1.715x | 1.879x |
| `32x8192x1024`, AVX-512 | 1.186x | 1.526x |
| `64x4096x4096`, AVX-512 | 1.720x | 1.809x |

The `64x4096x4096` calibration cell remains valid support for the retained
`K >= 4096` rule: 4-/12-thread gains were 1.715x/1.879x for AVX2 and
1.720x/1.809x for AVX-512. The `32x8192x1024` rows are preserved above only
to disclose the experiment chronology; they are not accepted as calibration
or blind holdout evidence and their production threshold branch was removed.
The broader rejected matrix observed ratios down to 0.816x. These are
host-bounded diagnostics, not a claim for other processors.

All four retained `64x4096x4096` cell-median point estimates favored the
candidate. No geometric mean, confidence interval, or sample-level win/loss
claim is available, and this is not a direct comparison with Milestone 5 or
the final Milestone 7 checkpoint.

Final raw JSON is untracked outside every Git worktree under
`/var/tmp/MatcoreDSL-m7-cooperative-b-final-clean.4VqD1X`.

## Handoff

Initial implementation commit: `a008a57e84af17bef7113b108d34141f8a7e3ed7`.
Integrated implementation commits before the methodology correction:
`d13d264`, `648ef7d`, `4462677`, and `4719528`.

The integration owner should rerun the complete Release/Debug/sanitizer and
Windows matrix after resolving overlap with concurrent planner/runtime
diagnostic work. In particular, `CpuParallelGemmReportV1` is private to the
standalone runtime, but the new diagnostic field may require a straightforward
conflict resolution if that struct changed concurrently.
