# Native BLAS parity v1: bounded Milestone 7 disposition

Status date: 2026-07-26

Verdict: **partially passed**

This report is the authoritative Milestone 7 disposition for the validated
Linux x86-64 host. It does not claim general BLAS parity. The implementation,
correctness, artifact, ABI, package, and sanitizer gates pass, and a measured
short-wide packing bottleneck improved materially. The complete declared
native/OpenBLAS parity envelope was not established.

## Scope and evidence identity

- Operation: F32 row-major contiguous GEMM, overwrite output, F32
  accumulation.
- Host: AMD Ryzen AI 9 HX 370, 12 physical cores, 24 logical CPUs, one NUMA
  node.
- Operating system: Ubuntu Linux, x86-64.
- Compiler/tooling: Clang/LLVM 21.1.8, Release and Debug.
- External provider: OpenBLAS 0.3.32, pthread, LP64 CBLAS.
- Implementation and measurement-contract checkpoint:
  `2863253d1f2b06a943c2028ae298d0381d15ddf4`.
- Exact Release benchmark SHA-256:
  `61028379491877db4383cc73359dcce009e0d96d36f85ad471ae30ceab1fb8b9`.
- Runner SHA-256:
  `ec57c6a2876bb2b492ad9f716818dd78fc1509a55319f8066119e2dd55292766`.
- Summarizer SHA-256:
  `9a13e5cbc09db2250c98a2f537b7f3debd94f6549f06501399e0268a7506d67e`.
- Frozen forward plan SHA-256:
  `c050c6a8d700abdc89f6a7aaee6898b510948cf59cc1f2c831b8542b7b1c09fc`.

The benchmark binary authenticated a clean Git worktree and exact tracked
source commit. Benchmark schema v6 and parity manifest v3 distinguish complete
calls, allocation-excluded execution, packing, prepacked-B, diagnostics,
thread placement, and independent correctness replay.

## Implemented and retained changes

1. The private AVX-512 full-tile body widened from one 4x16 packed-B panel to
   two adjacent panels (MR=4, NR=32). The Release symbol uses 14 distinct ZMM
   registers and eight packed-FMA sites with no vector stack spill.
2. Parallel GEMM task planning represents M-only, N-only, and two-dimensional
   output grids. Planner diagnostics report row tasks, column tasks, total
   task capacity, effective thread ceiling, and capacity limiting.
3. The persistent parallel executor can cooperatively prepare disjoint final
   packed-B panels within the existing caller-owned workspace. A publication
   barrier makes the completed image read-only before computation.
4. Cooperative B preparation is deliberately restricted to the measured
   short-wide envelope: more than one worker, `M <= 64`, `N >= 4096`,
   `K >= 4096`, packed B at least 4 MiB, and more than one NC panel.
5. AVX2 gained a private prevalidated 4x16 full-tile symbol for parallel use,
   but independent complete-call evidence rejected routing the serial executor
   through it. Serial routing was restored to the checked entry.
6. Exact AVX2 and AVX-512 artifact gates now inspect the production full-tile
   symbols and reject optimized vector stack spill/reload.
7. The parity runner and deterministic summarizer fail closed on provenance,
   identity, timing-scope, thread-capacity, result-coverage, and raw-digest
   mismatches.

No public C ABI symbol was removed or changed. There is no hidden allocation,
implicit copy, global transformed-operand cache, K split, silent provider
fallback, or runtime autotuning.

## Bounded measured improvements

The retained ABBA experiment compares the pre-optimization baseline
`6a26994849aadf` with the reviewed cooperative-packing implementation. Raw JSON
remains external and untracked. Runs used guarded complete calls, packing
included, allocation excluded, caller-owned reused workspace, compact native
placement, and exact forced native variants.

For `64x4096x4096`, the candidate/baseline throughput ratios were:

| Native variant | 4 threads | 12 threads |
| --- | ---: | ---: |
| packed AVX2/FMA | 1.715x | 1.879x |
| packed AVX-512/FMA | 1.720x | 1.809x |

A separate current-implementation S-P-P-S diagnostic measured:

| Native variant | 1-thread complete call | 4-thread complete call | Speedup |
| --- | ---: | ---: | ---: |
| AVX2/FMA | 25.654 ms, 83.71 GFLOP/s | 12.007 ms, 178.85 GFLOP/s | 2.14x |
| AVX-512/FMA | 24.188 ms, 88.78 GFLOP/s | 13.305 ms, 161.40 GFLOP/s | 1.82x |

These measurements establish removal of one serial short-wide B-packing
bottleneck. They do not meet the declared 3.0x four-thread target and do not
establish native/OpenBLAS parity.

The attempted serial AVX2 full-tile promotion was 0.62--2.47% slower on five
stable representative complete-call cells and was not retained as a
performance change. This negative result is part of the Milestone 7 evidence,
not omitted from the envelope.

## Complete-matrix collection disposition

The frozen matrix contains 368 cases in each stable order. Multiple fresh
forward attempts were stopped by the interference guard when unrelated
multi-process compiler, pytest, checksum, and physical-device workloads became
active on the shared host. An authenticated resumable forward manifest reached
258/368 cases at the exact checkpoint, but no complete forward manifest and no
matching reverse manifest were produced.

The incomplete manifest is not summarized, committed, or used for a
performance claim. There is therefore no final table of per-shape
native/OpenBLAS ratios, no complete forward/reverse scaling aggregate, and no
complete parity-envelope planner-regret aggregate for this checkpoint.

The runner's eleven-shape full-registry regret subset is diagnostic only.
Automatic and forced parity cells do not reconstruct full-registry regret for
the rest of the declared matrix. Likewise, task-capacity-limited comparisons
are exact at the actual capacity but do not establish every originally
requested four-thread and physical-core ceiling.

## Acceptance disposition

| Criterion | Disposition | Evidence |
| --- | --- | --- |
| Single-thread medium/large median native/OpenBLAS ratio >= 0.90 | not established | complete paired matrix unavailable |
| Every declared core-family median >= 0.75 or approved limitation | not established | complete paired matrix unavailable |
| Native equals or beats OpenBLAS on one meaningful family | not established | no complete matched-provider family aggregate |
| Native materially improves over Milestone 5 | passed, bounded | short-wide cooperative B packing, 1.715--1.879x AVX2 and 1.720--1.809x AVX-512 versus baseline |
| Multi-thread large-shape median native/OpenBLAS ratio >= 0.85 | not established | complete paired matrix unavailable |
| Four-thread native speedup >= 3.0x | failed on measured short-wide cell | AVX2 2.14x; AVX-512 1.82x |
| Physical-core execution improves over one thread | not established over complete envelope | bounded individual diagnostics only |
| Planner median/p95/max regret <= 1.05/1.15/1.35 | not established over full envelope | only eleven-shape diagnostic registry coverage exists |
| No automatic regret above 2.0 | not established over full envelope | incomplete full-registry coverage |
| Correctness, artifact, ABI, package, sanitizer gates | passed | exact integration report |

Because mandatory performance criteria are failed or not established, changing
the shape envelope, counting OpenBLAS-selected automatic plans as native
parity, or promoting bounded regret would be benchmark gaming. The milestone
is therefore partial.

## Supported claims

- The retained private AVX2/AVX-512 bodies contain the claimed packed-FMA ISA.
- Cooperative B preparation is correct, race-free under the tested sanitizer
  scope, allocation-free, and materially faster on the declared short-wide
  calibration cell.
- Existing native/compiler/runtime/package behavior remains correct and
  installed-consumer compatible.
- OpenBLAS remains a legal optional planner candidate and fails closed when
  unavailable.

## Claims explicitly not supported

- Native BLAS parity across the declared Milestone 7 envelope.
- A universal crossover, tile, packing, ISA, or thread policy.
- A complete planner-regret bound for all parity shapes.
- Three-times four-thread scaling.
- A cross-call packed-B cache or validated transformed-operand lifetime model.
- New Windows performance results.
- Public API/ABI/backend-contract freeze readiness.

## Reproduction on an exclusive host

Build the exact Release configuration, then run both orders without concurrent
compiler, test, checksum, benchmark, or device-validation workloads:

```text
python3 compiler/tests/performance/run_native_blas_parity.py \
  --bench BUILD/bin/matcore-bench \
  --output-dir RAW_FORWARD_OUTSIDE_GIT \
  --physical-cores 12 --suites all --case-order stable-forward

python3 compiler/tests/performance/run_native_blas_parity.py \
  --bench BUILD/bin/matcore-bench \
  --output-dir RAW_REVERSE_OUTSIDE_GIT \
  --physical-cores 12 --suites all --case-order stable-reverse

python3 compiler/tests/performance/summarize_native_blas_parity.py \
  --forward-manifest RAW_FORWARD_OUTSIDE_GIT/manifest.json \
  --reverse-manifest RAW_REVERSE_OUTSIDE_GIT/manifest.json \
  --markdown-out SANITIZED_REPORT.md \
  --json-out SANITIZED_REPORT.json
```

`--resume` is supported only after the runner authenticates the exact source,
runner blob, benchmark binary, plan, existing raw files, and their digests.
Raw reports, profiler output, and partial manifests must remain outside Git.
