# Native BLAS parity v1: bounded Milestone 7 disposition

Status date: 2026-07-26

Milestone disposition: **partially passed (manual, evidence-bounded)**

Performance acceptance: **not evaluated / not established** because no
complete authenticated forward/reverse pair exists. This is not a verdict
emitted by the parity summarizer.

This report is the candidate local Milestone 7 disposition pending hosted
gates. It does not claim general BLAS parity. The implementation, correctness,
artifact, ABI, package, sanitizer, and local independent-review gates pass.
Four bounded
short-wide diagnostic cell medians favored cooperative B preparation. The
complete declared native/OpenBLAS parity envelope was not established.

## Scope and evidence identity

- Operation: F32 row-major contiguous GEMM, overwrite output, F32
  accumulation.
- Host: AMD Ryzen AI 9 HX 370, 12 physical cores, 24 logical CPUs, one NUMA
  node.
- Operating system: Ubuntu Linux, x86-64.
- Compiler/tooling: Clang/LLVM 21.1.8, Release and Debug.
- External provider: OpenBLAS 0.3.32, pthread, LP64 CBLAS.
- Incomplete physical-run pre-final-review checkpoint:
  `2863253d1f2b06a943c2028ae298d0381d15ddf4`.
- Final locally validated code checkpoint:
  `ff483afcfc06c491deed88b8e194737940701086`.
- Exact Release benchmark SHA-256:
  `61028379491877db4383cc73359dcce009e0d96d36f85ad471ae30ceab1fb8b9`.
- Runner SHA-256:
  `ec57c6a2876bb2b492ad9f716818dd78fc1509a55319f8066119e2dd55292766`.
- Historical checkpoint summarizer SHA-256:
  `9a13e5cbc09db2250c98a2f537b7f3debd94f6549f06501399e0268a7506d67e`.
- Frozen forward plan SHA-256:
  `c050c6a8d700abdc89f6a7aaee6898b510948cf59cc1f2c831b8542b7b1c09fc`.

These hashes identify the incomplete pre-final-review sweep only. The
historical ABBA and S-P-P-S diagnostics below carry their own source
checkpoints and are not results from this benchmark binary.

Final-review commits from `0fe66e7` through `b0f5cb7` subsequently made
cooperative packing dormant, replaced summary-v2 partial semantics with the
self-authenticating summary-v3 bounded assessment, invalidated stale outputs
on rejection, authenticated the complete ordered planner registry, and
required timing validity to match legal complete-call comparability. The
reviewed v3 summarizer has SHA-256
`e92a5c7198cb7c56988a2b2ab54c5a79da1609fcf105256bfddf91ebfcde96fb`
and Git blob `191e68a2c6a8ea1954454e811ea147a690c48e08`. It intentionally rejects
the earlier incomplete receipt and must be used for any future complete
collection.

Test-only follow-up `ff483af` enlarged the synthetic first-use/steady-state
calibration gap so the benchmark contract remains deterministic when Windows
rounds short sleeps to its scheduler quantum. It does not change benchmark or
runtime production behavior.

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
3. Private persistent-executor infrastructure can cooperatively prepare
   disjoint final packed-B panels within the existing caller-owned workspace.
   A publication barrier makes the completed image read-only before
   computation.
4. Independent final review rejected the former broad activation rule because
   its full boundary matrix was not measured at the final checkpoint.
   Production selection is therefore dormant and continues to use serial
   B preparation pending new authenticated evidence.
5. AVX2 gained a private prevalidated 4x16 full-tile symbol for parallel use,
   but independent complete-call evidence rejected routing the serial executor
   through it. Serial routing was restored to the checked entry.
6. Exact AVX2 and AVX-512 artifact gates now inspect the production full-tile
   symbols and reject optimized vector stack spill/reload.
7. The parity runner and deterministic summarizer fail closed on provenance,
   identity, timing-scope, thread-capacity, result-coverage, and raw-digest
   mismatches.

No public C ABI symbol was removed or changed. MDSLC adds no hidden
packing/workspace allocation, implicit tensor copy, global
transformed-operand cache, K split, silent provider fallback, or runtime
autotuning. An opaque OpenBLAS provider may manage internal memory under its
own contract.

## Bounded measured improvements

The retained ABBA experiment compares the pre-optimization Milestone 7
checkpoint `6a26994849aadf738910e18a0cebb66ea9b238dc` with
cooperative-packing checkpoint
`4719528354575f5aff74def97b534e763cb2033c`. The exact candidate and baseline
identities were recovered from the external clean-source benchmark receipts;
the raw JSON remains external and untracked. Later changes removed an
unrelated `32x8192x1024` threshold while retaining the measured
`64x4096x4096` path. Runs used guarded complete calls, packing included,
allocation excluded, caller-owned reused workspace, exact forced native
variants, and contiguous process masks (CPUs 0--3 for four-thread cells and
0--11 for twelve-thread cells); workers were not individually pinned.

For `64x4096x4096`, the candidate/baseline throughput ratios were:

| Native variant | 4 threads | 12 threads |
| --- | ---: | ---: |
| packed AVX2/FMA | 1.715x | 1.879x |
| packed AVX-512/FMA | 1.720x | 1.809x |

These are four host-bounded cell-median point estimates, all favoring the
candidate. No geometric mean, confidence interval, or sample-level win/loss
claim is available.

An intermediate clean implementation at
`a008a57e84af17bef7113b108d34141f8a7e3ed7` produced this historical S-P-P-S
diagnostic:

| Native variant | 1-thread complete call | 4-thread complete call | Speedup |
| --- | ---: | ---: | ---: |
| AVX2/FMA | 25.654 ms, 83.71 GFLOP/s | 12.007 ms, 178.85 GFLOP/s | 2.14x |
| AVX-512/FMA | 24.188 ms, 88.78 GFLOP/s | 13.305 ms, 161.40 GFLOP/s | 1.82x |

Later packing/runtime hardening changed the implementation. These measurements
identify an unresolved scaling risk but do not establish final-checkpoint
scaling or native/OpenBLAS parity.

The attempted serial AVX2 full-tile promotion was 0.62--2.47% slower on five
stable representative complete-call cells and was not retained as a
performance change. This negative result is part of the Milestone 7 evidence,
not omitted from the envelope.

## Complete-matrix collection disposition

The frozen 12-core matrix contains 368 cases in each stable order.
Operator/external host-quiescence monitoring stopped collection when unrelated
multi-process workloads became active on the shared host. An external,
untracked authenticated resumable forward receipt at the pre-final-review
checkpoint has SHA-256
`26e75ecbcfbb19d024fa8a5fa9790b65a2deb5743b39f16a4f22dd39381cfe69`
and records 258/368 cases (`252` reused and `6` newly passed). No complete
same-checkpoint forward/reverse pair was produced.

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
| Multi-thread large-shape median native/OpenBLAS ratio >= 0.85 | not established | complete paired matrix unavailable |
| Four-thread native speedup >= 3.0x | not established at the final code checkpoint | intermediate `a008a57` diagnostic was 2.14x AVX2 / 1.82x AVX-512, below target |
| Physical-core execution improves over one thread | not established over complete envelope | bounded individual diagnostics only |
| Planner median/p95/max regret <= 1.05/1.15/1.35 | not established over full envelope | only eleven-shape diagnostic registry coverage exists |
| No automatic regret above 2.0 | not established over full envelope | incomplete full-registry coverage |
| Correctness, artifact, ABI, package, sanitizer gates | passed | exact integration report |

The bounded cooperative-packing diagnostic reported large positive point
estimates relative to the pre-optimization Milestone 7 checkpoint; improvement
over Milestone 5 and at the final code checkpoint is not established.

Because mandatory performance criteria are not established, changing
the shape envelope, counting OpenBLAS-selected automatic plans as native
parity, or promoting bounded regret would be benchmark gaming. The milestone
is therefore partial.

## Supported claims

- The retained private AVX2/AVX-512 bodies contain the claimed packed-FMA ISA.
- The private cooperative-preparation candidate is allocation-free and passed
  its historical focused correctness/race checks. Production selection is
  dormant pending a final-checkpoint boundary matrix.
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
