# Independent generated GEMM execution and performance controls v1

2026-09-09 local date (run began `2026-09-08T22:46:01.872478+00:00`).
**EXECUTED, bounded finding:** real row-contiguous generated source execution
passed this independent oracle and improved all six sampled end-to-end shapes
against the current scalar-schedule generated baseline. Explicit 4x8x1 masked
vectorization passed correctness but regressed two primitive shapes and lost to
the row-contiguous primitive on all six. Legal OpenBLAS remained substantially
faster on larger examples. This is not parity, a universal schedule, transformed
whole-region execution, or final integration acceptance.

## Scope and immutable identities

Canonical inspection input: `0e8c040809ae1fe6b0f1c01e41fc509b07212be7`, clean.
Harness: `9f3aed9917ee6a49d765f2681721d67b25fd5393`, clean at the complete run.
Only `compiler/experiments/hpc_execution_v1/` and this report belong to this lane.
No production edits, full build, existing-build modification, push or merge was
performed here. Existing artifacts were read-only; small host fixtures were built
in this lane's ignored output directories.

| Artifact | SHA-256 |
| --- | --- |
| Baseline Release `mdslc-region` | `eee73557f8769569f45b51e95a269237d2863fff5fb3b5146a69daa57fdc69e6` |
| Baseline generated `strict-normal.o` | `8a5334297a2b680f2df2cea28945ce773b4b250abfa567b19f313c446b50b847` |
| Row-selected Release `mdslc-region` | `f183efd71903d7a48b2c98c72bb7e722707db21a17cdea7ce3d2acdfe2518832` |
| Row-contiguous primitive | `26b5df728c64c3ae746da30373e821cba19c009ba6474f81ef056293282abc1e` |
| Advocate's masked-vector primitive | `5d472f0eaa7ffbf51fb16c1016f35ed990606f8ed18511191b2f0ba51c1aa231` |
| Shared canonical Runtime, both builds | `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014` |
| Configured OpenBLAS provider | `be2e7d119279836105e0361be92c78dbcd8a6e7357e74339bdbb32a5195ee35e` |

Baseline artifacts came from `MatcoreDSL-wt-private-candidate-dso-v1/build-isolated-release`,
whose built implementation is `b0e7ac9ac67c3c314a44e5db108be390485404eb`.
Production compiler paths are unchanged between that implementation and this
lane's harness HEAD. This is production-source-equivalent baseline reuse, not a
claim that these artifacts were freshly rebuilt at the documentation/main HEAD.

The row build is `MatcoreDSL-wt-mlir-hpc-campaign-v1/build-hpc-release`, explicitly
configured with `MDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE=row-contiguous`.
At evidence capture its source HEAD was `82db9a04ef3c224bbf3af539474a281bbf6be1d2`
with an uncommitted compiler diff, captured completely in the command journal;
diff SHA-256 `78c838a63b5e313af027f840e07ae883d3c015fc61c4b54ac1f9016bf22cb6b4`.
The frozen driver/object above identify what ran; this is not mislabeled as a
clean committed candidate build. The vector artifact was supplied at
`/tmp/mdslc-mlir-hpc.kr7V2m/research.o`; it has no product execution authority.
All supplied compiler, harness-source, library, provider, object and discovered LLVM/manifest
hashes remained unchanged throughout the complete run.

## Reproducible route and actual checks

Use the [harness README](../../../compiler/experiments/hpc_execution_v1/README.md)
and `run.py --help`. Supply the exact artifacts above, the baseline build commit,
`--provider /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so`, and these
extra entries:

```text
--extra-primitive tiled-masked /tmp/mdslc-mlir-hpc.kr7V2m/research.o _mlir_ciface_research_gemm
--extra-primitive row-contiguous <row-build>/row-contiguous.o _mlir_ciface___matcore_strict_gemm_f32_v1
--extra-driver row-contiguous <row-build>/bin/mdslc-region <row-driver-sha-above> <row-source-root>
--timing --cpu 2 --quiet-window-note 'Root and vector agent paused builds/tests; ordinary desktop remains, SMT sibling not isolated'
```

Default timing settings used: six shapes, three deterministically randomized lane
rounds, three samples per series, target batch duration at least 30ms (repetition
cap 1,048,576), two explicit warmups and calibration. No user allocation, provider,
governor or system toolchain configuration was modified. Driver-generated host
execution uses the existing `-O2` route; primitive callers use Clang `-O2`,
`-ffp-contract=off -frounding-math`, without recompiling the supplied object.

Actual outcomes of the complete run:

- **224 mathematical case executions:** 24 each for five strict region lanes
  (baseline generated/native/automatic, row generated/native) and three primitive
  lanes; 16 each for numerically legal existing-native/OpenBLAS region lanes.
- **42 dynamic failure cases:** all seven region lanes check early requested-shape
  failure and late read-only publication, for ordinary, zero-K and zero-M shapes.
  Exact status/frontier/publication counts and unchanged failing destinations pass.
- **48 strict policy refusals:** existing-native and OpenBLAS reject at GEMM 3,
  after reads 1/2 and before publication, including empty shapes.
- **10 checker negative controls:** intentional output corruption exits 4 with the
  expected oracle mismatch, one per lane. No crash is accepted as this negative.
- ELF64/x86-64 relocatable and executable checks, primitive C-wrapper definitions,
  and direct candidate/Runtime/provider dependencies pass mechanically.
- Three small parser/source-pair tests pass, including wrong ELF kind/class/machine
  refusal and exact strict-to-reassociated fixture correspondence.

The 16 ordinary cases cover squares 1/16/64/128/256/512, rectangular
extents, skinny/tall and tails, dynamic dimensions, zero M/N/K/all-zero, and null
empty buffers. Deterministic small dyadics have exact double/f32 oracle agreement;
every output bit, input bit and both buffer guard regions are checked.
Eight strict controls cover scalar and 7x13 tails: increasing-K cancellation,
separate product/add, gradual underflow and signed zero. The FMA falsifier first
accumulates -1, then `(1+2^-23)*(1-2^-23)`: separate operations give +0, while fused
update gives -2^-46. The harness independently asserts that fused counteroracle.
Provider lanes deliberately exclude these strict-only rounding controls.

The earlier `hpc-execution-initial` run passed a draft harness but is not the final
evidence: the FMA case was strengthened before the committed complete run and
input comparisons were changed to bitwise equality, including signed zero.
No old result is substituted for the corrected oracle.

## Timing protocol and observed results

Linux x86-64, AMD Ryzen AI 9 HX 370, coherent Clang/LLVM/MLIR 21.1.8; provider 0.3.32.
Executions were pinned to CPU 2, a full Zen 5 core (maximum 5157.895 MHz reported by
`lscpu`), not a Zen 5c core. SMT sibling 14 was not reserved. Existing policy was
`amd-pstate-epp` / `performance`, configured maximum 5157000 kHz; these are policy
metadata, not a measured sustained frequency. Normal desktop services remained.
Root and the vector agent held builds/tests; timing preflight saw no compiler/build
processes, and the runner rechecked before each series. Preflight load average
was approximately 1.17/1.20/0.95, not a claim of a completely idle isolated machine.

**540 measured batch-mean samples**, nine per lane/shape, across 180 series; each
series also rechecked correctness. Tables show median microseconds per invocation.
All per-sample repetitions, timings, extrema, order and commands remain in raw JSON.

### Public region, end-to-end

| M×N×K | Baseline generated strict | Row generated strict | Native strict baseline | OpenBLAS, reassociation allowed |
| --- | ---: | ---: | ---: | ---: |
| 16×16×16 | 1.310 | 0.783 | 1.154 | 0.741 |
| 128×128×128 | 979.424 | 128.575 | 868.277 | 31.272 |
| 512×512×512 | 126290.448 | 8252.191 | 135702.888 | 1953.779 |
| 128×256×64 | 894.735 | 138.131 | 780.329 | 40.084 |
| 16×512×256 | 1929.238 | 131.991 | 1970.040 | 47.867 |
| 65×67×63 | 94.593 | 25.848 | 74.106 | 4.870 |

The observed baseline-generated/row-generated median ratios are 1.67×, 7.62×,
15.30×, 6.48×, 14.62×, 3.66× in table order. These ratios describe these inputs and
this session only. The independently built row-native control medians were within
about 1.5% of baseline-native across these shapes; baseline automatic closely
tracked baseline generated. These controls support, but do not universally prove,
that the selected schedule rather than unrelated build differences caused the gains.

### Standalone primitive, preallocated buffers

| M×N×K | Baseline generated | Row-contiguous | Explicit masked vector 4×8×1 |
| --- | ---: | ---: | ---: |
| 16×16×16 | 0.996 | 0.459 | 1.637 |
| 128×128×128 | 970.079 | 126.348 | 794.675 |
| 512×512×512 | 127040.196 | 7960.349 | 51369.660 |
| 128×256×64 | 875.531 | 127.225 | 796.967 |
| 16×512×256 | 1916.593 | 123.374 | 800.616 |
| 65×67×63 | 94.520 | 24.943 | 113.904 |

Negative performance evidence matters: explicit Vector IR is not itself an HPC
win. The tested vector candidate regressed the 16-square and 65×67×63 tail relative
to the baseline primitive, despite all mathematical controls passing. It should
not be selected universally on these results. No vector candidate was injected
into the production region adapter during this lane's measurements.

## Interpretation and nonclaims

Region timing includes two snapshots, private output allocation/initialization,
FP/candidate checks, one publication, completion and cleanup; it has no observation
operation. Primitive timing excludes those layers and uses preallocated buffers,
but includes the primitive's own output initialization. Both reuse warm inputs.
Do not subtract the columns and label the difference pure copy overhead: allocator,
alignment, FP scopes, initialization, linkage and dispatch also differ. There is
no direct OpenBLAS primitive timing in this report.

The source executable's public Result reports effects/status, not candidate
identity. Requested policies and pinned artifacts are recorded; no new identity
telemetry is inferred, especially for empty-output/zero-reduction bypasses.
The current whole-region exact paired witness is unchanged. Generated primitive
schedule execution is not whole-region transformation authority or buffer-reuse
legality. The stronger per-publication normal-return guarantee still applies.

There was no ASan/UBSan build or generated fault-instrumentation experiment by
this lane in this run. Other lanes' sanitizer results must retain their own
provenance. This corpus is not alias-coverage replacement for the existing public
storage-conformance matrix, nor broad special-value/provider conformance.
No thread scaling, cold-cache, NUMA, application throughput, cache/packing envelope,
planner regret, arbitrary rank/dtype/device, BLAS parity or universal speed floor
is established. No successful local result asserts hosted CI or merge completion.

Raw evidence (ignored, not committed):
`/home/hamza-usta/MatcoreDSL-wt-mlir-hpc-execution-evidence-v1/benchmark_reports/hpc-execution-comparison-v1/evidence.json`.
SHA-256: `da38159a3651853b2af10fa09121f2ed07739787b74a278e7b0228ab0c8be591`.
The journal contains 297 commands and 494 successful correctness records: 314
explicit correctness/failure/refusal cases plus 180 timing prechecks, separately
from the 10 expected corruption failures. Generated outputs remain untracked.
