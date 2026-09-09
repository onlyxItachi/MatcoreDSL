# Independent connected-source measurement audit v1

## Verdict and identity

**ACCEPT with bounded comparison caveats**, 2026-09-09. Read-only independent
review of harness `53f15add56b4e0163ad00af1f60929b5da53de9a` and the report at
`f16d9b75b34696f6eea0aa00b7e346be49f1e193`:
[connected measurement record](generated-reassociate-source-evidence-v1.md).
The reviewer did not execute or modify the benchmark, raw records, or production
compiler. The data-validation/data-quality workflow guided independent identity,
grain, arithmetic, and comparison checks; no visualization was needed.

Raw files remain in `benchmark_reports/` under
`/home/hamza-usta/MatcoreDSL-wt-generated-reassociate-source-evidence-v1`:

| Record | Recomputed SHA256 |
| --- | --- |
| `generated-reassociate-source-correctness-v1/evidence.json` | `52dec4238fdcffbfe042120ef54a9e739c719d1902909dd201a9ce071df412fe` |
| `generated-reassociate-source-timing-v1/evidence.json` | `caf42fb4a0ae5168041b664822991d3ba19c7da1a2a74f0b3f5e12bb85506f61` |

In **each** record all 32 recorded artifact/source paths were independently
rehashed after the coordinated timing hold was released; all still matched.
This includes every issued executable, not merely compiler input filenames.
The tested compiler build is `73cdeb927c160c9eda691aeb7f666b9f22d3a6b3`, not the
research harness HEAD. Their explicitly recorded source differences are not
silently treated as a same-source compiler build.

Key verified artifacts:

- driver: `6423db0db4ea0db91d4c4f747f4160de5508c432b496be08267745d690bf96b5`;
- private candidate DSO: `262cfc9c40bf23d2bfa326fb1a17d78afd91243cea03e3e1c0fbbf98060c9e4c`;
- runtime DSO: `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014`;
- reassociate leaf: `22285cebb5ea339caf3b65170a41b316de1110091b7dac7f5ca8f0580d71cfd3`;
- strict leaf: `5951cec81bd48b3a72b25583e41e3422ee58e309c127cd0faaf27398c4321a50`;
- configured provider: `be2e7d119279836105e0361be92c78dbcd8a6e7357e74339bdbb32a5195ee35e`.

## Checks and recalculation

- Correctness-only: **122 commands**, 112 exit 0 and ten deliberate corruption
  exit 4. Timing run: **302 commands**, 292 exit 0 and the same ten intentional
  corruption controls. No other exit codes were present.
- Each record retains ten lanes and **192 ordinary correctness cases**. Strict
  lanes retain 24 cases; explicitly reassociated lanes use 16 ordinary dyadic
  controls. Three separately classified strict-policy refusal executables each
  returned 24 successful refusal records with empty stderr. These 72 refusal
  cases are not mislabeled as part of the 192 ordinary cases.
- Timing contains exactly **180 distinct (lane, shape, round) series**, six
  shapes, ten lanes, three rounds, three samples per series: **540 samples**.
  Each timing command was pinned to CPU 2, exited zero, emitted empty stderr,
  and included a successful correctness record before its three timing records.
  Every serialized timing sample matched its originating command output.
- Independently recomputed every `total_ns / repetitions` value, allowing only
  the recorded three-decimal printing precision; all denominators and times
  were positive. All **60** reported medians, minima, maxima and nine-sample
  counts exactly matched aggregation of the raw samples.
- Reproducible local audit code:
  `/tmp/mdslc-multi-source-review.oCbkLO/measurement_audit.py`.
  Headline reductions below use medians of the unrounded integer-time ratios,
  not an average of rounded percentages:
  `100 * (strict_median - new_median) / strict_median`.

| Shape M,N,K | Same-source latency reduction |
| --- | ---: |
| 16,16,16 | 27.279327% |
| 128,128,128 | 57.623943% |
| 512,512,512 | 50.704222% |
| 128,256,64 | 52.962733% |
| 16,512,256 | 47.524239% |
| 65,67,63 | 51.365285% |

The source comparator is `region-reassociate-generated-strict` versus
`region-reassociate-generated-reassociate`: identical original mathematical
source and per-operation permission, different selected legal realizations.
OpenBLAS is faster on five of the six measured shapes. The small generated case
is 0.561884 us versus provider 0.721016 us; this does not establish a crossover.

## Required interpretation limits

This is one local warm, single-thread, six-shape study on pinned CPU 2 of a
Ryzen AI 9 HX 370. Coordinated builds/tests stopped; SMT sibling 14, desktop
activity, thermal state, and hardware frequency were not isolated. Nine batch
means per cell are not nine independent machines or deployment populations.

The intervention combines schedule, AVX2/FMA, contraction and optimization
level. It is **not** a causal FMA-only result, a strict-profile speedup, or a
full x86-64-v3 comparison. The relaxed dyadic oracles do not replace the separate
adversarial numerical/source/ISA admission tests. Strict controls were retained,
not weakened to admit the new candidate.

Source timing and primitive timing are different scopes. The 512-square source
4.07731125 ms versus primitive 4.575250875 ms cannot be subtracted as a causal
snapshot cost or used to claim negative overhead or zero-copy. Their storage,
calling, allocation and cache histories differ.

Public `Result` does not report actual candidate identity. These records label
requested source policies and frozen artifacts; separate private identity and
source FMA-discriminator tests provide stronger implementation evidence.
`existing-native` here is the explicit forced C-ABI **reference** route, not
Matcore's packed/AVX2 native performance envelope. Configured provider bytes do
not independently prove arbitrary future dynamic-loader identity.

The findings support a separately gated optional candidate. They do not justify
changing automatic dispatch, adding performance thresholds, claiming BLAS
parity, closing Issue #15, or inferring another target's performance. No hosted
CI or canonical merge result is claimed by this measurement audit.
