# Connected generated reassociate measurement v1

## Identity and claim boundary

**OBSERVED**, one local warm single-thread CPU study, not a cost model or a
BLAS-parity envelope. The generated candidate is real source-connected execution
at compiler head `73cdeb927c160c9eda691aeb7f666b9f22d3a6b3` (PR #66 candidate,
not yet canonical). Harness head `53f15ad` extends the previously reviewed
`62c6237e0396084cc3b96f16548f24577794edfa` research runner. It adds explicit
`--generated-reassociate-source`, keeps all original strict controls, and compiles
generated-reassociate/generated-strict/native-strict from the **same** reassociated
source file. Seven runner unit controls pass. No production policy is modified.

Frozen compiler/artifacts:

- driver `6423db0db4ea0db91d4c4f747f4160de5508c432b496be08267745d690bf96b5`;
- private candidate DSO `262cfc9c40bf23d2bfa326fb1a17d78afd91243cea03e3e1c0fbbf98060c9e4c`;
- canonical runtime DSO `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014`;
- generated reassociate leaf `22285cebb5ea339caf3b65170a41b316de1110091b7dac7f5ca8f0580d71cfd3`;
- strict row-contiguous leaf `5951cec81bd48b3a72b25583e41e3422ee58e309c127cd0faaf27398c4321a50`;
- configured OpenBLAS 0.3.32 `be2e7d119279836105e0361be92c78dbcd8a6e7357e74339bdbb32a5195ee35e`.

The reassociate object is actually compiled for baseline x86-64 plus AVX2/FMA,
not the earlier wider x86-64-v3 research object. Strict/generated automatic remain
unchanged. `existing-native` here is the closed-region forced reference route,
**not** the packed AVX2 implementation. Public Result does not expose invocation
identity; requested policy and frozen linked artifacts are recorded separately.
Independent source FMA-discriminator/private report tests, not these exact-dyadic
timing cases alone, establish the connected implementation's numerical identity.

## Protocol and raw evidence

Ryzen AI 9 HX 370, CPU 2 (full Zen 5) pinned. SMT sibling 14, desktop/system
activity and thermal state are not reserved. No governor/boost/system changes.
Root's Release **172/172** and ASan+UBSan **117/117** runs finished; all three
agents explicitly paused builds/tests for the timing window. The runner also
refuses observed compiler/build activity before and between timing commands.

Ten lanes, six timed shapes, three deterministic-randomized lane-order rounds,
three samples per round: **180 timing series / 540 samples / 60 medians**.
Each sample is a calibrated warm batch targeting at least 30 ms, not a single
instruction timing. Region time includes reads/snapshots, private value allocation,
candidate legality/FP scopes, GEMM, publication and Result/session cleanup.
Host vector allocation/filling/oracles are outside timing; no observe operation.
Primitive timing uses preallocated descriptors and excludes those region costs.
All issued executables and frozen inputs are checked before/after commands.

Raw files are retained under the ignored `benchmark_reports/` directory in
`/home/hamza-usta/MatcoreDSL-wt-generated-reassociate-source-evidence-v1`:

- `generated-reassociate-source-correctness-v1/evidence.json`, SHA256
  `52dec4238fdcffbfe042120ef54a9e739c719d1902909dd201a9ce071df412fe`:
  **PASS**, 122 commands, 10 lanes, 192 correctness cases, no timing.
- `generated-reassociate-source-timing-v1/evidence.json`, SHA256
  `caf42fb4a0ae5168041b664822991d3ba19c7da1a2a74f0b3f5e12bb85506f61`:
  **PASS**, 302 commands, the same 10 lanes/192 correctness cases, 180 timing series.

Each run includes 10 deliberate output-corruption failures (exact exit 4),
48 source shape/access failure checks, and three strict-policy refusal executables
with 24 cases each, including the newly connected candidate. The relaxed lanes'
16 exact-dyadic cases do not substitute for the separately reviewed 1,417-case
reassociate arithmetic study. Full commands, output, samples and hashes are in JSON.

## Like-source results

Median microseconds; all three columns compile identical `reassociate.mdsl` with
the same per-GEMM numerical permission and region-level observable semantics.

| M,N,K | Generated strict candidate | Generated reassociate candidate | OpenBLAS |
| --- | ---: | ---: | ---: |
| 16,16,16 | 0.772661 | 0.561884 | 0.721016 |
| 128,128,128 | 128.063199 | 54.268135 | 31.206724 |
| 512,512,512 | 8271.116500 | 4077.311250 | 1963.379375 |
| 128,256,64 | 136.159734 | 64.045818 | 40.211737 |
| 16,512,256 | 132.089766 | 69.315109 | 48.512792 |
| 65,67,63 | 24.925214 | 12.122307 | 4.875961 |

**OBSERVED:** the new generated candidate takes 27.28–57.62% less time than the
existing generated strict candidate on these six shapes (1.38–2.36x speedup).
This is a whole-candidate comparison: schedule, target features, contraction and
optimization level differ together. It is not a causal FMA-only or tiling-only
measurement. Identical source permissions make both candidates legal choices.

**OBSERVED:** OpenBLAS is faster on five of six; the new generated path wins the
16-square case by 22.07%. The five losses range from 1.43x to 2.49x slower than
OpenBLAS. These points do not determine a reliable crossover threshold.

**UNRESOLVED:** the 512-square region measurement (4077.31 us) is faster than
its standalone primitive measurement (4575.25 us). Address placement, cache
history, allocation and calling context differ. Subtracting them as a causal
"snapshot overhead" would be invalid; no negative-overhead or zero-copy claim.

## Architectural interpretation

**PROVEN WITHIN THIS BOUNDED CONTRACT:** existing MLIR Transform/Linalg/Vector
machinery can produce a useful real CPU candidate without a new machine-level
abstraction. Matcore still owns operation permissions, output/effect legality,
candidate admission, source authentication and provider/native coexistence.
LLVM owns machine lowering. This does not prove that MLIR discovers globally
optimal schedules or that no additional higher-level planning abstraction will
ever be useful.

No automatic thresholds, source numerical defaults, provider selection policy,
GPU/NPU capability, fused region execution or Issue #15 closure follows.
Independent measurement audit is a separate pending gate from the passing runs;
preserve any corrections and negative observations in normal history.
