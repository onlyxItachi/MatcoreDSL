# Family C Restore Snapshot

Date: 2026-04-29

> Historical engineering record. Commands below are normalized to repository-
> relative paths; results describe the recorded machine and are not a current
> performance claim.

## Objective

Restore the Family C fusion pipeline after two incremental optimization
experiments failed:

- Score-cache padding (`Br x (Bc + 1)`) was correct but slower.
- Naive row-subtile geometry introduced a native SIGSEGV during Family C smoke
  tests and was reverted.

The stable baseline is the existing block-cooperative scalar Family C path with
route label `score_cached_block_coop_dtile64`.

## Validation

```text
cmake --build <legacy-build-dir> --parallel 2
env MATCORE_CACHE_DIR=/tmp/matcore_restore_smoke \
  python3 -u -c \
  "import tests.test_family_c as t; print('start-small'); t.test_attention_uses_unscaled_softmax_scores(); print('done-small')"
env MATCORE_CACHE_DIR=/tmp/matcore_family_c_restore_full \
  python3 -u tests/test_family_c.py
env MATCORE_CACHE_DIR=/tmp/matcore_fusion_analysis_restore \
  python3 -u tests/test_fusion_analysis.py
env MATCORE_CACHE_DIR=/tmp/matcore_fusion_contracts_restore \
  python3 -u tests/test_fusion_contracts.py
env MATCORE_CACHE_DIR=/tmp/matcore_family_a_restore \
  python3 -u tests/test_family_a.py
env MATCORE_CACHE_DIR=/tmp/matcore_family_b_restore \
  python3 -u tests/test_family_b.py
```

All listed validation commands passed.

## Benchmark Snapshot

Command:

```text
env MATCORE_CACHE_DIR=/tmp/matcore_bench_restore \
  python3 -u tests/bench_attention.py
```

| Shape | MatcoreDSL ms | Torch Naive ms | Torch SDPA ms | max_err |
| --- | ---: | ---: | ---: | ---: |
| 16x16x8 | 0.170 | 0.014 | 0.065 | 0.000 |
| 32x32x16 | 0.187 | 0.017 | 0.060 | 0.000 |
| 64x64x32 | 0.363 | 0.018 | 0.056 | 0.000 |
| 128x128x64 | 0.966 | 0.030 | 0.223 | 0.000 |

Score-cache padding comparison from the rejected run:

| Shape | Padded score cache ms | Restored block-coop ms |
| --- | ---: | ---: |
| 16x16x8 | 0.174 | 0.170 |
| 32x32x16 | 0.231 | 0.187 |
| 64x64x32 | 0.341 | 0.363 |
| 128x128x64 | 1.212 | 0.966 |

Interpretation: padding was not worth keeping. It regressed the largest tracked
shape and did not fix the launch-geometry bottleneck.

## NCU Snapshot

Command:

```text
env MATCORE_CACHE_DIR=/tmp/matcore_ncu_restore \
  ncu --profile-from-start off \
  --section MemoryWorkloadAnalysis --section SourceCounters \
  --section LaunchStats --launch-count 4 --print-summary per-kernel \
  python3 -u tests/ncu_profile_family_c.py \
  --m 128 --n 128 --d 64 --warmup 1 --profile-iters 1
```

Profiled kernel: `fused_family_c_attention_fused_kernel`

| Metric | Value |
| --- | ---: |
| Grid | 8 blocks |
| Block | 32 threads |
| Registers/thread | 31 |
| Static shared memory/block | 5.31 KB |
| Waves/SM | 0.02 |
| Memory throughput | 422.88 MB/s |
| Mem busy | 2.54% |
| Max bandwidth | 1.05% |
| L1/TEX hit rate | 97.12% |
| L2 hit rate | 79.74% |
| Local/shared spilling | 0 |
| Branch efficiency | 99.91% |

Interpretation:

- This is not primarily a memory-layout/bank-conflict failure after the
  block-cooperative rewrite. Cache hit rates are high and spilling is zero.
- The remaining hard bottleneck is decomposition: `128x128x64` launches only
  eight CTAs on a 24-SM GPU.
- The next safe performance step should be a feature-flagged split-row or
  FlashAttention/MMA-style Family C path, not ad-hoc score-cache padding.
