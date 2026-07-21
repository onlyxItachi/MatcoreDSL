# Family C Attention Profile

Date: 2026-04-26

> Historical engineering record. Commands below are normalized to repository-
> relative paths; results describe the recorded machine and are not a current
> performance claim.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Driver: 595.45.04
- VRAM: 8188 MiB
- Nsight Compute CLI: 2026.1.0.0
- `nsys`: available

## Commands Run

```text
python3 tests/bench_attention.py
python3 -c "import importlib.util, pathlib; ...; mod.run_family_c()"
ncu --launch-skip 1 --launch-count 1 --section SpeedOfLight --section SchedulerStats --section MemoryWorkloadAnalysis ...
ncu --launch-skip 1 --launch-count 1 --section LaunchStats --section WarpStateStats --section SourceCounters ...
```

## Benchmark Results

`tests/bench_attention.py`

| Shape | MatcoreDSL ms | Torch Naive ms | Torch SDPA ms | max_err |
| --- | ---: | ---: | ---: | ---: |
| 16x16x8 | 0.204 | 0.013 | 0.079 | 0.000 |
| 32x32x16 | 0.257 | 0.017 | 0.038 | 0.000 |
| 64x64x32 | 0.516 | 0.018 | 0.039 | 0.000 |
| 128x128x64 | 1.731 | 0.022 | 0.050 | 0.000 |

Family C only from `tests/bench_fusion_fair.py`

| Shape | MatcoreDSL ms | PyTorch eager ms | Torch SDPA ms | vs eager | vs SDPA | max_err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32x32x16 | 0.237 | 0.017 | 0.009 | 0.07x | 0.04x | 0.000000 |
| 64x64x32 | 0.498 | 0.017 | 0.011 | 0.03x | 0.02x | 0.000000 |
| 128x128x64 | 1.416 | 0.023 | 0.023 | 0.02x | 0.02x | 0.000000 |
| 256x256x128 | 3.905 | 0.019 | 0.061 | 0.00x | 0.02x | 0.000000 |

Notes:
- The benchmark runs emitted the known `DT_TEXTREL` linker warning from cached kernels.
- Accuracy stayed exact to float32 tolerance in every case.

## NCU Findings

Profiled shape: 128x128x64, with one warmup launch skipped.

### Speed Of Light

- Duration: 1.46 ms
- Memory Throughput: 5.18%
- DRAM Throughput: 0.04%
- L1/TEX Cache Throughput: 15.43%
- L2 Cache Throughput: 0.47%
- Compute (SM) Throughput: 1.02%
- SM Active Cycles: 929257.25

### Memory Workload Analysis

- Memory Throughput: 95.23 MB/s
- Mem Busy: 5.18%
- Max Bandwidth: 1.02%
- L1/TEX Hit Rate: 98.41%
- L2 Hit Rate: 82.15%
- Mem Pipes Busy: 1.02%
- Local/shared spilling: 0

### Scheduler / Warp State

- Active Warps Per Scheduler: 1
- Eligible Warps Per Scheduler: 0.10
- No Eligible: 90.20%
- Issued Warp Per Scheduler: 0.10
- Warp Cycles Per Issued Instruction: 10.24
- Avg. Active Threads Per Warp: 16.07
- Avg. Not Predicated Off Threads Per Warp: 16.00

NCU stall summary:

- About 6.0 cycles per warp are spent waiting on an L1TEX scoreboard dependency.
- NCU attributes about 58.2% of the issue interval to that stall class.
- Branch divergence is not the issue: branch efficiency is 100% and divergent branches are 0.

### Launch / Access Pattern

- Block Size: 32
- Grid Size: 8
- Threads: 256
- Registers Per Thread: 38
- Static Shared Memory Per Block: 5.12 KB
- Waves Per SM: 0.02

NCU launch warning:

- The grid is too small to fill the device, leaving only 0.02 full waves across 24 SMs.

Source counter signals:

- Uncoalesced global accesses: 78% of total sectors are excessive.
- Uncoalesced shared accesses: 94% of total wavefronts are excessive.

## Interpretation

The current Family C path is not compute-bound. The kernel is under-occupied and spends most of its issue time stalled on memory dependencies, especially L1TEX scoreboard stalls. The source counters show the access pattern is also poor: both global and shared-memory traffic are heavily excessive, which matches the low SM throughput and low eligible-warp count.

The measured wall times line up with the NCU kernel duration, so the bottleneck is inside the fused attention kernel itself, not just host-to-device transfer.

## Recommended Next Kernel Changes

1. Replace the scalar online-softmax path with a warp-cooperative, tiled online softmax that keeps row statistics in registers/shared memory and reduces scoreboard latency.
2. Fix the access pattern for K/V and the score cache so global loads and shared-memory accesses are coalesced; the current excessive-sector numbers are too high for a memory-sensitive kernel.
3. Increase work per launch or occupancy. The current launch uses 8 blocks of 32 threads and only 0.02 waves per SM, which leaves most of the GPU idle.
4. If the implementation can support it, move Family C toward a FlashAttention-style fused kernel or MMA-backed tile path instead of scalar softmax plus separate accumulation.

## Follow-up Implementation Result

Codex implemented the first incremental kernel change on 2026-04-26:

- Family C now uses a block-cooperative score-cache path.
- Score generation is distributed over the `Br x Bc` score tile.
- Online-softmax row state (`m`, `l`, and correction scale) is stored in
  workgroup memory.
- V accumulation and final output stores are distributed over the
  `Br x Dtile` output tile.
- The route label is now `score_cached_block_coop_dtile64`.
- Disk cache version was bumped to avoid stale kernel reuse.

Post-change validation:

```text
cmake --build <legacy-build-dir> --parallel 2
python3 tests/test_family_c.py
python3 tests/test_fusion_analysis.py
python3 tests/test_fusion_contracts.py
```

Post-change benchmark snapshots:

| Benchmark | Shape | Before | After | Notes |
| --- | --- | ---: | ---: | --- |
| `tests/bench_attention.py` | 128x128x64 | 1.48-1.73 ms | 0.83-0.90 ms | exact to displayed tolerance |
| `tests/bench_fusion_fair.py` Family C | 128x128x64 | 1.416 ms | 0.765 ms | focused `run_family_c()` |
| `tests/bench_fusion_fair.py` Family C | 256x256x128 | 3.905 ms | 2.076 ms | focused `run_family_c()` |

Post-change NCU snapshot for 128x128x64:

- Duration: 597.6 us
- Memory Throughput: 2.54%
- DRAM Throughput: 0.09%
- L1/TEX Cache Throughput: 7.64%
- Compute (SM) Throughput: 1.42%
- Registers Per Thread: 31
- Static Shared Memory Per Block: 5.31 KB
- Grid/Block: 8 blocks x 32 threads
- Achieved Occupancy: 2.08%

Interpretation:

- The incremental cooperative rewrite is worth keeping: it cuts the profiled
  kernel duration by about 2.4x on 128x128x64 and improves benchmark wall time
  by roughly 1.7-1.9x for the larger attention shapes.
- The remaining bottleneck is still launch geometry and algorithmic structure:
  only 8 CTAs are launched for 128x128x64, so achieved occupancy remains about
  2%. The next major step should be a real FlashAttention/MMA-style Family C
  path or a split-row/CTA decomposition that increases parallelism without
  duplicating too much QK work.
