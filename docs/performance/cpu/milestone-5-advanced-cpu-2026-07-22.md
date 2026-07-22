# MDSLC Milestone 5 advanced CPU performance evidence

Date: 2026-07-22

Status: accepted as host-specific performance evidence for the Linux advanced
CPU backend. This report does not claim universal performance, BLAS parity,
physical multi-node NUMA validation, or Windows validation.

## Evidence identity and integrity

The tables in this report were mechanically recomputed from 46 guarded
schema-v4 JSON files produced by a clean Release build at source commit:

```text
5f634aef2a0b47cd033df77c40d709456603b405
```

Every file reports `source_worktree_dirty=false`,
`source_provenance_state=clean`, and
`source_provenance_origin=git-worktree`. All 46 top-level results are timing
valid, correct against the independent double-precision oracle, and have their
final timed output authenticated. Every legal timed candidate in the two
planner-regret matrices is plan-authenticated and correct. The top-level
results checked 303,411,656 output elements; the largest absolute error was
`2.31798e-5`, and the largest observed error-to-allowed-error fraction was
0.008814.

The evidence snapshot predates later review and documentation-only integration
commits. Those later commits do not retroactively change the measured binary or
the source identity recorded here. A production-code change requires a new
guarded performance snapshot.

Raw JSON remains outside Git under:

```text
/home/hamza-usta/archives/MatcoreDSL-M5-perf-20260722/
```

The reviewed content-digest inventory is:

| Evidence set | JSON files | SHA-256 of sorted content-digest list |
| --- | ---: | --- |
| `v4-final-calibration-compact` | 20 | `6f21f11a3bdc2a16fa156ba0331a9bc10914945af7c0dd573ab239882a5737d1` |
| `v4-final-balanced-regret-provider` | 6 | `4ec5af3fd42dc2f54f918a4dc79a1ef1d0ef720c0a2f97fc266807d202d710b9` |
| `v4-final-tiny` | 2 | `8f4c62aef7d9825cbdc208b36208a5118b654d8461251f5abc2a0fab38e53b76` |
| `v4-final-4096-scaling` | 18 | `2de21d259631ce42a5c472ab6c1a74340271044d1e9affdd01fc27425737918e` |
| Combined | 46 | `2b654699d05dbd2ba0516760d2be977b4f56f356e4a6ae94737182dfca37ac2b` |

Only this sanitized summary is committed. The raw measurements, stdout, build
products, and intermediate analyses remain untracked.

## Validation host and toolchain

- OS: Ubuntu 26.04 LTS, kernel `7.0.0-27-generic`, x86-64.
- CPU: AMD Ryzen AI 9 HX 370 with Radeon 890M.
- Topology: one socket, 12 physical cores, 24 logical CPUs, SMT enabled, one
  physical NUMA node.
- Cache topology: cores 0-3 and their siblings share 16 MiB L3; cores 4-11
  and their siblings share 8 MiB L3. Each core has 48 KiB L1D and 1 MiB L2.
- Frequency policy: `amd-pstate-epp`, `performance` governor, boost enabled;
  recorded policy range `605264..5157000` kHz. Frequency was not fixed.
- Compiler: Ubuntu Clang 21.1.8 (`6ubuntu1`), Release `-O3 -DNDEBUG`.
- Provider: OpenBLAS 0.3.32, pthread, LP64,
  `DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=128`.

The capability-v2 record runtime-authenticated portable F32, AVX2, FMA, and
AVX-512F implementations on this physical host. AVX-512DQ/BW/VL/VNNI/BF16
were hardware- and OS-visible but had no MDSLC implementation or runtime
validation. AMX was not present and no AMX execution is claimed.

## Measurement contract

All retained measurements used row-major contiguous F32 GEMM, F32
accumulation, 64-byte input alignment, hot-cache mode, caller-reused workspace,
and included packing. Allocation and persistent-context construction were
outside the timed region. Native packed timings include transient A/B packing,
compute, tails, and stores. Parallel native timings additionally include worker
submission and synchronization. The OpenBLAS interval is the complete CBLAS
call, including opaque provider-internal packing.

Planner regret uses the stable planner-v3 registry in forward order and exact
reverse order. Each candidate estimate is the arithmetic mean of the two pass
medians. Equal-cardinality untimed oracle replay is placed after forward timing
and before reverse timing; each final timed output is authenticated immediately
after its timed block. Forced plan fingerprints include variant, planner,
thread, workspace, packing, affinity, and validation-placement identity.

For the aggregate regret summaries, values are sorted and the benchmark-style
discrete median is `x[floor(N/2)]` (the upper middle value for even `N`). The
p95 is nearest rank `x[ceil(0.95*N)-1]`. For transparency, the conventional
average of the two central compact-matrix values is 1.003913; the discrete
value reported below is 1.005207. Either convention is well within the declared
acceptance bounds.

Representative command shapes were:

```sh
# Bound native/compact planner-regret matrix
taskset -c 0-11 matcore-bench --m M --n N --k K \
  --variant auto --threads 4 --warmup 5 --iterations 11 \
  --hot-cache --include-packing --reuse-workspace --alignment 64 \
  --physical-cores-only --affinity compact --timer-floor-us 5 \
  --planner-regret --guard --json-out RAW_EXTERNAL_PATH

# Provider-permitted planner-regret matrix
taskset -c 0-11 matcore-bench --m M --n N --k K \
  --variant auto --threads 4 --warmup 7 --iterations 15 \
  --hot-cache --include-packing --reuse-workspace --alignment 64 \
  --allow-smt --affinity none --timer-floor-us 20 \
  --planner-regret --guard --json-out RAW_EXTERNAL_PATH

# 4096-cube forced scaling point; T and ISA/provider policy varied
taskset -c 0-23 matcore-bench --m 4096 --n 4096 --k 4096 \
  --variant STABLE_ID --threads T --warmup 2 --iterations 5 \
  --hot-cache --include-packing --reuse-workspace --alignment 64 \
  --timer-floor-us 1000 --guard --json-out RAW_EXTERNAL_PATH
```

## Stable-ID abbreviations

The tables use compact labels for the following exact registry IDs:

| Label | Stable variant ID |
| --- | --- |
| reference | `cpu.reference.f32.v1` |
| tiled | `cpu.tiled.f32.v1` |
| compiler AVX2 | `cpu.compiler-vectorized.avx2-fma.f32.v1` |
| OpenBLAS | `cpu.external.openblas.f32.v1` |
| packed AVX2 | `cpu.native-packed.avx2-fma.f32.v1` |
| packed AVX-512 | `cpu.native-packed.avx512-fma.f32.v1` |
| parallel AVX2 | `cpu.native-parallel.avx2-fma.f32.v1` |
| parallel AVX-512 | `cpu.native-parallel.avx512-fma.f32.v1` |

## Compact native planner calibration

This matrix restricted the process to CPUs 0-11, used physical-core-only
placement and compact native affinity, and reserved CPU 11 for the benchmark
caller. Bound workers did not share that logical CPU. OpenBLAS was deliberately
rejected in this regime because multi-thread provider execution is unavailable
under bound native workers; the table therefore measures the deterministic
native planner against the legal compact/native candidate set.

Times are balanced complete-call estimates. The thread count in parentheses is
the selected plan's actual count.

| MxNxK | Automatic selection (threads) | Selected estimate | Fastest legal | Fastest estimate | Regret |
| --- | --- | ---: | --- | ---: | ---: |
| 16x16x16 | packed AVX2 (1) | 6.280 us | tiled | 6.035 us | 1.0406 |
| 24x24x24 | packed AVX2 (1) | 6.595 us | packed AVX2 | 6.595 us | 1.0000 |
| 32x32x32 | packed AVX-512 (1) | 6.850 us | packed AVX-512 | 6.850 us | 1.0000 |
| 33x35x37 | packed AVX-512 (1) | 7.500 us | packed AVX2 | 7.415 us | 1.0115 |
| 48x48x48 | packed AVX-512 (1) | 8.625 us | packed AVX2 | 8.505 us | 1.0142 |
| 63x65x67 | packed AVX-512 (1) | 12.630 us | packed AVX2 | 12.485 us | 1.0116 |
| 64x7x19 | packed AVX2 (1) | 7.170 us | packed AVX-512 | 7.035 us | 1.0192 |
| 64x64x64 | packed AVX-512 (1) | 11.920 us | packed AVX-512 | 11.920 us | 1.0000 |
| 64x256x512 | packed AVX-512 (1) | 150.324 us | packed AVX-512 | 150.324 us | 1.0000 |
| 96x96x96 | packed AVX-512 (1) | 20.835 us | packed AVX-512 | 20.835 us | 1.0000 |
| 127x129x131 | packed AVX-512 (1) | 45.355 us | packed AVX-512 | 45.355 us | 1.0000 |
| 128x128x128 | packed AVX-512 (1) | 42.570 us | packed AVX2 | 42.349 us | 1.0052 |
| 192x192x192 | parallel AVX-512 (2) | 158.670 us | packed AVX2 | 116.405 us | 1.3631 |
| 255x257x259 | parallel AVX-512 (2) | 182.110 us | parallel AVX-512 | 182.110 us | 1.0000 |
| 256x256x256 | parallel AVX-512 (2) | 199.929 us | parallel AVX-512 | 199.929 us | 1.0000 |
| 256x512x128 | parallel AVX-512 (2) | 174.369 us | parallel AVX2 | 173.914 us | 1.0026 |
| 384x384x384 | parallel AVX-512 (3) | 366.349 us | parallel AVX2 | 349.789 us | 1.0473 |
| 511x513x515 | parallel AVX-512 (4) | 708.438 us | parallel AVX2 | 693.383 us | 1.0217 |
| 512x64x512 | parallel AVX-512 (4) | 108.394 us | parallel AVX-512 | 108.394 us | 1.0000 |
| 512x512x512 | parallel AVX-512 (4) | 646.953 us | parallel AVX2 | 627.243 us | 1.0314 |

The mechanically recomputed aggregate is:

| Shapes | Discrete median | Nearest-rank p95 | Maximum | Above 1.50 | Above 2.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.005207 | 1.047343 | 1.363081 | 0 | 0 |

The worst observed compact decision was 192x192x192: the planner selected
two-thread parallel AVX-512 while single-thread packed AVX2 was fastest. Its
1.3631 regret remains below the declared p95 1.50 and catastrophic 2.0 bounds,
but it is a visible crossover weakness rather than a perfect selection.

## Provider-permitted planner calibration

This separate regime retained the CPUs 0-11 process mask but used
`--allow-smt --affinity none`. OpenBLAS owned provider scheduling, native
workers were not individually pinned, and caller affinity was not requested.
These results must not be merged numerically with the compact/native matrix.

| MxNxK | Automatic selection (threads) | Selected estimate | Fastest legal (threads) | Fastest estimate | Regret |
| --- | --- | ---: | --- | ---: | ---: |
| 16x16x16 | OpenBLAS (4) | 0.493 us | packed AVX2 (1) | 0.367 us | 1.3437 |
| 33x35x37 | OpenBLAS (4) | 1.253 us | OpenBLAS (4) | 1.253 us | 1.0000 |
| 64x7x19 | OpenBLAS (4) | 0.761 us | OpenBLAS (4) | 0.761 us | 1.0000 |
| 127x129x131 | OpenBLAS (4) | 22.520 us | OpenBLAS (4) | 22.520 us | 1.0000 |
| 256x256x256 | OpenBLAS (4) | 62.040 us | OpenBLAS (4) | 62.040 us | 1.0000 |
| 512x512x512 | OpenBLAS (4) | 709.457 us | OpenBLAS (4) | 709.457 us | 1.0000 |

| Shapes | Discrete median | Nearest-rank p95 | Maximum | Above 1.50 | Above 2.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1.000000 | 1.343740 | 1.343740 | 0 | 0 |

In this sampled provider-permitted matrix, packed AVX2 won 16x16x16 and
OpenBLAS won the other five shapes. This brackets an observed shape-dependent
crossover between the 16-cube fixture and the larger/tail fixtures; it does not
establish one universal dimension threshold. The 64x7x19 result also shows why
the crossover cannot be reduced to square size alone.

## Tiny call-overhead diagnostics

The two tiny results used the compact four-thread request but correctly planned
one-thread reference execution. Their complete-call interval includes
submission to pinned persistent worker 0 and synchronization.

| MxNxK | Selected | Median | p95 | GFLOP/s |
| --- | --- | ---: | ---: | ---: |
| 1x1x1 | reference (1) | 5.901 us | 7.090 us | 0.00034 |
| 2x3x2 | reference (1) | 5.221 us | 6.335 us | 0.00460 |

These are correctness and call-overhead observations. They are excluded from
the planner-regret aggregate and are not useful kernel-throughput claims.

## 4096-cube thread scaling

Each point computes 137,438,953,472 floating-point operations. Native runs used
compact affinity. Threads 1-12 used physical-core-only placement; thread 24
allowed SMT. Native one-thread rows use the packed variant and rows with two or
more threads use the corresponding parallel variant, so the speedup is a
backend-family comparison rather than one identical function body.

OpenBLAS used `--allow-smt --affinity none`; its worker scheduling and packing
remain provider-managed. Consequently, native/OpenBLAS throughput is useful
end-to-end process evidence but not an exact identical-affinity comparison.

### Native AVX2/FMA

| Requested threads | Actual | Median ms | p95 ms | GFLOP/s | Speedup | Efficiency | Workspace MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1099.541 | 1102.668 | 125.00 | 1.000 | 1.000 | 0.375 |
| 2 | 2 | 538.839 | 540.316 | 255.06 | 2.041 | 1.020 | 64.250 |
| 4 | 4 | 284.665 | 286.885 | 482.81 | 3.863 | 0.966 | 64.500 |
| 6 | 6 | 262.275 | 266.766 | 524.03 | 4.192 | 0.699 | 64.750 |
| 12 | 12 | 232.906 | 244.224 | 590.10 | 4.721 | 0.393 | 65.500 |
| 24 | 24 | 221.629 | 224.520 | 620.13 | 4.961 | 0.207 | 67.000 |

### Native AVX-512F/FMA

| Requested threads | Actual | Median ms | p95 ms | GFLOP/s | Speedup | Efficiency | Workspace MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1730.963 | 2194.383 | 79.40 | 1.000 | 1.000 | 0.375 |
| 2 | 2 | 591.103 | 602.476 | 232.51 | 2.928 | 1.464 | 64.250 |
| 4 | 4 | 546.592 | 553.029 | 251.45 | 3.167 | 0.792 | 64.500 |
| 6 | 6 | 416.187 | 424.958 | 330.23 | 4.159 | 0.693 | 64.750 |
| 12 | 12 | 226.189 | 241.514 | 607.63 | 7.653 | 0.638 | 65.500 |
| 24 | 24 | 207.637 | 220.839 | 661.92 | 8.336 | 0.347 | 67.000 |

The AVX-512 one-thread point was degraded and variable in this retained sweep,
as its 2.194-second p95 and the nominally superlinear two-thread efficiency
make clear. The resulting speedup ratios are arithmetically correct relative to
that measured baseline but must not be interpreted as ideal hardware scaling.
In this sweep AVX2 was faster at 1, 2, 4, and 6 threads; AVX-512 was faster at
12 and 24. That observed crossover is host-state- and problem-specific.

### OpenBLAS

| Requested threads | Actual | Median ms | p95 ms | GFLOP/s | Speedup | Efficiency | Workspace MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 892.413 | 894.081 | 154.01 | 1.000 | 1.000 | 0.000 |
| 2 | 2 | 460.474 | 549.293 | 298.47 | 1.938 | 0.969 | 0.000 |
| 4 | 4 | 242.862 | 249.507 | 565.91 | 3.675 | 0.919 | 0.000 |
| 6 | 6 | 232.499 | 234.956 | 591.14 | 3.838 | 0.640 | 0.000 |
| 12 | 12 | 126.566 | 127.777 | 1085.91 | 7.051 | 0.588 | 0.000 |
| 24 | 24 | 138.668 | 147.000 | 991.14 | 6.436 | 0.268 | 0.000 |

OpenBLAS was fastest among the three retained 4096-cube families at every
thread count. It peaked at 12 requested threads in this sweep; 24 threads was
slower, so SMT did not improve this problem/provider run. No external
workspace is reported because provider-internal storage is opaque and remains
inside the complete CBLAS-call interval.

## Acceptance-target evaluation

- Compact/native automatic regret: median 1.005 <= 1.20, p95 1.047 <= 1.50,
  maximum 1.363 < 2.0.
- Provider-permitted automatic regret: median 1.000 <= 1.20, p95/maximum
  1.344 <= 1.50 and < 2.0.
- Four-thread 4096-cube native speedup: AVX2 3.863x and AVX-512 3.167x, both
  above the declared 2.5x target.
- Physical-core-count execution improved over one thread for both native ISA
  families.
- Tiny operations remained one-thread reference executions; automatic planning
  did not introduce tiny parallel work.
- Actual native and provider thread counts never exceeded the explicit request
  in these retained files.
- The results do not show native parity with OpenBLAS on 4096-cube SGEMM.

## Caller placement and comparison limitations

For native scaling at 1, 2, 4, and 6 threads, CPU 23 was a dedicated caller
logical CPU on a physical core unused by workers; siblings 11 and 23 were
reserved. At 12 threads the caller remained excluded as logical CPU 23 but
shared the physical core used by worker CPU 11, so
`dedicated_physical_core=false`. At 24 threads every logical CPU was a worker
and the JSON explicitly reports that caller isolation was unavailable. The
OpenBLAS process was deliberately unbound within CPUs 0-23 and requested no
caller isolation.

These distinctions prevent false placement claims, but they also mean that the
12/24-thread native points and provider points are not identical placement
experiments. No NUMA page placement, page migration, or memory interleaving was
performed or claimed.

## Defensible claims and remaining limitations

The retained evidence supports these bounded claims on the declared host:

- packed and persistent-parallel AVX2/FMA and AVX-512F/FMA implementations
  executed correctly;
- compact native and provider-permitted automatic planning met the declared
  regret bounds on their respective sampled matrices;
- native four-thread execution exceeded the declared 2.5x large-GEMM speedup
  target;
- OpenBLAS remained the stronger 4096-cube throughput baseline;
- the planner avoided parallel execution for the retained tiny problems.

It does not support universal crossover rules, frequency-independent ISA
conclusions, native BLAS parity, confidence intervals across days or machines,
physical multi-node NUMA behavior, AVX-512 BF16/VNNI, AMX, Windows, or any GPU
claim. Frequency was not fixed, only one retained sweep exists per scaling
point, the final AVX-512 single-thread point was visibly perturbed, OpenBLAS
used a different affinity policy, and only hot-cache/include-packing reused-
workspace mode is summarized here. Those limitations are part of the evidence,
not silently normalized away.
