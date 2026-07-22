# MDSLC Milestone 4 single-thread CPU calibration

Date: 2026-07-22

Status: accepted as validation-host evidence for the deterministic planner v2
rules. This is not a universal CPU-performance claim.

## Measurement contract

All measurements below used the Release `matcore-bench` implementation from
the standalone compiler project. Runs were pinned with `taskset -c 0`, set
OpenBLAS local execution to one thread, used hot caches, 64-byte inputs,
caller-reused workspace, and included transient packing in every measured
execution. Inputs and correctness checking were outside the timing interval.
The quick profile used 3 warmups, 11 measured samples, and a 2 ms aggregate
timer floor. The representative profile used 2 warmups and 7 samples with the
same floor. Every result passed `--guard` and the independent double-precision
oracle.

The validation host was an AMD Ryzen AI 9 HX 370, Ubuntu 26.04, kernel
7.0.0-27, Clang 21.1.8 Release (`-O3 -DNDEBUG`), and OpenBLAS 0.3.32 pthread,
LP64:

```text
OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=128
```

Logical CPU 0 is paired with CPU 12, has private 48 KiB L1D and 1 MiB L2, and
belongs to the 16 MiB LLC group shared by CPUs 0-3 and 12-15. The
`amd-pstate-epp` governor and energy preference were both `performance`, boost
was enabled, and the configured maximum frequency was 5.157 GHz. Frequency was
not fixed, so small differences between otherwise identical forced and
automatic runs are ordinary sampling variation.

The broad standard-profile scalar attempt was stopped without producing JSON
when it exceeded the practical calibration duration. The replacement matrix
kept every declared quick shape and representative square, tail, and
rectangular problems at or below 2,147,483,648 SGEMM operations. In particular,
1536 and 2048 squares were not used for cross-variant regret because repeating
their scalar baseline would not improve the crossover decision.

Representative command shape:

```sh
OPENBLAS_NUM_THREADS=1 taskset -c 0 matcore-bench \
  --m M --n N --k K --variant VARIANT --threads 1 \
  --warmup 2 --iterations 7 --hot-cache --include-packing \
  --reuse-workspace --alignment 64 --timer-floor-us 2000 \
  --guard --json-out RAW_EXTERNAL_PATH
```

The five forced variants and `auto` were executed for all 30 calibration
shapes: 180 guarded results. Eight additional native-packed runs measured
caller-owned prepacked B separately. Raw JSON, standard output, build logs, and
intermediate analyses stayed outside Git.

## Planner calibration result

The first provisional fixed OpenBLAS cost caused catastrophic quick-profile
regret at 16x16x16 and 64x7x19. The measured static correction changes its cost
to `work + 2000`. A second sweep exposed a provider-specific row-vector
crossover: at M=1, N=4096, K=4096, compiler-vectorized took 1.254 ms while
OpenBLAS took 8.091 ms. The deterministic rule therefore excludes OpenBLAS
from automatic selection when `M == 1`; forced OpenBLAS remains legal and
inspectable. This is intentionally an exact row-major M rule, not an
unmeasured `min(M,N)` generalization.

After those two rules, measured automatic-planner regret was:

| Matrix | Count | Median | p95 | Maximum | Choices above 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quick | 8 | 1.006 | 1.132 | 1.132 | 0 |
| Representative | 22 | 1.000 | 1.099 | 1.124 | 0 |
| Combined | 30 | 1.000 | 1.124 | 1.132 | 0 |

For these aggregate values, ratios below 1.0 were conservatively normalized to
1.0 because forced and automatic measurements were independent repetitions.
The raw combined median without this normalization was 0.997, p95 was 1.124,
and maximum was 1.132. Both treatments meet the declared median <= 1.20, p95
<= 1.50, and no-catastrophic-choice <= 2.0 targets.

Across the 30 forced comparisons, OpenBLAS was fastest on 26 shapes,
compiler-vectorized on two, reference on one, and native packed on one. The
automatic policy chose the correct class except for accepted near-crossover
sampling and the 4096x4096x64 case: native packed took 15.807 ms at 135.85
GFLOP/s, while automatic OpenBLAS took 17.365 ms at 123.66 GFLOP/s, for 1.099
regret.

## Quick profile

Time/GFLOP/s columns show median execution time followed by throughput.
Measured ratios below 1.0 reflect independent-run noise, not negative regret.

| MxNxK | Fastest forced | Fastest time / GFLOP/s | Automatic selection | Auto time / GFLOP/s | Measured ratio |
| --- | --- | ---: | --- | ---: | ---: |
| 1x1x1 | `cpu.reference.f32.v1` | 43.66 ns / 0.05 | `cpu.reference.f32.v1` | 49.43 ns / 0.04 | 1.132 |
| 2x3x2 | `cpu.compiler-vectorized.avx2-fma.f32.v1` | 48.80 ns / 0.49 | `cpu.reference.f32.v1` | 50.45 ns / 0.48 | 1.034 |
| 16x16x16 | `cpu.external.openblas.f32.v1` | 103.27 ns / 79.32 | `cpu.external.openblas.f32.v1` | 104.99 ns / 78.02 | 1.017 |
| 33x35x37 | `cpu.external.openblas.f32.v1` | 700.11 ns / 122.08 | `cpu.external.openblas.f32.v1` | 695.84 ns / 122.83 | 0.994 |
| 64x7x19 | `cpu.external.openblas.f32.v1` | 305.12 ns / 55.79 | `cpu.external.openblas.f32.v1` | 308.87 ns / 55.12 | 1.012 |
| 127x129x131 | `cpu.external.openblas.f32.v1` | 30.759 us / 139.55 | `cpu.external.openblas.f32.v1` | 30.565 us / 140.43 | 0.994 |
| 128x128x128 | `cpu.external.openblas.f32.v1` | 28.545 us / 146.94 | `cpu.external.openblas.f32.v1` | 28.284 us / 148.29 | 0.991 |
| 256x256x256 | `cpu.external.openblas.f32.v1` | 218.825 us / 153.34 | `cpu.external.openblas.f32.v1` | 217.263 us / 154.44 | 0.993 |

The two sub-50 ns per-call results are aggregate-loop throughput observations.
They satisfy the 2 ms aggregate floor but remain call-overhead diagnostics, not
useful kernel-throughput claims.

## Representative profile

| MxNxK | Fastest forced | Fastest time / GFLOP/s | Automatic selection | Auto time / GFLOP/s | Measured ratio |
| --- | --- | ---: | --- | ---: | ---: |
| 1x4096x4096 | `cpu.compiler-vectorized.avx2-fma.f32.v1` | 1.254 ms / 26.75 | `cpu.compiler-vectorized.avx2-fma.f32.v1` | 1.410 ms / 23.79 | 1.124 |
| 8x4096x4096 | `cpu.external.openblas.f32.v1` | 8.130 ms / 33.02 | `cpu.external.openblas.f32.v1` | 8.315 ms / 32.28 | 1.023 |
| 24x24x24 | `cpu.external.openblas.f32.v1` | 286.38 ns / 96.54 | `cpu.external.openblas.f32.v1` | 283.91 ns / 97.38 | 0.991 |
| 31x33x35 | `cpu.external.openblas.f32.v1` | 555.36 ns / 128.94 | `cpu.external.openblas.f32.v1` | 557.71 ns / 128.40 | 1.004 |
| 32x32x32 | `cpu.external.openblas.f32.v1` | 461.20 ns / 142.10 | `cpu.external.openblas.f32.v1` | 463.33 ns / 141.44 | 1.005 |
| 32x4096x4096 | `cpu.external.openblas.f32.v1` | 12.720 ms / 84.41 | `cpu.external.openblas.f32.v1` | 12.680 ms / 84.68 | 0.997 |
| 48x48x48 | `cpu.external.openblas.f32.v1` | 1.442 us / 153.36 | `cpu.external.openblas.f32.v1` | 1.444 us / 153.20 | 1.001 |
| 63x65x67 | `cpu.external.openblas.f32.v1` | 3.631 us / 151.13 | `cpu.external.openblas.f32.v1` | 3.602 us / 152.33 | 0.992 |
| 64x64x64 | `cpu.external.openblas.f32.v1` | 3.327 us / 157.61 | `cpu.external.openblas.f32.v1` | 3.316 us / 158.13 | 0.997 |
| 64x4096x4096 | `cpu.external.openblas.f32.v1` | 19.961 ms / 107.58 | `cpu.external.openblas.f32.v1` | 19.929 ms / 107.76 | 0.998 |
| 96x96x96 | `cpu.external.openblas.f32.v1` | 11.063 us / 159.94 | `cpu.external.openblas.f32.v1` | 11.032 us / 160.40 | 0.997 |
| 192x192x192 | `cpu.external.openblas.f32.v1` | 91.467 us / 154.76 | `cpu.external.openblas.f32.v1` | 90.965 us / 155.62 | 0.995 |
| 255x257x259 | `cpu.external.openblas.f32.v1` | 227.436 us / 149.26 | `cpu.external.openblas.f32.v1` | 226.498 us / 149.88 | 0.996 |
| 256x1024x4096 | `cpu.external.openblas.f32.v1` | 15.154 ms / 141.71 | `cpu.external.openblas.f32.v1` | 15.043 ms / 142.75 | 0.993 |
| 384x384x384 | `cpu.external.openblas.f32.v1` | 728.802 us / 155.39 | `cpu.external.openblas.f32.v1` | 725.343 us / 156.13 | 0.995 |
| 511x513x515 | `cpu.external.openblas.f32.v1` | 1.768 ms / 152.73 | `cpu.external.openblas.f32.v1` | 1.755 ms / 153.88 | 0.992 |
| 512x512x512 | `cpu.external.openblas.f32.v1` | 1.719 ms / 156.16 | `cpu.external.openblas.f32.v1` | 1.722 ms / 155.91 | 1.002 |
| 768x768x768 | `cpu.external.openblas.f32.v1` | 5.775 ms / 156.88 | `cpu.external.openblas.f32.v1` | 5.751 ms / 157.53 | 0.996 |
| 1024x256x4096 | `cpu.external.openblas.f32.v1` | 14.516 ms / 147.93 | `cpu.external.openblas.f32.v1` | 14.455 ms / 148.57 | 0.996 |
| 1024x1024x1024 | `cpu.external.openblas.f32.v1` | 13.896 ms / 154.54 | `cpu.external.openblas.f32.v1` | 13.961 ms / 153.82 | 1.005 |
| 4096x32x4096 | `cpu.external.openblas.f32.v1` | 11.077 ms / 96.93 | `cpu.external.openblas.f32.v1` | 11.084 ms / 96.87 | 1.001 |
| 4096x4096x64 | `cpu.native-packed.avx2-fma.f32.v1` | 15.807 ms / 135.85 | `cpu.external.openblas.f32.v1` | 17.365 ms / 123.66 | 1.099 |

## Native packed evidence

Transient-packing native AVX2/FMA was faster than the existing
compiler-vectorized candidate on 27 of 30 shapes. The exceptions were the two
tiny calls and M=1,N=4096,K=4096, where packing B dominates. Representative
native/compiler speedups were 6.73x at 16x16x16, 2.81x at 33x35x37, 2.22x at
128 square, 2.01x at 256 square, 2.11x at 512 square, 2.40x at 1024 square,
and 2.50x at 4096x4096x64. These are measurements on this host, not portable
ratios.

Prepacked-B mode excludes B packing but still includes A packing and compute.
It uses caller-owned storage; it is not a compute-only comparison.

| MxNxK | Transient median / GFLOP/s | Prepacked-B median / GFLOP/s | Speedup | A workspace | Prepacked B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1x4096x4096 | 7.882 ms / 4.26 | 1.324 ms / 25.34 | 5.95x | 4 KiB | 64 MiB |
| 8x4096x4096 | 9.580 ms / 28.02 | 2.329 ms / 115.27 | 4.11x | 8 KiB | 64 MiB |
| 33x35x37 | 1.342 us / 63.69 | 1.082 us / 79.01 | 1.24x | 5.25 KiB | 6.94 KiB |
| 128x128x128 | 33.899 us / 123.73 | 30.659 us / 136.80 | 1.11x | 64 KiB | 64 KiB |
| 256x256x256 | 252.862 us / 132.70 | 236.842 us / 141.67 | 1.07x | 128 KiB | 256 KiB |
| 512x512x512 | 1.967 ms / 136.44 | 1.908 ms / 140.72 | 1.03x | 128 KiB | 1 MiB |
| 1024x1024x1024 | 15.868 ms / 135.33 | 15.537 ms / 138.21 | 1.02x | 128 KiB | 4 MiB |
| 4096x4096x64 | 15.807 ms / 135.85 | 15.539 ms / 138.20 | 1.02x | 32 KiB | 1 MiB |

OpenBLAS remained the stronger general single-thread baseline: for example,
154.54 GFLOP/s versus native packed 135.33 GFLOP/s at 1024 square. Native
packed did have a measured advantage on 4096x4096x64 (135.85 versus 123.58
GFLOP/s). The evidence supports a useful independent native engine, not a
claim of general OpenBLAS parity.

## Scope limits

- Results apply to this CPU, logical core, compiler, provider, governor, and
  single-thread hot-cache contract only.
- Boost was enabled and frequency was not locked. The benchmark records policy
  metadata but does not measure invariant per-sample core clocks.
- Calibration intervals reused allocated workspace. End-to-end
  `--include-allocation`, cold-cache, and compute-only modes are distinct and
  were not used to derive these rules.
- OpenBLAS 0.3.32 was locally authenticated and thread-controlled. A different
  provider/version may have different crossovers.
- No multithread, AVX-512, BF16, INT8, AMX, Windows, or multi-node NUMA result is
  implied.
- The M=1 exception is intentionally narrow. Additional aspect-ratio rules
  require additional evidence rather than extrapolation.
