# MDSLC CPU Performance Deep Audit — Sanitized Evidence

This report contains authenticated aggregates only; raw benchmark JSON remains external and untracked.
Complete-hot and one-shot timing entries are arithmetic midpoints of run-order medians with the explicit `[minimum, maximum]` direction range. Diagnostic, prepack, and regret entries remain stable-forward only.

## Coverage

| Evidence | Count |
|---|---:|
| Executable cases | 711 |
| Passed | 583 |
| Reused after authenticated resume | 0 |
| Expected legality rejections | 128 |
| Predeclared runtime-bound skips | 58 |
| Accepted raw reports | 583 |
| Direction treatment | complete/oneshot paired stable-forward/stable-reverse; diagnostic/prepack/regret stable-forward only |
| Reverse passed | 429 |
| Reverse reused after authenticated resume | 0 |
| Reverse expected legality rejections | 110 |

All `failed` and incomplete states are fatal to this summarizer.

## Evidence identity and environment

- Benchmark binary SHA-256: `a5a07cf06b6274aeba50a66c20713847f2d65a28ab28021e6d27e64a941c31f5`
- Runner SHA-256: `be1db49ce5e82d34fc8b455d86c2fe2ad46ea5363a71b0af43b31104f1fd010d`
- Forward manifest SHA-256: `b3f872bd0085b15a8cd0cfcc7663af2a41f445355a3e3237c979dc52618362c0`
- Reverse manifest SHA-256: `4939c0c77586e4115dfe5c1aab1ff044d716e9a5d060c9f2ef52f265634df7f8`
- Source commit: `509ef2b775e501783dfa7f2c4aa21e91f513bd6a`
- Seed: `5566823262476977457`
- Host: AMD Ryzen AI 9 HX 370 w/ Radeon 890M
- Platform/compiler: linux x86_64; clang 21.1.8 (6ubuntu1); Release
- Compiler flags: ` -O3 -DNDEBUG`
- Timer: std::chrono::steady_clock; resolution=1 ns
- External provider: OpenBLAS 0.3.32
- External provider config: `OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=128`
- Frequency metadata: governor=performance, policy=min_khz=605264 max_khz=5157000, boost=enabled
- Capability record first line: cpu-capabilities-v2{version=2,architecture=x86_64,usable-vector-bits=512,xstate=0x2e7,amx-permission=not-granted,features=[portable-scalar-f32:hardware=yes/os=yes/compiler=yes/implementation=yes/runtime=yes,avx2:hardware=yes/os=yes/compiler=yes/implementation=yes/runtime=yes,fma:hardware=yes/os=yes/compiler=yes/implementation=yes/runtime=yes,avx512f:hardware=yes/os=yes/compiler=yes/implementation=yes/runtime=yes,avx512dq:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,avx512bw:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,avx512vl:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,avx512vnni:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,avx512bf16:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,amx-tile:hardware=no/os=no/compiler=yes/implementation=no/runtime=no,amx-bf16:hardware=no/os=no/compiler=yes/implementation=no/runtime=no,amx-int8:hardware=no/os=no/compiler=yes/implementation=no/runtime=no]}
- Capability record SHA-256: `d92ed9f8c6640f95b5e5dfd88770b45c6a170cf34068347d36e17d9b1bdc54c6`
- Topology record first line: cpu-topology-v1{version=1,architecture=x86_64,discovery=complete,logical-cpus=24,physical-cores=12,sockets=1,numa-nodes=1,cpu-map=[0:core=0/socket=0/node=0/thread=0,1:core=1/socket=0/node=0/thread=0,2:core=2/socket=0/node=0/thread=0,3:core=3/socket=0/node=0/thread=0,4:core=8/socket=0/node=0/thread=0,5:core=9/socket=0/node=0/thread=0,6:core=10/socket=0/node=0/thread=0,7:core=11/socket=0/node=0/thread=0,8:core=12/socket=0/node=0/thread=0,9:core=13/socket=0/node=0/thread=0,10:core=14/socket=0/node=0/thread=0,11:core=15/socket=0/node=0/thread=0,12:core=0/socket=0/node=0/thread=1,13:core=1/socket=0/node=0/thread=1,14:core=2/socket=0/node=0/thread=1,15:core=3/socket=0/node=0/thread=1,16:core=8/socket=0/node=0/thread=1,17:core=9/socket=0/node=0/thread=1,18:core=10/socket=0/node=0/thread=1,19:core=11/socket=0/node=0/thread=1,20:core=12/socket=0/node=0/thread=1,21:core=13/socket=0/node=0/thread=1,22:core=14/socket=0/node=0/thread=1,23:core=15/socket=0/node=0/thread=1],cache-groups=38}
- Topology record SHA-256: `bf61a06db3e2dfaee1332289b7ff287f6a7f9cdb29924c7d30ca321e5e383edc`

## Complete-hot single-thread native versus OpenBLAS

Only requested/actual one-thread cells with identical SMT, affinity, and applied-affinity metadata are compared.

| Family | Shapes | Median native/OpenBLAS throughput ratio |
|---|---:|---:|
| large-square | 5 | 0.849 |
| medium-square | 6 | 0.868 |
| short-wide | 4 | 0.884 |
| small-square | 7 | 0.942 |
| tail-heavy | 5 | 0.843 |
| tall-skinny | 4 | 0.795 |
| vector-like | 4 | 1.903 |

### Complete comparable shape matrix

| Family | Partition | M×N×K | Fastest native | Native GFLOP/s | OpenBLAS GFLOP/s | Native/OpenBLAS ratio |
|---|---|---|---|---:|---:|---:|
| large-square | holdout | 768×768×768 | `cpu.native-packed.avx512-fma.f32.v1` | 132.497 | 152.618 | 0.868 |
| large-square | holdout | 1024×1024×1024 | `cpu.native-packed.avx2-fma.f32.v1` | 131.217 | 147.119 | 0.892 |
| large-square | holdout | 1536×1536×1536 | `cpu.native-packed.avx512-fma.f32.v1` | 125.572 | 147.857 | 0.849 |
| large-square | holdout | 2048×2048×2048 | `cpu.native-packed.avx512-fma.f32.v1` | 124.647 | 148.737 | 0.838 |
| large-square | holdout | 4096×4096×4096 | `cpu.native-packed.avx512-fma.f32.v1` | 123.887 | 149.849 | 0.827 |
| medium-square | calibration | 96×96×96 | `cpu.native-packed.avx512-fma.f32.v1` | 82.688 | 96.541 | 0.857 |
| medium-square | calibration | 128×128×128 | `cpu.native-packed.avx2-fma.f32.v1` | 98.538 | 117.771 | 0.837 |
| medium-square | calibration | 192×192×192 | `cpu.native-packed.avx512-fma.f32.v1` | 120.536 | 140.596 | 0.857 |
| medium-square | calibration | 256×256×256 | `cpu.native-packed.avx512-fma.f32.v1` | 129.425 | 143.504 | 0.902 |
| medium-square | calibration | 384×384×384 | `cpu.native-packed.avx512-fma.f32.v1` | 130.429 | 148.432 | 0.879 |
| medium-square | calibration | 512×512×512 | `cpu.native-packed.avx512-fma.f32.v1` | 133.436 | 149.480 | 0.893 |
| short-wide | holdout | 32×8192×1024 | `cpu.native-packed.avx512-fma.f32.v1` | 65.584 | 77.324 | 0.848 |
| short-wide | calibration | 64×4096×4096 | `cpu.native-packed.avx512-fma.f32.v1` | 90.338 | 104.764 | 0.862 |
| short-wide | calibration | 128×4096×1024 | `cpu.native-packed.avx2-fma.f32.v1` | 111.434 | 121.217 | 0.919 |
| short-wide | holdout | 256×2048×4096 | `cpu.native-packed.avx512-fma.f32.v1` | 119.686 | 132.246 | 0.905 |
| small-square | diagnostic | 4×4×4 | `cpu.tiled.f32.v1` | 0.024 | 0.023 | 1.036 |
| small-square | diagnostic | 8×8×8 | `cpu.tiled.f32.v1` | 0.191 | 0.188 | 1.019 |
| small-square | diagnostic | 16×16×16 | `cpu.native-packed.avx2-fma.f32.v1` | 1.407 | 1.450 | 0.971 |
| small-square | diagnostic | 24×24×24 | `cpu.native-packed.avx2-fma.f32.v1` | 4.398 | 4.780 | 0.920 |
| small-square | diagnostic | 32×32×32 | `cpu.native-packed.avx512-fma.f32.v1` | 9.526 | 10.111 | 0.942 |
| small-square | diagnostic | 48×48×48 | `cpu.native-packed.avx2-fma.f32.v1` | 26.107 | 27.952 | 0.934 |
| small-square | diagnostic | 64×64×64 | `cpu.native-packed.avx512-fma.f32.v1` | 47.070 | 55.295 | 0.851 |
| tail-heavy | diagnostic | 31×33×35 | `cpu.native-packed.avx512-fma.f32.v1` | 9.598 | 11.344 | 0.846 |
| tail-heavy | diagnostic | 63×65×67 | `cpu.native-packed.avx512-fma.f32.v1` | 44.134 | 54.224 | 0.814 |
| tail-heavy | diagnostic | 127×129×131 | `cpu.native-packed.avx512-fma.f32.v1` | 94.342 | 111.890 | 0.843 |
| tail-heavy | diagnostic | 255×257×259 | `cpu.native-packed.avx512-fma.f32.v1` | 114.600 | 142.023 | 0.807 |
| tail-heavy | diagnostic | 511×513×515 | `cpu.native-packed.avx512-fma.f32.v1` | 125.076 | 144.156 | 0.868 |
| tall-skinny | holdout | 2048×256×4096 | `cpu.native-packed.avx512-fma.f32.v1` | 128.756 | 141.399 | 0.911 |
| tall-skinny | calibration | 4096×64×4096 | `cpu.native-packed.avx512-fma.f32.v1` | 87.221 | 116.524 | 0.749 |
| tall-skinny | calibration | 4096×128×1024 | `cpu.native-packed.avx512-fma.f32.v1` | 113.398 | 135.113 | 0.839 |
| tall-skinny | holdout | 8192×32×1024 | `cpu.native-packed.avx512-fma.f32.v1` | 65.811 | 87.779 | 0.750 |
| vector-like | diagnostic | 1×4096×4096 | `cpu.compiler-vectorized.avx2-fma.f32.v1` | 23.268 | 4.128 | 5.637 |
| vector-like | diagnostic | 8×4096×4096 | `cpu.compiler-vectorized.avx2-fma.f32.v1` | 24.445 | 29.985 | 0.815 |
| vector-like | diagnostic | 4096×4096×1 | `cpu.native-packed.avx512-fma.f32.v1` | 8.224 | 4.246 | 1.937 |
| vector-like | diagnostic | 4096×4096×8 | `cpu.native-packed.avx2-fma.f32.v1` | 62.030 | 33.182 | 1.869 |

### Weak shapes (native/OpenBLAS throughput ratio below 0.75)

| Family | M×N×K | Fastest native | Native ms [range] | OpenBLAS ms [range] | Ratio |
|---|---|---|---:|---:|---:|
| tall-skinny | 4096×64×4096 | `cpu.native-packed.avx512-fma.f32.v1` | 24.621 [22.649, 26.593] | 18.430 [17.992, 18.867] | 0.749 |
| tall-skinny | 8192×32×1024 | `cpu.native-packed.avx512-fma.f32.v1` | 8.158 [8.144, 8.171] | 6.116 [6.080, 6.153] | 0.750 |

## Allocation-inclusive one-shot

Ratios are one-shot time divided by the equivalent reused-workspace complete-call time.

| Comparable cells | Median ratio | P95 ratio | Maximum ratio |
|---:|---:|---:|---:|
| 46 | 1.063 | 2.425 | 2.621 |

## Prepacked-B preparation and amortization

Preparation is measured once outside steady-state execution. The amortized ratio includes that preparation cost.

| Repeated A inputs | Cells | Median preparation ms | Median steady-state ms/execution | Median amortized/complete ratio |
|---:|---:|---:|---:|---:|
| 1 | 8 | 7.785 | 1.721 | 1.245 |
| 4 | 8 | 7.953 | 1.786 | 0.781 |
| 16 | 8 | 8.354 | 1.915 | 0.599 |
| 64 | 8 | 7.848 | 1.868 | 0.547 |

## Cold-cache and compute-only diagnostics

- **complete-cold:** Cold-cache diagnostic; each sample is independently evicted and uses the diagnostic timer floor. Accepted cells: 63; comparable-to-hot cells: 63; median diagnostic/hot ratio: 1.155.
- **compute-only-hot:** Compute-only microkernel diagnostic; packing is excluded and results are not complete BLAS comparisons. Accepted cells: 35; comparable-to-hot cells: 35; median diagnostic/hot ratio: 0.903.

## Planner regret

The sanitized regret recomputation compares only candidates with the selected candidate's actual thread count and placement. The raw all-legal-candidate regret remains authenticated but is not used for this cross-placement summary.

| Cells | Median | P95 | Maximum | Candidate timings excluded for thread/placement mismatch |
|---:|---:|---:|---:|---:|
| 24 | 1.000 | 1.159 | 1.213 | 45 |

## Thread and placement exclusions

Multi-thread OpenBLAS uses unbound `allow-smt/none` placement in this audit and is retained only as a separate diagnostic; it is not compared with compact physical-core native runs.

| Actual OpenBLAS threads | Diagnostic cells | Median GFLOP/s |
|---:|---:|---:|
| 2 | 35 | 166.482 |
| 4 | 35 | 252.490 |
| 12 | 35 | 374.638 |

## Native-parallel scaling diagnostic

This is a native-only diagnostic, not BLAS parity. Each legal parallel cell is compared with its corresponding one-thread packed ISA baseline only when SMT, affinity, and applied-affinity metadata share the same native placement contract. Repeated requested-thread clamps are collapsed per shape/ISA/actual-thread cell.

| Family | Actual native threads | Comparable shape/ISA cells | Median speedup | Median parallel efficiency |
|---|---:|---:|---:|---:|
| large-square | 2 | 10 | 1.844 | 0.922 |
| large-square | 4 | 10 | 3.279 | 0.820 |
| large-square | 6 | 2 | 3.490 | 0.582 |
| large-square | 8 | 2 | 4.149 | 0.519 |
| large-square | 12 | 6 | 6.108 | 0.509 |
| medium-square | 2 | 8 | 1.429 | 0.714 |
| medium-square | 3 | 2 | 2.310 | 0.770 |
| medium-square | 4 | 2 | 3.077 | 0.769 |
| short-wide | 2 | 2 | 1.418 | 0.709 |
| tail-heavy | 2 | 4 | 1.625 | 0.812 |
| tail-heavy | 4 | 2 | 2.988 | 0.747 |
| tall-skinny | 2 | 8 | 2.175 | 1.088 |
| tall-skinny | 4 | 8 | 4.104 | 1.026 |
| tall-skinny | 12 | 8 | 7.110 | 0.593 |
| vector-like | 2 | 4 | 1.076 | 0.538 |
| vector-like | 4 | 4 | 1.257 | 0.314 |
| vector-like | 12 | 4 | 1.256 | 0.105 |

- Requested-to-actual thread clamps retained and excluded where comparability changed: 49 cells.

## Rejections and predeclared skips

| Category | Count |
|---|---:|
| expected rejection: parallel-output-macro-tile-count | 128 |
| predeclared skip: audit runtime bound: 1073741824 scalar products exceeds 134217728 | 9 |
| predeclared skip: audit runtime bound: 1073741824 scalar products exceeds 33554432 | 9 |
| predeclared skip: audit runtime bound: 1073741824 scalar products exceeds 536870912 | 9 |
| predeclared skip: audit runtime bound: 134217728 scalar products exceeds 33554432 | 3 |
| predeclared skip: audit runtime bound: 135003645 scalar products exceeds 134217728 | 1 |
| predeclared skip: audit runtime bound: 135003645 scalar products exceeds 33554432 | 1 |
| predeclared skip: audit runtime bound: 2147483648 scalar products exceeds 134217728 | 2 |
| predeclared skip: audit runtime bound: 2147483648 scalar products exceeds 33554432 | 2 |
| predeclared skip: audit runtime bound: 2147483648 scalar products exceeds 536870912 | 2 |
| predeclared skip: audit runtime bound: 268435456 scalar products exceeds 134217728 | 2 |
| predeclared skip: audit runtime bound: 268435456 scalar products exceeds 33554432 | 2 |
| predeclared skip: audit runtime bound: 3623878656 scalar products exceeds 134217728 | 1 |
| predeclared skip: audit runtime bound: 3623878656 scalar products exceeds 33554432 | 1 |
| predeclared skip: audit runtime bound: 3623878656 scalar products exceeds 536870912 | 1 |
| predeclared skip: audit runtime bound: 452984832 scalar products exceeds 134217728 | 1 |
| predeclared skip: audit runtime bound: 452984832 scalar products exceeds 33554432 | 1 |
| predeclared skip: audit runtime bound: 536870912 scalar products exceeds 134217728 | 2 |
| predeclared skip: audit runtime bound: 536870912 scalar products exceeds 33554432 | 2 |
| predeclared skip: audit runtime bound: 56623104 scalar products exceeds 33554432 | 1 |
| predeclared skip: audit runtime bound: 68719476736 scalar products exceeds 134217728 | 1 |
| predeclared skip: audit runtime bound: 68719476736 scalar products exceeds 33554432 | 1 |
| predeclared skip: audit runtime bound: 68719476736 scalar products exceeds 536870912 | 1 |
| predeclared skip: audit runtime bound: 8589934592 scalar products exceeds 134217728 | 1 |
| predeclared skip: audit runtime bound: 8589934592 scalar products exceeds 33554432 | 1 |
| predeclared skip: audit runtime bound: 8589934592 scalar products exceeds 536870912 | 1 |

## Interpretation boundary

These host-bounded measurements authenticate correctness and timing provenance. They do not establish universal performance, physical multi-node NUMA behavior, or hardware-counter claims.
