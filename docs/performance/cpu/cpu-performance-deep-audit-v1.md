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
- Frequency metadata: governor=performance, policy=min_khz=605264 max_khz=5157000, boost=enabled
- Capability record: cpu-capabilities-v2{version=2,architecture=x86_64,usable-vector-bits=512,xstate=0x2e7,amx-permission=not-granted,features=[portable-scalar-f32:hardware=yes/os=yes/compiler=yes/implementation=yes/runtime=yes,avx2:hardware=yes/os=yes/compiler=yes/implementation=yes/runtime=yes,fma:hardware=yes/os=yes/compiler=yes/implementation=yes/runtime=yes,avx512f:hardware=yes/os=yes/compiler=yes/implementation=yes/runtime=yes,avx512dq:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,avx512bw:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,avx512vl:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,avx512vnni:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,avx512bf16:hardware=yes/os=yes/compiler=yes/implementation=no/runtime=no,amx-tile:hardware=no/os=no/compiler=yes/implementation=no/runtime=no,amx-bf16:hardware=no/os=no/compiler=yes/implementation=no/runtime=no,amx-int8:hardware=no/os=no/compiler=yes/implementation=no/runtime=no]} planner_probe=cpu-planner-v3 request-id=0 request=automatic requested-threads=1 max-threads=0 allow-smt=false external-provider-parallelism=false worker-affinity-active=true physical-cores=12 logical-processors=24 available-processors=24 numa-nodes=1 placement-complete=true affinity-requested=true affinity-applied=true affinity=compact numa-policy=single-node local-logical-capacity=1 local-physical-capacity=1 selected-numa-nodes=[0] cross-numa=false caller-first-touch=false status=selected selected=cpu.reference.f32.v1 reason=lowest deterministic static cost; ties use priority then registry order candidates=[cpu.reference.f32.v1:legal:reason=legal:cost=16:priority=80:workspace=0:shared-workspace=0:per-worker-workspace=0:alignment=1:threads=1:runtime-validated=true:required-hardware=1:required-os=1:required-compiler=1:required-implementation=1:cross-numa=false,cpu.tiled.f32.v1:legal:reason=legal:cost=4103:priority=70:workspace=0:shared-workspace=0:per-worker-workspace=0:alignment=1:threads=1:runtime-validated=true:required-hardware=1:required-os=1:required-compiler=1:required-implementation=1:cross-numa=false,cpu.compiler-vectorized.avx2-fma.f32.v1:legal:reason=legal:cost=16388:priority=60:workspace=0:shared-workspace=0:per-worker-workspace=0:alignment=1:threads=1:runtime-validated=true:required-hardware=7:required-os=7:required-compiler=7:required-implementation=7:cross-numa=false,cpu.external.openblas.f32.v1:legal:reason=legal:cost=18446744073709551615:priority=10:workspace=0:shared-workspace=0:per-worker-workspace=0:alignment=1:threads=1:runtime-validated=true:required-hardware=1:required-os=1:required-compiler=1:required-implementation=1:cross-numa=false,cpu.native-packed.avx2-fma.f32.v1:legal:reason=legal:cost=48014:priority=50:workspace=128:shared-workspace=128:per-worker-workspace=0:alignment=64:threads=1:runtime-validated=true:required-hardware=7:required-os=7:required-compiler=7:required-implementation=7:cross-numa=false,cpu.native-packed.avx512-fma.f32.v1:legal:reason=legal:cost=48011:priority=40:workspace=128:shared-workspace=128:per-worker-workspace=0:alignment=64:threads=1:runtime-validated=true:required-hardware=13:required-os=13:required-compiler=13:required-implementation=13:cross-numa=false,cpu.native-parallel.avx2-fma.f32.v1:rejected:reason=parallel AVX2/FMA implementation is not runtime-validated:cost=18446744073709551615:priority=30:workspace=128:shared-workspace=64:per-worker-workspace=64:alignment=64:threads=1:runtime-validated=false:required-hardware=7:required-os=7:required-compiler=7:required-implementation=7:cross-numa=false,cpu.native-parallel.avx512-fma.f32.v1:rejected:reason=parallel AVX-512 implementation is not runtime-validated:cost=18446744073709551615:priority=20:workspace=128:shared-workspace=64:per-worker-workspace=64:alignment=64:threads=1:runtime-validated=false:required-hardware=13:required-os=13:required-compiler=13:required-implementation=13:cross-numa=false]
- Topology: logical=24, physical=12, NUMA nodes=1, record SHA-256=`f083415dba9a092cba7c2876b7e87a0942e089ba0a6deb91889dc93e2b640f94`

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
