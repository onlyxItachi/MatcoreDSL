# Milestone 4 planner-calibration lane report

Date: 2026-07-22

## Ownership

This lane owned only the sanitized summary under `docs/performance/cpu/` and
this report. It did not edit planner, runtime, benchmark, public ABI, package,
legacy, or generated-artifact files. The integration lead supplied the two
deterministic policy commits evaluated here; their equivalent integration
commits are the source of truth.

## Build and evidence

The lane began from integration commit `e614dc73bbfd15dd6a6f62df880641b1d5993ad2`.
It configured a fresh Release build with `/usr/bin/clang-21` and
`/usr/bin/clang++-21`, required native LibTooling, and required the coherent
pkg-config OpenBLAS provider. `matcore-bench --list-variants` reported exactly:

```text
auto
cpu.reference.f32.v1
cpu.tiled.f32.v1
cpu.compiler-vectorized.avx2-fma.f32.v1
cpu.external.openblas.f32.v1
cpu.native-packed.avx2-fma.f32.v1
```

The external evidence directory is:

```text
/home/hamza-usta/archives/MDSLC-m4-calibration-20260722T162854
```

It contains 368 non-build evidence files totaling 944,304 bytes. The complete
SHA-256 inventory is `final-evidence.sha256`; its digest is:

```text
e956f05a8d8cf26e127f373d86a97c9b34599b5d415a06f5fe67de6044622378
```

Raw JSON, stdout, logs, old-policy differential runs, and TSV analysis remain
outside the repository. No generated benchmark artifact is part of this
commit.

## Commands and matrix

The exact configure tuple used Clang/LLVM 21.1.8, native and bootstrap
frontends enabled, OpenBLAS enabled and required, and Release mode. Benchmark
runs used the command contract recorded in the sanitized performance summary:
CPU 0, one provider thread, hot cache, reused caller workspace, included
packing, 64-byte alignment, a 2 ms aggregate timer floor, and `--guard`.

All five forced variants and automatic selection passed correctness on all
eight quick shapes and 22 bounded representative shapes. The representative
set comprised squares from 24 through 1024, four additional tail cases, and
eight declared rectangular/aspect-ratio cases. The operation cap was
2,147,483,648; the unbounded standard scalar attempt was interrupted before it
produced JSON and is explicitly marked in the external archive.

Eight caller-owned prepacked-B native runs also passed. In total, 188 guarded
benchmark results used an independent double-precision oracle, finite-output
checks, and checksums where the large-shape sampled oracle applied.

## Differential calibration

The provisional OpenBLAS fixed cost produced quick-profile regret 11.19 at
16 square and 11.02 at 64x7x19. The lead's first static correction, evaluated
in this lane as commit `51e1ccb`, changed the provider model to `work + 2000`.
It reduced quick median/p95/max regret to 1.006/1.132/1.132, with no choice
above 2.0.

The bounded representative sweep then exposed M=1,N=4096,K=4096: automatic
OpenBLAS took 8.055 ms against compiler-vectorized 1.254 ms, regret 6.42. The
lead's exact M==1 rule, evaluated here as commit `d61ea6a`, selected
compiler-vectorized. This lane's repeat measured 1.410 ms and 1.124 regret; the
lead independently reproduced 1.258 ms. The final representative median,
p95, and maximum were 1.000, 1.099, and 1.124 after conservative normalization
of below-one independent-run ratios. There were no choices above 2.0.

Combined quick plus representative results were median 1.000, p95 1.124,
maximum 1.132, and zero choices above 2.0. Native packed beat
compiler-vectorized on 27 of 30 shapes and provided clear improvements on
medium, tail, and large shapes. OpenBLAS was fastest on 26 of 30 shapes;
native packed was fastest on 4096x4096x64. The sanitized report records all
selection decisions, measured medians/GFLOP/s, prepacked-B results, and
limitations.

## Validation and hygiene

- Fresh Release configuration and `matcore-bench` build passed.
- OpenBLAS 0.3.32 CBLAS and local one-thread control were required at configure
  time and present at runtime.
- Every retained result reports affinity `0`, correctness `true`, and valid
  timing at or above the aggregate floor.
- Every forced variant was exercised; automatic runs were repeated after each
  planner correction instead of reusing stale selection metadata.
- Raw evidence was kept outside Git and protected by the SHA-256 manifest.
- `git diff --check` and repository hygiene were run before handoff.

## Claim boundary

These results calibrate a deterministic single-thread rule on one physical
host. They do not establish universal crossover thresholds, locked-frequency
benchmarking, multithread scaling, AVX-512, BF16/INT8, AMX, Windows runtime,
or real multi-node NUMA behavior. They demonstrate that the native packed
engine is materially better than the existing native compiler-vectorized path
on supported medium/large problems while OpenBLAS remains the honest general
performance baseline.
