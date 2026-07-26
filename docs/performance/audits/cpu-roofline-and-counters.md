# CPU roofline and hardware-counter audit

Milestone: MDSLC Milestone 6 — CPU Performance Deep Audit

Physical measurement date: 2026-07-26

Production source checkpoint:
`951239f1bee5541a4cf5ad72fab2192de07cf89d`

Audit-runner checkpoint:
`8c606d6de2e09429f85a24ff6d081a9c610ecd8e`

This report answers a narrow question: which current F32 GEMM regions are
limited by arithmetic execution, data movement, packing, latency, or parallel
coordination on the declared host? It does not change a kernel, planner rule,
or public ABI.

## Claim convention

Every substantive conclusion is prefixed with one of:

- **measured** — a timed or inspected result from the identified host,
  executable, source checkpoint, and command;
- **derived** — arithmetic from measured or explicitly stated model inputs;
- **source-backed** — supported by an identified upstream document or source
  revision;
- **hypothesis** — plausible but not established by the available
  observability;
- **proposed** — a Milestone 7 experiment, not accepted production behavior.

In particular, `llvm-mca` is a static scheduler model. Its IPC, latency,
resource pressure, and throughput are never called measured hardware
counters.

## Host and evidence boundary

**Measured.** The physical host is an AMD Ryzen AI 9 HX 370, family 26 model
36, running Ubuntu 26.04 and Linux `7.0.0-27-generic`. Linux reports one socket,
12 physical cores, 24 logical CPUs, SMT enabled, and one NUMA node. CPUs 0–3
have a 5157 MHz policy maximum and share a 16 MiB L3; CPUs 4–11 have a
3289474 kHz policy maximum and share an 8 MiB L3. Every core has a 48 KiB L1D
and a 1 MiB L2. The governor is `performance`, `amd-pstate-epp` is active, and
boost is enabled.

**Source-backed.** AMD's live product specification identifies the processor
as four Zen 5 plus eight Zen 5c cores, with a 5.1 GHz maximum boost and a
3.3 GHz maximum Zen 5c clock. It also documents AVX2, AVX-512, and FMA3:
[AMD Ryzen AI 9 HX 370 specification](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html),
accessed 2026-07-26.

**Measured.** The analysis tools were Clang/LLVM 21.1.8 and `perf` 7.0.12.
The MDSLC Release binaries used `-O3 -DNDEBUG`. The authenticated external
provider was OpenBLAS 0.3.32, pthread, LP64. Exact provider and benchmark
provenance are embedded in every retained JSON file.

**Measured limitation.** `kernel.perf_event_paranoid=4`. This command failed
with exit status 1 before executing `true`:

```sh
perf stat \
  -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,\
dTLB-loads,dTLB-load-misses,stalled-cycles-frontend,stalled-cycles-backend \
  -- true
```

The diagnostic was `No supported events found` followed by the kernel
permission explanation. No system policy was changed. Consequently:

| Requested observation | Status |
| --- | --- |
| cycles, instructions, hardware IPC | unavailable |
| cache references and misses | unavailable |
| DTLB loads and misses | unavailable |
| branch count and mispredicts | unavailable |
| frontend/backend stalled cycles | unavailable |
| APERF/MPERF and per-sample effective frequency | unavailable |
| memory-controller/fabric traffic | unavailable |

**Source-backed.** The access model is consistent with the kernel's
[Perf events and tool security](https://docs.kernel.org/admin-guide/perf-security.html)
documentation. Static scheduling and elapsed time cannot substitute for these
missing events.

## Models and formulas

The roofline convention follows Williams, Waterman, and Patterson,
["Roofline: an insightful visual performance model for multicore
architectures"](https://doi.org/10.1145/1498765.1498785).

For row-major F32 GEMM:

```text
F = 2 M N K                                         FLOP
B_ideal = 4 (M K + K N + M N)                      bytes
AI_ideal = F / B_ideal                              FLOP/byte
B_write_allocate = 4 (M K + K N + 2 M N)           bytes
```

`B_ideal` is an optimistic compulsory-payload lower bound: A and B are read
once and C is written once. `B_write_allocate` exposes one possible extra C
read-for-ownership. Neither is a measured DRAM byte count.

The current native loop order has `MR=4`, `NR=16`, `MC=128`, `NC=256`, and
`KC=256`. An outer-loop, code-visible packing/output payload model is:

```text
Q = ceil(N / NC)
P = ceil(K / KC)
B_native_outer =
    4 [2 K N + 2 Q M K + (2 P - 1) M N]             bytes
AI_native_outer = F / B_native_outer                 FLOP/byte
```

This counts read-plus-write for B packing, repeated read-plus-write for A
packing once per `NC` panel, and the C store for the first `KC` slab plus a
load/store pair for later slabs. It intentionally does not pretend to know
which accesses reach DRAM. Packed panels are designed to remain in cache, and
the microkernel rereads them from the cache hierarchy.

For one complete 4×16 microtile and one K step, either current microkernel
performs 128 FLOP and consumes 16 packed-B floats plus four packed-A floats:

```text
AI_microkernel_input = 128 / (20 * 4) = 1.6 FLOP/byte
```

That is an L1-facing input model, not a DRAM arithmetic intensity.

## Bounded bandwidth measurement

**Measured.** A standalone 64-byte-aligned C++20 probe used one persistent
thread team per kernel, pinned worker `i` to logical CPU `i`, first-touched
three arrays in parallel, and timed vectorized copy and triad loops. Clang
reported vector width 16 for both timed loops. Initialization, thread
creation, checksum, and allocation were outside the interval. Every retained
sample exceeded 3 ms. Three warmups and nine measured samples were used.

The byte convention is application payload:

```text
copy  = read A + write C       = 2 * array bytes
triad = read A + read B + write C = 3 * array bytes
```

Write-allocate traffic is not added. Therefore this is not a memory-controller
counter and must not be reported as physical bus utilization.

### Single-core footprint sweep

| Three-array footprint | Intended level on CPU 0 | Copy GB/s | Triad GB/s |
| ---: | --- | ---: | ---: |
| 48 KiB | L1D-scale | 629.81 | 445.55 |
| 768 KiB | L2-scale | 311.80 | 224.55 |
| 12 MiB | local L3-scale | 149.85 | 136.97 |
| 192 MiB | beyond LLC | 37.06 | 38.76 |
| 768 MiB | beyond LLC | 35.69 | 38.90 |

**Measured.** The retained operational memory-bandwidth ceiling is
38.90 GB/s of triad payload. The 768 MiB footprint is 32 times the 24 MiB
aggregate LLC and 64 times CPU 0's 12 MiB three-array local-LLC footprint.

**Measured caveat.** A three-run, interleaved 256 MiB-per-array thread sweep
did not increase payload bandwidth:

| Threads | Copy median GB/s | Triad median GB/s |
| ---: | ---: | ---: |
| 1 | 35.80 | 38.34 |
| 4 | 33.47 | 38.20 |
| 12 | 30.08 | 34.83 |
| 24 | 28.47 | 33.04 |

**Derived.** CPU 0 alone saturates this probe's operational bandwidth. Equal
chunks assigned across the heterogeneous core set make the slowest worker end
the interval, so this sweep is not a claim that additional cores reduce the
physical memory-controller capability. It is a bounded bandwidth ceiling for
the current host state and an intentionally simple static partition.

**Source-backed.** AMD describes STREAM as a sustainable-memory-bandwidth
benchmark and recommends explicit placement and thread control. The audit
probe follows those principles but is not the official AMD binary:
[AMD Zen STREAM, version 2024-10-08](https://www.amd.com/en/developer/zen-software-studio/applications/pre-built-applications/zen-stream.html).

## Static arithmetic ceilings

The exact Release microkernel symbols were isolated with:

```sh
objdump -d -Mintel \
  --disassemble=matcore_cpu_packed_avx2_4x16_microkernel_f32_v1 \
  libmatcore_runtime.so.0.0.0

objdump -d -Mintel \
  --disassemble=matcore_cpu_packed_avx512_4x16_microkernel_f32_v1 \
  libmatcore_runtime.so.0.0.0
```

**Measured artifact.** The AVX2 K loop contains two 256-bit packed-B loads,
four scalar broadcasts, and eight YMM packed FMAs. The AVX-512 K loop contains
one 512-bit packed-B load and four ZMM packed FMAs with scalar broadcast memory
operands. No scalarized body is being modelled.

**Static model.** `llvm-mca-21 -mcpu=znver5 -iterations=100` produced:

| Inner loop | Instructions/iteration | FLOP/iteration | Model block throughput | Model FLOP/cycle |
| --- | ---: | ---: | ---: | ---: |
| independent scalar FMA | 6 | 8 | 4 cycles | 2 |
| current AVX2 4×16 | 17 | 128 | 4 cycles | 32 |
| current AVX-512 4×16 | 8 | 128 | 4 cycles | 32 |

The AVX2 model reports 4.10 static uops/cycle and pressure distributed across
the model's FP and load resources. The AVX-512 model reports 1.94 static
uops/cycle; each memory-broadcast ZMM FMA has reciprocal throughput 1.0, so
the four independent accumulators still require four cycles. These are LLVM
model results, not hardware IPC. See the
[LLVM `llvm-mca` command guide](https://llvm.org/docs/CommandGuide/llvm-mca.html).

**Derived nominal ceilings.** Multiplying the model FLOP/cycle by sysfs policy
maximum frequencies gives:

| Scope | Scalar GFLOP/s | AVX2/AVX-512 GFLOP/s |
| --- | ---: | ---: |
| one Zen 5 policy domain at 5.157 GHz | 10.32 | 165.05 |
| one Zen 5c policy domain at 3.289 GHz | 6.58 | 105.26 |
| 4 Zen 5 + 8 Zen 5c, all at policy maxima | 93.89 | 1502.32 |

The all-core number is deliberately optimistic: simultaneous maximum boost is
not measured, and the processor has a 15–54 W configurable power envelope.
It is an upper modelling reference, not achieved silicon throughput.

**Derived roofline knees.** Using the measured 38.90 GB/s payload ceiling, the
nominal roofline crossover is 4.24 FLOP/byte for one CPU 0 vector core and
38.62 FLOP/byte for the optimistic all-core vector ceiling.

## Representative roofline and bottleneck classification

The current guarded JSON results below used clean Release binaries, 64-byte
alignment, hot cache, caller-reused workspace, packing included unless
explicitly prepacked, and an independent double-precision oracle. Every quoted
result passed correctness.

| Region | Ideal AI | Native outer AI | Retained throughput | Classification |
| --- | ---: | ---: | --- | --- |
| 64³ | 10.67 | 6.40 | native AVX2 101.35 GF/s | latency plus compute |
| 128³ | 21.33 | 12.80 | native AVX2 122.16; compute diagnostic 127.29 | execution-port/compute |
| 512³ | 85.33 | 28.44 | native AVX2 132.80; prepacked 136.52 | execution-port/cache |
| 1024³, 1 thread | 170.67 | 30.12 | native AVX2 132.02; AVX-512 133.47; OpenBLAS 149.35 | execution-port/cache |
| 1×4096×4096 | 0.50 | 0.25 | native transient 4.17; prepacked 22.58 | packing, then bandwidth/cache |
| 8×4096×4096 | 3.98 | 1.88 | native transient 21.44; prepacked 99.29 | packing, then bandwidth/cache |
| 64×1024×1024 | 28.44 | 10.89 | native transient 115.13; prepacked 135.22 | mixed packing and compute |
| 1024×64×1024 | 28.44 | 12.49 | native transient 122.56; prepacked 124.11 | execution/cache, not B packing |
| 31×33×35 | 5.49 | 3.25 | native transient 51.40 | latency/tail/packing |
| 4096³, 12 threads | 682.67 | 31.51 | native AVX2 590.10; OpenBLAS 1085.91 | native cache/data movement/synchronization |
| 4096×4096×64, 1 thread | 31.03 | 20.90 | native AVX2 135.85; OpenBLAS 123.66 | compute/execution |

### Serial square GEMM

**Derived, high confidence.** DRAM bandwidth is not the primary limit for
serial 128³–1024³. Even the deliberately pessimistic native-outer intensities
place their 38.90 GB/s memory roofs at 498–1172 GFLOP/s, well above the
165.05 GFLOP/s static CPU0 execution ceiling.

**Derived, medium confidence.** At 1024³, native AVX2 achieves 80.0% of the
nominal CPU0 model ceiling and OpenBLAS achieves 90.5%. Prepacking B improves
native 1024³ by only 2.0% in the current data-movement sweep. The remaining
serial gap is therefore primarily in microkernel scheduling and cache-level
data supply, not transient B packing or main memory. Actual effective core
frequency and hardware port counters are unavailable, so the two causes cannot
be separated numerically.

**Static model, high confidence.** The current AVX-512 body has the same
32 FLOP/cycle model throughput as AVX2. Its wider instructions halve the FMA
instruction count but also halve the accumulator count from eight YMM
accumulators to four ZMM accumulators and use memory-broadcast FMAs with model
reciprocal throughput 1.0. AVX-512 is not entitled to a 2× performance
assumption.

### Vector-like and aspect-ratio-sensitive GEMM

**Measured, high confidence.** Transient B packing is dominant at
1×4096×4096 and 8×4096×4096. Prepacking raises throughput 5.42×
(4.17→22.58 GFLOP/s) and 4.63× (21.44→99.29 GFLOP/s), respectively.

**Derived, medium confidence.** Their ideal arithmetic intensities are only
0.50 and 3.98 FLOP/byte. Once B packing is removed, these cases move toward a
bandwidth/cache ceiling rather than the 165 GFLOP/s arithmetic ceiling. The
prepacked M=1 result exceeds the simple 38.90 GB/s × 0.50 roof because the
hot-cache/prepacked execution does not have the same physical traffic as the
single-pass beyond-LLC probe. That disagreement is evidence not to relabel
payload bytes as measured DRAM traffic.

**Measured, high confidence.** B prepacking changes 1024×64×1024 by only 1.3%,
but changes 64×1024×1024 by 17.5%. One static blocking and packing order is
therefore shape-asymmetric.

### Parallel large GEMM

**Derived, medium confidence.** At 4096³, the native outer-payload intensity
is 31.51 FLOP/byte. Its operational memory roof is 1225.7 GFLOP/s, below the
optimistic 1502.3 GFLOP/s all-core execution ceiling. OpenBLAS reached
1085.9 GFLOP/s (72.3% of the optimistic arithmetic ceiling and 88.6% of that
operational payload roof), while native AVX2 reached 590.1 GFLOP/s (39.3% and
48.1%). The providers do not have identical internal traffic, so these
percentages are diagnostic bounds, not physical bandwidth utilization.
The retained Milestone 5 native run used compact worker affinity, whereas
OpenBLAS owned unbound provider scheduling. Requested and actual thread counts
were equal, but placement was not identical; this is not a native/BLAS parity
claim.

**Measured/source-backed, high confidence.** Native parallel work is divided
only into `MC=128` row bands, and B is packed once into shared caller-owned
workspace before task dispatch. A 1024³ request for 12 native threads exposes
only eight row-band tasks and reports eight actual threads; it is not a fair
12-thread comparison with OpenBLAS. At 4096³ there are enough row bands, but
native still delivers about half OpenBLAS's 12-thread throughput.

**Hypothesis, medium confidence.** The large parallel deficit is a combination
of cache-level packed-panel traffic, row-only decomposition, heterogeneous-core
load balance, and synchronization/worker dispatch. Hardware cache, stall, and
fabric counters are required to rank those contributors quantitatively.

### Tiny and tails

**Measured, high confidence.** Full and edge 4×16 diagnostics with small K
remain near a roughly 2 μs complete microkernel-call floor before useful
throughput scales. The 31×33×35 complete native result reaches only
51.40 GFLOP/s even though its simple operational memory roof is
126.55 GFLOP/s.

**Derived, medium confidence.** Small/tail regions are latency- and
edge-handling-bound rather than pure bandwidth-bound. At 511-scale the
performance loss relative to nearby full tiles is much smaller, so "tails are
always dominant" is not supported.

## Requested bound classifications

| Bound class | Current conclusion |
| --- | --- |
| latency-bound | measured for tiny calls and small tail-heavy calls |
| packing-bound | measured for transient M=1/M=8 wide calls |
| bandwidth-bound | derived for vector-like prepacked operation; physical DRAM counters unavailable |
| cache-bound | supported as part of large parallel and rectangular deficits, but not separable from execution without counters |
| frontend-bound | not supported by the static inner-loop model; hardware frontend stalls unavailable |
| execution-port-bound | static model and serial-square roofline support it as a primary inner-loop constraint |
| synchronization-bound | contributes to parallel calls; isolated worker-dispatch evidence belongs to the parallel-runtime audit |
| compute-bound | serial medium/large square and 4096×4096×64 lie above the single-core roofline knee |

## Root causes ranked

1. **Measured + static model; high impact/high confidence — microkernel
   throughput.** Native serial 1024³ leaves about ten percentage points more of
   the nominal CPU0 ceiling unused than OpenBLAS. AVX-512's current four-chain
   body has no static throughput advantage over the eight-chain AVX2 body.
2. **Measured; high impact/high confidence — aspect-ratio-dependent packing.**
   M=1/M=8 improve 5.42×/4.63× with prepacked B. A is repacked once per NC
   panel, so very wide N multiplies A packing.
3. **Measured/source-backed; high impact/high confidence — parallel
   decomposition.** Row bands cap usable workers at `ceil(M/128)` and cannot
   expose N parallelism for short-wide cases. At 4096³ the native 12-thread
   path is 54% of equal-thread OpenBLAS throughput.
4. **Derived; high impact/medium confidence — one static blocking
   configuration.** The native outer-payload intensity asymptotes near
   32 FLOP/byte for large square matrices, below the optimistic all-core
   roofline knee, while the shape sweep shows opposite B-prepack sensitivity
   for short-wide versus tall-narrow inputs.
5. **Measured; medium impact/high confidence — tiny/tail latency.** Fixed
   call, validation, packing, and edge-copy costs dominate before enough
   K work is present.
6. **Hypothesis; medium impact/low confidence — AVX-512 frequency effects.**
   Prior 4096³ AVX-512 one-thread results were degraded and variable, but
   APERF/MPERF and per-sample frequency were unavailable. No downclock
   magnitude is claimed.
7. **Hypothesis; unknown impact/low confidence — TLB misses, hardware
   prefetch quality, and software prefetch.** The required counters are
   blocked. No prefetch change is justified by this lane alone.

## Milestone 7 experiments proposed by this lane

- **Proposed.** Add a larger-accumulator AVX-512 microkernel and at least one
  alternate AVX2 MR/NR kernel; accept only variants whose exact symbols improve
  static dependency/resource pressure and guarded complete-call performance.
- **Proposed.** Add explicit shape-family blocking tables rather than one
  universal MC/NC/KC tuple. Keep a held-out matrix so the rules cannot be
  selected from one favorable square.
- **Proposed.** Add no-pack or persistent-prepack paths for M≤8 and explicitly
  owned shared packed-B reuse. Do not add an implicit global cache.
- **Proposed.** Add N-tile tasks for short-wide GEMM and compare row, column,
  and 2-D output-tile ownership. Keep K partitioning out until a deterministic
  reduction contract exists.
- **Proposed.** Add a benchmark-only dispatch/null-task timing mode to isolate
  persistent-worker coordination from packing and compute.
- **Proposed.** Collect privileged hardware counters only in a separately
  approved environment. At minimum: cycles, instructions, effective
  frequency, L1/L2/LLC demand misses, DTLB misses, branch misses, FP retired
  work, and memory-controller bytes.

None of these proposals changes production behavior in Milestone 6.

## Reproduction and raw evidence

The small bandwidth probe, disassemblies, `llvm-mca` inputs/outputs, permission
diagnostic, and environment snapshot are external raw artifacts under:

```text
/home/hamza-usta/.tmp/mdslc-m6-roofline/
```

The SHA-256 of the sorted per-file SHA-256 inventory is:

```text
8679bc39feca726004d26d6317a541e4093f863745d0a0f1a64ced8c54bfccc9
```

Key raw hashes:

| Artifact | SHA-256 |
| --- | --- |
| bandwidth probe source | `981fe3bb49a611b9c8dce00839221699764a2212b7e5268a34b43f5795db344d` |
| cache/beyond-LLC sweep | `28eea156defa9f97999de54c06e9ce3fd7445b9358bb1b8dd3e9986e78136572` |
| interleaved thread sweep | `3f1432dfdb3e7da4a13c9dba7b171fb1734bf46d04a72f645dd800507746302d` |
| AVX2 disassembly | `96199cfc3f308b5f01cf2df6147208a63feff95289bab921231b03d7f424c22b` |
| AVX-512 disassembly | `4e5b810110ad3b31910fdc5c57839d45e9308af68adb12e72b3df69b512c2bed` |
| AVX2 `llvm-mca` output | `f6587e2f6472aec8b3ef1cbe3b114145068e0f505b2cd9781dd3ab66704836be` |
| AVX-512 `llvm-mca` output | `1c644ec9c411a41ffefe005d841b260f8017b08963f6596825d02e4109f5c345` |
| blocked `perf stat` diagnostic | `e51a366d5de5265f80f779ed824ff0a0b660a2dce8e39ce4145b7c1c2f7bc4dd` |
| environment snapshot | `8c0f1e8ecdfe3cbccbc76ba33b3ddb9128ff67b30b161c8837b5885e9e9c2c48` |

The source-checkpoint data-movement bundle contains 162 guarded JSON files;
the SHA-256 of its sorted per-file digest inventory is:

```text
6e5a9013947c0e76a917c2db5950aa937310737757002018502739601344b268
```

The audit-runner 1024³ smoke bundle digest is:

```text
84b35bb6e62807503b3566a5afdf43769f639efc245c8386e9def7a872b9e294
```

Representative commands:

```sh
clang++-21 -O3 -DNDEBUG -std=c++20 -march=znver5 -pthread \
  -Wall -Wextra -Werror -Rpass=loop-vectorize \
  -Rpass-missed=loop-vectorize bandwidth_probe.cpp -o bandwidth_probe

# 256 MiB per array, one thread, 3 warmups, 9 samples.
./bandwidth_probe 268435456 1 3 9 1

llvm-mca-21 -mcpu=znver5 -iterations=100 avx2-inner.s
llvm-mca-21 -mcpu=znver5 -iterations=100 avx512-inner.s
```

Raw benchmark output, object dumps, profiler diagnostics, binaries, and probe
source are intentionally not committed.

## Audit verdict

**Derived verdict.** The present single-thread square engine is primarily an
execution/cache-level optimization problem, not a DRAM problem. The
vector-like region is demonstrably packing-sensitive and then
bandwidth/cache-sensitive. The large parallel region is limited by current
data movement and row-only scheduling well before it reaches the host's
optimistic arithmetic ceiling.

**Measured limitation.** Hardware IPC, cache/TLB misses, stalled cycles,
physical memory bytes, and effective AVX frequency remain unmeasured because
counter access is blocked. Any stronger microarchitectural attribution would
be fabricated.
