# CPU performance deep-audit methodology

Milestone: MDSLC Milestone 6 — CPU Performance Deep Audit  
Host calibration date: 2026-07-26  
Repository base:
`951239f1bee5541a4cf5ad72fab2192de07cf89d`

This document freezes the evidence contract before any Milestone 7
optimization. Milestone 6 may improve benchmark instrumentation and
diagnostics, but it does not change automatic planner behavior or the public
ABI.

## Claim vocabulary

Every conclusion in the Milestone 6 audit uses one of these labels:

- **measured**: produced by an identified executable, source commit, host,
  command, and timing or artifact-inspection method;
- **derived**: calculated from measured or architectural inputs with the
  formula shown;
- **source-backed**: supported by an identified source revision or technical
  reference;
- **hypothesis**: plausible but not established by the available evidence;
- **proposed**: a candidate future change that has not been accepted as
  production behavior.

Raw measurements, disassemblies, profiler output, logs, and generated JSON
remain outside Git. Only reviewed summaries and small intentional textual
fixtures may be committed.

## Validation host and observability boundary

The physical calibration host is an AMD Ryzen AI 9 HX 370 running Ubuntu 26.04
and Linux 7.0.0-27-generic. It has one socket, twelve physical cores,
twenty-four logical CPUs, two LLC groups, one NUMA node, and heterogeneous
maximum-frequency domains. The coherent compiler tuple is Clang/LLVM 21.1.8.
The external baseline is OpenBLAS 0.3.32, pthread, LP64.

The governor and energy preference are `performance`, and boost is enabled.
Frequency remains observational rather than fixed. Single-thread runs use a
declared logical CPU; multi-thread runs record the exact placement selected by
the runtime and keep the benchmark caller off worker cores where the topology
permits.

Linux exposes `perf`, but `kernel.perf_event_paranoid=4` blocks unprivileged
software and hardware counters. Milestone 6 therefore distinguishes:

- dynamic timings and correctness, which are available;
- static disassembly and `llvm-mca-21 -mcpu=znver5` analysis, which are
  available;
- hardware-counter claims, which are unavailable unless separately collected
  under an explicitly authorized privileged policy.

Static scheduler modelling is not reported as a measured hardware counter.
No system performance policy is changed by the audit.

## Compared implementations

The complete implementation-call comparison set is:

```text
cpu.reference.f32.v1
cpu.tiled.f32.v1
cpu.compiler-vectorized.avx2-fma.f32.v1
cpu.native-packed.avx2-fma.f32.v1
cpu.native-packed.avx512-fma.f32.v1
cpu.native-parallel.avx2-fma.f32.v1
cpu.native-parallel.avx512-fma.f32.v1
cpu.external.openblas.f32.v1
```

AVX-512 runs only when the capability record, OS state, compiled
implementation, exact artifact check, and numerical runtime validation all
succeed. OpenBLAS is compared only when the coherent adapter is linked.
Forced unavailable variants fail; they are not replaced by fallback results.

OpenBLAS and native thread counts are explicit. Native and provider pools are
never nested. A planner result that selects OpenBLAS is not evidence of native
kernel throughput.

## Shape matrix

Dimensions are written as `M x N x K`.

### Small square

```text
4, 8, 16, 24, 32, 48, 64
```

### Medium square

```text
96, 128, 192, 256, 384, 512
```

### Large square

```text
768, 1024, 1536, 2048, 4096
```

### Tall-skinny

```text
4096 x 64  x 4096
4096 x 128 x 1024
8192 x 32  x 1024
2048 x 256 x 4096
```

### Short-wide

```text
64  x 4096 x 4096
128 x 4096 x 1024
32  x 8192 x 1024
256 x 2048 x 4096
```

### Vector-like

```text
1    x 4096 x 4096
8    x 4096 x 4096
4096 x 4096 x 1
4096 x 4096 x 8
```

### Tail-heavy

```text
31  x 33  x 35
63  x 65  x 67
127 x 129 x 131
255 x 257 x 259
511 x 513 x 515
```

### Repeated, fixed B

The prepacked-B experiment uses one fixed B and 1, 4, 16, and 64 repeated A
inputs. The report distinguishes one-time preparation, each complete
prepacked execution, and the amortized total. Reusing a timed output without
executing GEMM is prohibited and correctness is authenticated after timing.

## Thread and placement matrix

The primary thread set is:

```text
1, 2, 4, physical-core-count
```

The logical-CPU count is a separate SMT experiment, never folded silently into
the physical-core result. Impossible or duplicate counts are removed
deterministically. Single-thread results are pinned separately from
multi-thread results because this host contains heterogeneous core/frequency
domains and two LLC groups.

Provider and native comparisons require equal requested and actual thread
counts. Results with incomplete affinity, unavailable cores, or provider
thread-count ambiguity remain diagnostic and are excluded from parity or
regret aggregates.

## Measurement modes

Each retained result names exactly one timing scope:

1. **End-to-end one-shot** includes planning, workspace preparation,
   allocation, packing, compute, and synchronization.
2. **Reused workspace, packing included** excludes allocation and includes
   preparation/packing, compute, and synchronization.
3. **Prepacked B** excludes B packing from repeated execution and reports the
   separate one-time B preparation cost.
4. **Compute-only diagnostic** prepares packed operands outside the interval
   and times only the internal packed compute call plus required
   synchronization. It is not compared directly with complete CBLAS SGEMM as
   an end-to-end claim.

Hot-cache is the primary reproducible comparison. Cold-cache is a separate,
best-effort 64 MiB eviction experiment and is not claimed to model a
particular application cache state.

Allocation, input initialization, and the double-precision oracle are excluded
from every complete-call timing except the explicitly named one-shot
allocation mode. The benchmark records the actual scope in each result.

## Sampling and correctness

- Inputs use a fixed recorded seed.
- Warmups execute the same implementation and mode as measured samples.
- Complete-call hot-cache aggregates run until the configured one-millisecond
  timer floor.
- Retained runs use at least seven measured aggregate samples; primary
  comparisons use eleven where the total matrix remains bounded.
- Minimum, median, and nearest-rank p95 are retained.
- Every timed final output is authenticated.
- Small problems use a full independent double-precision element oracle.
- Larger problems use the benchmark's independent double-precision checksum
  oracle with its recorded error bound.
- NaN, infinity, timer-floor violations, provenance ambiguity, plan mismatch,
  and output corruption reject guarded runs.

SGEMM throughput uses:

```text
operations = 2 * M * N * K
GFLOP/s = operations / median_seconds / 1e9
```

## Experiment partition

To prevent Milestone 7 from selecting only favorable shapes, the audit fixes
three partitions before optimization:

- **diagnostic**: small square, tail-heavy, and vector-like shapes used to
  identify overhead and boundary behavior;
- **calibration**: medium square plus the first two shapes from each
  tall-skinny and short-wide family;
- **holdout**: large square, the remaining rectangular shapes, and repeated
  prepacked-B cases.

Milestone 7 may derive implementation rules from diagnostic and calibration
evidence. The declared parity envelope must report the unchanged holdout set
as well. A result may be excluded only for a predeclared timer, memory,
capability, correctness, or execution-completeness reason, and the exclusion
must remain visible.

## Reproducibility and raw-data policy

The benchmark emits schema-v4 JSON with exact source provenance, compiler
flags, environment, capability/topology records, provider information,
workspace and packing state, placement, correctness, timing, and plan
metadata. A raw-run directory must:

- be outside the source tree or below ignored `benchmark_reports/`;
- contain a manifest with the exact command sequence;
- never be added to Git;
- be retained externally long enough for independent review.

Sanitized summaries quote the source commit and a SHA-256 inventory of the raw
bundle. They contain enough rows to reproduce every aggregate claim but omit
large raw samples, generated binaries, logs, and profiler databases.

## Milestone 6 change boundary

Permitted changes are benchmark/reporting instrumentation, reproducible
collection helpers, small validation fixtures, and documentation. Production
kernel selection, planner thresholds, C ABI layouts, and public language
surface remain unchanged until:

1. the audit ranks root causes by impact and confidence;
2. candidate changes identify correctness, legality, and measurement gates;
3. an independent reviewer accepts the fairness and conclusions;
4. the Milestone 6 PR passes hosted validation and is merged.

