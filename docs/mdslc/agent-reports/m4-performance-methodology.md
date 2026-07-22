# Milestone 4 performance-methodology lane report

Date: 2026-07-22

## Ownership and commits

This lane owned only:

- `docs/mdslc/PERFORMANCE_PREFLIGHT.md`;
- ADR 0006, the CPU benchmark contract;
- `compiler/tools/matcore-bench/`;
- `compiler/tests/benchmark/`;
- this report.

It did not edit the root compiler CMake file, runtime, planner, public ABI,
`.gitignore`, legacy implementation, or existing shared tests.

Focused commits:

1. `433a7e19521fe65565f438a7ad780de956be60e2`
   `docs(perf): freeze CPU benchmark contract and preflight`
2. `544e3bc617135bea196d798bec08095eedf0360a`
   `feat(perf): add versioned CPU GEMM benchmark tool`

## Delivered benchmark contract

The new standalone `matcore-bench` subproject provides:

- explicit custom, quick, standard, and opt-in full shape profiles in M,N,K
  order;
- automatic or exact stable-ID variant selection;
- requested and actual thread metadata;
- one-shot allocation, reusable-workspace, packing-included,
  packing-excluded, and prepacked-B distinctions;
- hot-cache aggregation and best-effort cold-cache measurement;
- a configurable timer floor and explicit rejected-timing state;
- checked allocation arithmetic and a configurable hard memory cap;
- exact requested data alignment;
- full independent double-precision checking for bounded shapes;
- deterministic sampled double checking plus independent global checksum for
  large shapes;
- all-output NaN/infinity rejection;
- minimum, median, nearest-rank p95, and SGEMM GFLOP/s;
- compiler, flags, build type, source commit, CPU, affinity, governor,
  frequency policy, boost, timer, capability, provider, workspace, stride,
  packing, and selected-plan metadata;
- schema-v1 JSON with paths containing spaces tested;
- an internal runner interface that declares all workspace and prepacked-B
  storage and performs no hidden allocation inside `execute()`.

The current adapter deliberately exposes only the already implemented
reference, tiled, and compiler-vectorized planner-v1 variants. It rejects
prepacked-B and thread counts above one. A typed opaque plan state prevents
replanning inside reused-workspace timing while allowing one-shot mode to
include planning.

## Toolchain and dependency evidence

The primary build used Clang 21.1.8 with CMake 4.3.2 and Ninja 1.13.2. A
coherent OpenBLAS 0.3.32 pthread development provider was independently
compile/link/run verified, including row-major `cblas_sgemm` and local
single-thread control. No package was installed. The adapter itself remained
outside this lane so BLAS ownership and runtime ABI work can integrate without
overlapping edits.

The host is an AMD Ryzen AI 9 HX 370 with 12 physical cores, 24 logical CPUs,
one NUMA node, two LLC/frequency groups, performance governor/EPP, and boost
enabled. Full facts and the limitation on ISA claims are in the preflight.

## Validation evidence

Fresh standalone Release build:

```text
/tmp/matcore-bench-final-release.RdGaa2
benchmark.cpu.contract  PASS
benchmark.cpu.cli_json  PASS
2/2 passed
```

Fresh Debug build:

```text
/tmp/matcore-bench-debug.XY9xI0
2/2 passed
```

ASan plus UBSan build:

```text
/tmp/matcore-bench-sanitize.5y3el7
ASAN_OPTIONS=detect_leaks=1
UBSAN_OPTIONS=print_stacktrace=1
2/2 passed
```

The same two tests also passed with GCC 15.2.0 as a secondary portability
check. A fresh-prefix install produced `bin/matcore-bench`; the installed tool
executed a forced reference 2x3x2 GEMM with valid timing and independent
correctness.

A Release quick-profile guard was pinned with `taskset -c 0`. All eight shapes
passed correctness and the 1 ms aggregate timing floor. Automatic planning
selected reference for 1x1x1 and 2x3x2, tiled for 16x16x16, and the existing
compiler-vectorized AVX2/FMA variant for the remaining five shapes. These
results validate the harness path only; they are not retained as planner
calibration or universal performance evidence.

Additional adversarial checks covered:

- invalid alignment;
- dimension/storage overflow;
- pre-allocation memory cap;
- conflicting prepack/allocation modes;
- unsupported stable variant;
- unsupported thread count;
- unavailable prepacked-B path;
- sub-floor cold-cache timing rejection with correctness preserved;
- JSON parsing and required schema fields;
- stdout-only JSON without human-text contamination;
- paths containing spaces.

Repository hygiene and `git diff --check` passed. Raw JSON remained under
`/tmp`; no benchmark run output or build artifact was added to Git.

## Integration handoff

The integration owner needs one non-overlapping root build change:

```cmake
add_subdirectory(tools/matcore-bench)
```

The subdirectory already registers its two tests and its executable install
rule. The root should place it after the planner target exists. No benchmark
library needs to be exported as a public C++ ABI.

The OpenBLAS/native-packed integration should either extend the composite
runner in `planner_runner.cpp` or replace it with a runtime-backed runner while
preserving `GemmRunnerV1`. Each new plan must populate actual threads,
workspace bytes/alignment, packing requirements, prepacked-B storage,
provider metadata, deterministic diagnostic, and opaque executable state.

## Limitations

- No OpenBLAS or packed-kernel implementation was added by this lane.
- Current planner-v1 variants require one thread and zero workspace.
- Cold-cache eviction is best-effort and explicitly labeled; it is not a cache
  hardware guarantee.
- The runner interface is internal and intentionally not a stable installed
  C++ ABI.
- Clang-format was unavailable; all new C++ compiled under warning-as-error
  with Clang and GCC.
- Windows compilation was not performed locally. The new CMake warning flags
  distinguish MSVC/clang-cl, and non-Linux environment fields fail closed as
  `unknown`, but Windows remains a later hosted validation gate.
