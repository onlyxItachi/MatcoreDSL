# Matcore IR v1 and CPU planner final evidence

Date: 2026-07-21

## Scope and verdict

Milestone 2 implements the following executed path:

```text
matcore::mdsl::gemm
  -> authenticated native Clang capture
  -> verified Matcore JSON IR v0 compatibility record
  -> typed and verified Matcore IR v1
  -> lossless projection for the existing rewrite/codegen boundary
  -> validated v0 C tensor descriptors
  -> CPU capability discovery
  -> legality and deterministic cost evaluation of every registered variant
  -> selected-plan metadata and diagnostics
  -> selected reference, tiled, or compiler-vectorized CPU lowering
  -> synchronous native execution
```

No CUDA, HIP, Metal, NPU, heterogeneous placement, fusion, WGMMA, MFMA, or
autotuning work is included.

Local implementation and validation verdict: **PASS**. The independent review
also returned **PASS** after finding and fixing one Debug-lowering defect.
Final GitHub results are recorded below before merge.

## Exact Matcore IR v1 schema

The C++ schema is `matcore::mdslc::ir::v1` and the JSON envelope is exactly:

- `schema`: literal `matcore.ir`;
- `version`: integer `1`;
- `producer`: `clang-libtooling-v1` or the explicit
  `clang-ast-json-bootstrap-v0` compatibility producer;
- `translation_unit`: exact `identity` and `.mdsl` `source_file` strings;
- `operations`: ordered typed operations.

Each v1 operation contains:

- canonical lowercase unique `site_id`, `kind=gemm`, and
  `canonical_callee=matcore::mdsl::gemm`;
- source file, line, column, byte offset, nonempty half-open call range, and
  three or four ordered argument ranges;
- one output and ordered lhs/rhs operands;
- element dtype from `f16`, `bf16`, `f32`, `f64`, `i8`, `i32`;
- accumulation dtype with legal pairs `f16/bf16->f32`, `f32->f32`,
  `f64->f64`, `i8->i32`, and `i32->i32`;
- rank, shape, and strides, where every scalar expression is either a positive
  static `uint64` or a canonical dynamic symbol scoped to that operation;
- layout from row-major contiguous, column-major contiguous, or strided;
- host/device memory space and power-of-two required alignment at least as
  large as the dtype's natural alignment;
- read-only, write-only, or read-write mutability;
- accumulation dtype and canonical semantic requirements;
- ordered no-alias requirements, read/write effects, synchronization, target,
  and fallback policy.

The current verified GEMM semantic subset is deliberately narrower than the
type vocabulary: rank-2 host `f32` inputs/output, `f32` accumulation, exact
M/K/N equality, output write-only, inputs read-only, output no-alias with each
input, synchronous effects, `target=cpu`, and `fallback=error`. Row-major,
column-major, and structurally valid strided tensor types are representable;
only the canonical row-major dynamic subset can cross the current executable
v0 projection.

JSON parsing requires exact members, valid UTF-8, known enum spellings, fixed
semantic ordering, and a supported version. Serialization uses a fixed key
order, two-space indentation, and one trailing newline. Unknown versions are
reported once and are never retried as another schema.

## v0-to-v1 boundary

Direct `matcore-extract` output remains v0 by default for compatibility.
`--ir-version=1` emits typed v1, and `mdslc++` explicitly requests v1.

Every captured v0 module is verified, upgraded, verified as v1, projected back
to v0, and verified again before rewrite/codegen. The canonical upgrade is:

| Value | Shape | Strides | Other contracts |
| --- | --- | --- | --- |
| lhs | `[m,k]` | `[k,1]` | host f32, row-major, read, alignment 4 |
| rhs | `[k,n]` | `[n,1]` | host f32, row-major, read, alignment 4 |
| output | `[m,n]` | `[n,1]` | host f32, row-major, write, alignment 4 |

Projection rejects static dimensions, renamed canonical symbols, noncanonical
strides/layout, non-host/f32 values, alignment other than 4, changed effects,
aliasing, requirements, target, fallback, provenance, or ranges. This prevents
silent information loss while preserving byte-stable v0 output and the
existing `matcore_runtime_gemm_f32_v0` execution ABI.

## CPU capability model

`CpuCapabilitiesV1` contains version, architecture (`unknown`, `x86_64`, or
`aarch64`), discovery completeness, known feature bits, and usable vector
width. Portable scalar f32 is always modeled. x86-64 discovery uses compiler
CPU builtins for AVX2 and FMA; the compiler-vectorized candidate is never legal
when discovery is incomplete. Sanitizer-instrumented builds deliberately expose
only capabilities usable by their instrumented lowering, so they do not select
or mislabel a scalarized vector candidate.

Malformed records fail closed: unsupported versions or architectures, unknown
feature bits, complete discovery with unknown architecture, AVX2 on non-x86,
or vector widths outside the v1 domain are invalid capability records.

Detected on the validation host:

```text
arch=x86_64
detection_complete=true
features=[portable-scalar-f32,avx2,fma]
usable_vector_bits=256
CPU=AMD Ryzen AI 9 HX 370
```

## Variant registry and lowering

Registry order and stable metadata are fixed:

| Order | Stable ID | Required features | Priority | Lowering |
| --- | --- | --- | --- | --- |
| 0 | `cpu.reference.f32.v1` | portable scalar f32 | 30 | direct i/j/k float loop |
| 1 | `cpu.tiled.f32.v1` | portable scalar f32 | 20 | 32x32x64 tiled loop with complete tails |
| 2 | `cpu.compiler-vectorized.avx2-fma.f32.v1` | scalar, AVX2, FMA | 10 | x86 AVX2/FMA target function, restricted pointers, vectorized N loop, complete tails |

Debug and unset single-config builds compile the runtime, direct planner test,
and opt-in benchmark at `-O2`; Release keeps `-O3` and other optimized
configurations retain their configured level. An x86 artifact test resolves
the exact mangled vector function, disassembles only that body, and requires
function-local YMM packed-FMA instructions.

No coherent external BLAS development package was installed: OpenBLAS/BLAS
runtime libraries existed, but `openblas`, `blas`, and `cblas` pkg-config
records and CBLAS headers were absent. No optional adapter was therefore added,
and BLAS is not required.

## Legality and deterministic cost

All variants require positive M/N/K, f32 elements and accumulation, row-major
contiguous storage, power-of-two alignment at least `alignof(float)`, a valid
v1 capability record, and portable scalar f32. The compiler-vectorized variant
additionally requires complete x86-64 discovery, AVX2, FMA, and at least 256
usable vector bits. Alignment below 32 bytes remains legal but is cost-penalized.

Let `W = saturating_u64(M*N*K)`. Costs are saturating integer formulas:

```text
reference = 8*W
tiled = 4*W + 4096 + 64*((M mod 32) + (N mod 32) + (K mod 64))
compiler-vectorized = 2*W + 16384
                    + (alignment < 32 ? 24576 : 0)
                    + 128*(N mod 8)
```

Automatic selection minimizes cost, then the lower priority number, then
registry order. Forced requests execute only that legal variant; an illegal
forced request fails with no fallback and leaves output unchanged. These are
initial deterministic heuristics, not measured-time predictions or autotuning.

## Frozen plan decisions

Synthetic capability and explicit-alignment tests freeze these choices:

| M,K,N | Alignment/capability | Reference | Tiled | Vector | Selected |
| --- | --- | ---: | ---: | ---: | --- |
| 2,3,2 | 64, AVX2/FMA | 96 | 4,592 | 16,664 | reference |
| 16,16,16 | 64, AVX2/FMA | 32,768 | 23,552 | 24,576 | tiled |
| 32,32,32 | 64, AVX2/FMA | 262,144 | 137,216 | 81,920 | compiler-vectorized |
| 33,35,37 | 64, AVX2/FMA | 341,880 | 177,660 | 102,494 | compiler-vectorized |
| 24,24,24 | 64, AVX2/FMA | 110,592 | 64,000 | 44,032 | compiler-vectorized |
| 24,24,24 | 4, AVX2/FMA | 110,592 | 64,000 | 68,608 | tiled |
| 32,32,32 | 64, scalar only | legal | legal | illegal | tiled |

Forced correctness separately executes reference at 5x9x7, tiled at
33x65x37, and compiler-vectorized at 19x35x13. Runtime oracle cases cover
1x1x1, 2x3x2, 3x2x4, 16x16x16, 33x35x37, 64x7x19, and an explicitly
64-byte-aligned 24x24x24 case. Saturated maximum-size cost arithmetic ties at
`UINT64_MAX` and deterministically selects compiler-vectorized by priority.

## Correctness and benchmark evidence

The opt-in Release benchmark uses two warmups and nine measured iterations per
forced legal variant. All outputs are compared with the reference, non-finite
values fail, the working set is capped at 256 MiB, and `--guard` applies only a
generous 5 s median/10 s p95 absolute regression bound.

| M,K,N | Automatic selection | Reference median/p95 ms | Tiled median/p95 ms | Vector median/p95 ms |
| --- | --- | ---: | ---: | ---: |
| 4,4,4 | reference | 0.0001 / 0.0005 | 0.0001 / 0.0006 | 0.0001 / 0.0001 |
| 16,16,16 | tiled | 0.0013 / 0.0016 | 0.0007 / 0.0009 | 0.0017 / 0.0018 |
| 64,7,19 | tiled | 0.0027 / 0.0030 | 0.0020 / 0.0023 | 0.0034 / 0.0034 |
| 33,35,37 | compiler-vectorized | 0.0138 / 0.0140 | 0.0097 / 0.0100 | 0.0061 / 0.0062 |
| 128,128,128 | compiler-vectorized | 1.3413 / 1.3983 | 0.1852 / 0.2086 | 0.0880 / 0.0898 |

These are one local Clang 21.1.8 run on the named CPU. They demonstrate
correct execution and plausible crossover behavior only; they are not general
performance, BLAS, or cross-machine claims.

Linux `perf` capture was attempted, but `perf_event_paranoid=4` denied hardware
counters. PerfDigest correctly returned no fabricated metrics. PerfDigest also
compacted the fresh Debug Ninja log: 30 build units, with the hottest eight
covering 75.43% of summed build-step duration; native frontend compilation was
the dominant unit at 6.131 s.

## Validation commands and results

Fresh Ubuntu 24.04 RapidJSON/Release reproduction before the additive object
artifact test:

```sh
cmake -S compiler -B /tmp/matcore-m2-noble-rapidjson.vlsn4d -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DMATCORE_RAPIDJSON_INCLUDE_DIR=/tmp/rapidjson-noble.C9rB3R/root/usr/include
cmake --build /tmp/matcore-m2-noble-rapidjson.vlsn4d --parallel 2
ctest --test-dir /tmp/matcore-m2-noble-rapidjson.vlsn4d \
  --output-on-failure -j1
```

Result: 29 build steps and 13/13 tests passed in 60.48 s, including native,
bootstrap parity, adversarial, driver, integration, installed consumer, IR,
runtime, benchmark-support, and plan CLI tests.

Fresh Debug used the same configure tuple with
`-DCMAKE_BUILD_TYPE=Debug`, built 29 steps, and passed the final 14/14 suite in
63.76 s at
`/tmp/matcore-m2-final-debug.xNGqPr`.

Fresh sanitizer configure added:

```text
-fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
```

At `/tmp/matcore-m2-final-sanitize.dSaW3c`, 29 build steps completed and the
focused native/IR/runtime/planner set passed 7/7. The vector artifact test is
intentionally absent there because instrumented builds make that variant
unavailable. A separately instrumented
generated executable at
`/tmp/matcore-m2-final-sanitize-artifact.Hswo9G/gemm-sanitized` ran and printed:

```text
host-before
MDSLC CPU GEMM PASS
```

The system RapidJSON 1.1 null-pointer arithmetic category remains suppressed
only for the three source files that instantiate that dependency. Matcore
verifier, bridge, planner, runtime, and tests retain ASan/UBSan instrumentation.

Benchmark command:

```sh
for shape in '4 4 4' '16 16 16' '64 7 19' '33 35 37' '128 128 128'; do
  build-m2-integration/bin/matcore_cpu_planner_benchmark --guard $shape
done
```

All variants available on the host passed correctness and guard checks.

## Remaining limitations

- Linux/Ninja/Clang 21.1.8 is the validated native tuple.
- Actual runtime execution remains synchronous host rank-2 row-major f32 GEMM.
- AArch64 capability modeling is compile-time present but has no independent
  AArch64 runtime evidence in this milestone.
- The compiler-vectorized implementation is x86-64 AVX2/FMA only.
- The cost model is static and intentionally small; it does not model caches,
  threading, NUMA, frequency, or library crossover and performs no autotuning.
- No BLAS development package was available, so no BLAS adapter is claimed.
- The generated execution ABI retains its v0 name for compatibility; the
  additive v1 ABI is a read-only plan report, not a second execution ABI.
- Accelerator and fusion work remain separate milestones.

## Independent review and GitHub

The independent reviewer audited through `0ff17cd` and returned **PASS** in
`docs/mdslc/agent-reports/milestone2-independent-review.md`. It found one P1:
Debug/default builds could select the compiler-vectorized stable ID while `-O0`
had emitted scalar packed work. The integrated fix preserves Release `-O3`,
uses `-O2` only where Debug/default otherwise lacks optimization, and freezes
the exact function-local YMM packed-FMA artifact. Its fresh archive build
completed 29/29 steps; selected tests passed 7/7; Release and Debug runtime plus
artifact checks each passed 2/2; emitted v1, adversarial mutations, native
execution, C ABI, no-allocation, and oracle probes passed.

GitHub PR #5 check results are updated here after the final documentation push.
