# CPU planner and runtime lane report

Date: 2026-07-21

Branch: `mdslc/cpu-planner-runtime`

Base: `231d2c1ee35fad840e6db72bdc986124132c0afc`

Ownership: `compiler/lib/planner/**`, `compiler/lib/runtime/**`,
`compiler/include/matcore/runtime_c.h`, `compiler/tests/runtime/**`, and this
report. The integration owner explicitly approved the additive public runtime
planning ABI after lane assignment.

## Result

The existing `matcore_runtime_gemm_f32_v0` ABI now retains its validation and
status behavior, then constructs a concrete f32 GEMM problem, discovers a
versioned CPU capability record, deterministically selects one of three fixed
variants, and dispatches it synchronously. It still allocates and copies
nothing and lets no exception cross the C ABI.

An additive `matcore_runtime_plan_gemm_f32_v1` query validates the identical
descriptors and returns the selected plan without executing or changing the
output. Its fixed-layout report contains:

- ABI and planner versions, automatic request, and plan status;
- architecture, detection-complete flag, feature bitmask, and vector width;
- M/N/K and observed minimum input/output alignment;
- all three candidates in fixed registry order;
- candidate stable ID, legality, priority, saturated cost, and reason;
- selected stable ID and selection reason.

All returned strings have process lifetime. The caller initializes only the
report ABI version and size; dirty output fields are rejected without changing
the report. There is no environment switch and no ambient stdout/stderr.

## Capability and variant model

The fixed `CpuCapabilitiesV1` record contains version, architecture,
`detection_complete`, fixed feature bits, and usable vector bits. Discovery
uses compiler CPU builtins on x86_64 and has no LLVM, GPU, subprocess, cache,
or topology dependency. Portable scalar f32 remains available when feature
discovery is incomplete; feature-specific candidates fail closed.

This host reported:

```text
architecture=x86_64
model=AMD Ryzen AI 9 HX 370 w/ Radeon 890M
logical_cpus=24
detection_complete=true
features=[portable-scalar-f32,avx2,fma]
usable_vector_bits=256
```

The registry order and stable IDs are:

1. `cpu.reference.f32.v1` — direct i/j/k loop.
2. `cpu.tiled.f32.v1` — 32x32x64 blocking with complete M/N/K tails.
3. `cpu.compiler-vectorized.avx2-fma.f32.v1` — Clang-vectorized N loop in an
   x86 AVX2/FMA target function, with scalar/vector tails.

Common legality requires positive M/N/K, f32 elements and accumulation,
row-major contiguous layout, and float-aligned storage. Reference and tiled
require portable scalar f32. Compiler-vectorized additionally requires
complete discovery, x86_64, AVX2, FMA, and at least 256 usable vector bits.
Every rejection has a stable actionable reason. A forced illegal internal
request fails without fallback or output modification.

Automatic selection minimizes a saturating integer estimate, then uses fixed
priority and registry order as stable tie-breaks:

```text
reference = 8 * M*N*K
tiled     = 4 * M*N*K + 4096 + 64 * (M%32 + N%32 + K%64)
vector    = 2 * M*N*K + 16384 + (alignment<32 ? 24576 : 0)
            + 128 * (N%8)
```

This is an initial deterministic heuristic, not a claim of optimality.

## Correctness and selection coverage

The runtime test uses independent double-precision oracles and covers small,
medium, rectangular, tail-heavy, and explicitly 64-byte-aligned problems. It
forces every host-legal implementation. Synthetic capabilities cover missing
scalar, incomplete discovery (including spoofed feature bits), non-x86,
narrow vectors, unknown capability versions, saturated cost arithmetic,
illegal forced selection, deterministic repeat planning, output preservation,
and complete human/machine diagnostics.

Expected injected-capability decisions are:

| MxKxN | Alignment/capability | Selection |
|---|---|---|
| 2x3x2 | 64, AVX2/FMA | reference |
| 16x16x16 | 64, AVX2/FMA | tiled |
| 32x32x32 | 64, AVX2/FMA | compiler-vectorized |
| 33x35x37 | 64, AVX2/FMA, tails | compiler-vectorized |
| 24x24x24 | 64, AVX2/FMA | compiler-vectorized |
| 24x24x24 | 4, AVX2/FMA | tiled |
| 32x32x32 | scalar only | tiled; vector rejected |

All forced reference, tiled, and detected AVX2/FMA vector executions matched
their independent oracle. Legacy negative validation order, codes, static
messages, output preservation, and success message `ok` remain covered.

## Observed opt-in benchmark

Fresh Clang 21.1.8 Release build, two warmups and nine samples per forced legal
variant. Times are local observations only; there is no cross-machine or
global performance claim. The default suite has no timing threshold. The
optional `--guard` uses deliberately generous absolute hang/regression bounds.

| MxKxN | Auto plan | Reference median ms | Tiled median ms | Vector median ms |
|---|---|---:|---:|---:|
| 4x4x4 | reference | 0.0001 | 0.0001 | 0.0001 |
| 16x16x16 | tiled | 0.0013 | 0.0007 | 0.0017 |
| 64x7x19 | tiled | 0.0027 | 0.0020 | 0.0033 |
| 24x24x24 | tiled (observed alignment penalty) | 0.0044 | 0.0022 | 0.0087 |
| 33x35x37 | compiler-vectorized | 0.0137 | 0.0096 | 0.0061 |
| 128x128x128 | compiler-vectorized | 0.8710 | 0.1849 | 0.0879 |

Every reported benchmark variant passed checksum and elementwise comparison
against the reference result. Median/p95 values are intentionally reported
without extrapolation.

## Validation evidence

Commands included:

```sh
cmake -S compiler/lib/runtime -B /tmp/mdslc-cpu-planner-release.h31IEK \
  -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Release -DMDSLC_BUILD_RUNTIME_BENCHMARKS=ON
cmake --build /tmp/mdslc-cpu-planner-release.h31IEK -- -j2
ctest --test-dir /tmp/mdslc-cpu-planner-release.h31IEK \
  --output-on-failure -j1

cmake -S compiler/lib/runtime -B /tmp/mdslc-cpu-planner-sanitize \
  -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=address,undefined' \
  -DMDSLC_BUILD_RUNTIME_BENCHMARKS=ON
cmake --build /tmp/mdslc-cpu-planner-sanitize -- -j2
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ctest --test-dir /tmp/mdslc-cpu-planner-sanitize \
  --output-on-failure -j1
```

Results:

- fresh standalone Release runtime: 1/1 passed;
- standalone Debug ASan+UBSan runtime after the sizing-only fix: 1/1 passed;
- sanitizer tail benchmark: all three variants correct, no finding;
- Clang 21 C++ compilation with `-Wall -Wextra -Wpedantic -Werror`: passed;
- public runtime header C11 compilation with the same warning policy: passed;
- fresh full standalone build: 21/21 build steps passed;
- full CTest: seven tests passed; the integration matrix had 62 passes and one
  expected cross-lane assertion failure because its export whitelist predates
  the additive `matcore_runtime_plan_gemm_f32_v1` symbol. The integration owner
  was notified to allow exactly the two versioned C symbols.
- installed-consumer test within that full run: passed;
- runtime shared object: ordinary x86-64 ELF; only
  `matcore_runtime_gemm_f32_v0` and
  `matcore_runtime_plan_gemm_f32_v1` are exported;
- `objdump` confirmed YMM `vfmadd213ps` instructions are confined to the
  capability-gated compiler-vectorized function;
- `ldd` showed only normal C/C++ runtime dependencies, with no Python or
  nanobind dependency.

## Commit handoff

1. `3c955c16d784c23db5124300e96eaeeed5913f5c` — deterministic capability,
   registry, cost model, diagnostics, and kernels.
2. `587d34d0c14c11cf171cba85eda3f45600b0d9d7` — additive plan-report ABI,
   runtime dispatch, adversarial/correctness tests, and opt-in benchmark.
3. `274db6323af26b87283fb8009f56fee8a190e375` — null-safe sizing-only
   diagnostic formatting.
4. `aab3cc25e5664540b88740d6e8a2da6ea5c2fd61` — aligned runtime and exact
   registry metadata coverage.

An unexplained concurrent editor added useful adversarial assertions and
benchmark guard/compiler metadata in this isolated worktree. The integration
owner explicitly handed the remaining aligned test change to this lane. The
changes were audited, exercised in fresh Release and sanitizers, and retained;
no generated build directory was committed.

## Limitations

- Only synchronous host-resident row-major f32 GEMM is executable.
- The vector candidate is x86_64 AVX2/FMA only; AVX-512 is deliberately not
  modeled or selected even though this host advertises it.
- The cost model is deterministic and evidence-informed but intentionally
  simple; it is not autotuning or a maximum-performance guarantee.
- The C report API exposes automatic selection. Forced requests remain an
  internal deterministic test API so production callers cannot bypass
  legality.
- No BLAS adapter was added; BLAS remains optional and outside this lane.
- The full integration export whitelist requires the documented one-line
  cross-lane update before the complete suite can be green.
