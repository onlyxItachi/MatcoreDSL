# Milestone 2 final independent review

Date: 2026-07-21

## Verdict

**PASS for the declared standalone native CPU-only Milestone 2 scope.**

I reviewed the complete range from the immutable native-frontend checkpoint
`c025df534d11d1bc08285a174f2cd357aecadb0e` through the requested final head
`bf79a6dc3aac1ae452d205651eeb78e3340fba46`. The range contains 43 focused
commits and changes 45 files. The final worktree was clean and
`git diff --check` passed.

All confirmed high- and medium-severity findings are resolved. No unresolved
high or medium finding remains. I recommend integrating the report commit but
do not recommend expanding the milestone claims beyond synchronous,
host-resident, row-major f32 GEMM on the validated Linux/Clang 21 CPU scope.

## Review scope

The review attempted to reject the change across these boundaries:

- typed Matcore IR v1, exact schema dispatch, deterministic JSON, UTF-8
  handling, operation-scoped dimension symbols, and the loss-checked v0 bridge;
- CPU capability discovery, malformed capability records, fixed variant
  registry, legality, saturated costs, stable tie-breaking, and forced-request
  behavior;
- reference, tiled, and compiler-vectorized execution, including tails,
  alignment, non-finite results, allocation failures, and object-code shape;
- the additive C plan-report ABI, report initialization/lifetime rules,
  descriptor validation, output preservation, and no-allocation/no-copy
  behavior;
- native driver v1 artifacts, explicit bootstrap comparison mode, installed
  package relocation, external consumer regeneration, and path leakage;
- Release, Debug, ASan/UBSan, GitHub Clang 21, and relevant legacy regression
  surfaces.

The final architecture retains the proven frontend and code generator:

```text
authenticated native/bootstrap capture
  -> verified IR v0 compatibility DTO
  -> typed and verified Matcore IR v1
  -> loss-checked v1-to-v0 rewrite/codegen projection
  -> generated C ABI descriptors
  -> runtime capability discovery
  -> deterministic legal CPU plan
  -> selected reference/tiled/compiler-vectorized implementation
```

Detected capabilities and selected plans remain downstream of the
target-independent IR. No second schema, verifier, source rewriter, or code
generator was introduced.

## High-severity finding disposition

### H1: a Debug/default vector-named plan could contain scalar code — fixed

Before the fix, an AVX2/FMA host could select
`cpu.compiler-vectorized.avx2-fma.f32.v1` in a Debug/default build even though
the emitted function contained scalar `vfmadd213ss` and no packed YMM loop.
The result was numerically correct, but the selected-lowering claim and cost
model input were false.

Resolution:

- `97160b6` made the runtime implementation optimizable in Debug/default;
- `71b3a82` resolves the exact raw function symbol, disassembles only that
  function, and requires function-local YMM packed FMA;
- `0ff17cd` preserves configured Release/RelWithDebInfo/MinSizeRel
  optimization instead of overriding it;
- `55b3a18` removes AVX2/FMA/vector-width claims from sanitizer-instrumented
  capability discovery when instrumentation prevents the packed lowering;
- `6cb2f3a` applies one configuration-preserving optimization rule to runtime,
  benchmark, and direct runtime tests.

Fresh Release and Debug each pass the scoped object test. Direct Debug
disassembly contains `vbroadcastss ymm`, `vmovups ymm`, and packed
`vfmadd213ps` inside the exact compiler-vectorized function. The sanitizer plan
instead reports portable scalar f32, zero vector bits, rejects the vector
candidate, and selects tiled execution.

Disposition: **resolved and regression-tested**.

## Medium-severity finding disposition

### M1: installed-consumer depfile guard rejected legitimate `/tmp` roots — fixed

The old blanket `/tmp/mdslc-` check caused the installed-consumer test itself
to fail when a legitimate source/build prefix began with that text. `0914d0d`
now detects only the actual six-character MDSLC temporary-workspace component.
The original reproduction and the full installed-consumer test pass.

### M2: dynamic dimension symbol scope was ambiguous across operations — fixed

Two independent GEMMs reused literal `m/k/n` symbols without a documented
scope. `5f91aa9` defines dynamic symbols as operation-scoped and adds a
multi-operation verifier regression. Reusing `m/k/n` in separate operations no
longer implies cross-operation equality.

### M3: malformed UTF-8 could pass both IR parsers — fixed

A v1 document with a raw `0xff` byte in an expression was initially accepted.
`5f91aa9` enables RapidJSON encoding validation in both v0 and v1 parsers and
adds raw-byte regressions. The exact adversarial file now exits 1 with
`Invalid encoding in string`.

### M4: malformed or incoherent capability records could plan successfully — fixed

An injected architecture value `255`, unknown feature bit, and vector width
`65535` initially selected a plan and produced incomplete diagnostics.
`ac6be6d` and `66ee31f` validate version, enum domain, known feature mask,
architecture/feature coherence, discovery state, and supported vector widths.
The reproduction now returns `invalid-capabilities`, selects nothing, and
prints the invalid architecture and unknown bits.

### M5: benchmark failure and correctness guards were incomplete — fixed

The benchmark could terminate through uncaught `std::bad_alloc`, and its old
comparison could treat NaN as correct. `66ee31f` adds an explicit 256 MiB
working-set cap, allocation failure handling, and finite-value comparison.
Under a 64 MiB virtual-memory limit, the 5000x5000x1 reproduction now exits 2
with an actionable allocation diagnostic; 10000x10000x1 is rejected by the
explicit cap. NaN and infinity regressions pass.

### M6: sanitizer builds could misadvertise an unavailable vector lowering — fixed

ASan/UBSan prevented packed vectorization while capability discovery still
reported AVX2/FMA. `55b3a18` makes the build capability explicit and
conservative. The final sanitizer plan reports
`features=[portable-scalar-f32]`, `vector_bits=0`, selects tiled, and explains
that AVX2/FMA are required for the vector candidate.

### M7: the benchmark's forced `-O2` overrode Release `-O3` — fixed

The initial Debug hardening applied `-O2` unconditionally to the benchmark,
making recorded Release evidence differ from the final binary. `6cb2f3a`
limits the extra flag to Debug or an unset single-config build type. Fresh
Release compile commands contain `-O3` with no trailing `-O2`; the five-shape
benchmark was rerun under the final configuration.

### M8: CI/package compatibility gaps — fixed

The Ubuntu 24.04 RapidJSON header and a 32-bit frontend adversarial fixture
exposed missing hosted-build compatibility. `1608740` adds a configure-probed,
source-local RapidJSON compatibility mode, and `f345a12` installs the required
multilib target support in the Clang 21 workflow. PR #5's final production-code
head passed both the standalone Clang 21 workflow and the legacy
`build-and-test` workflow. Pull-request triggering provides the active planner
branch gate without duplicate push runs.

### M9: authoritative documentation described the pre-v1/reference-only state — fixed

`bf79a6d` updates `AGENTS.md`, `STATUS.md`, and final evidence to require the
v0 compatibility DTO -> typed v1 -> checked projection boundary, capability
planning, selected execution, final test counts, sanitizer semantics, and the
correct distinction between an independent runtime oracle and benchmark
comparison with the forced reference implementation. It also corrects the
projection claim: self-consistent provenance/ranges are structurally verified
and preserved, not authenticated against an external original by the bridge.

Disposition for M1-M9: **all resolved and revalidated**.

## Final validation evidence

### Release and Debug

An independent fresh Release tree was configured with Clang/LLVM 21.1.8,
native and bootstrap frontends enabled, benchmarks enabled, Ninja, and `-j2`:

```sh
cmake -S compiler -B /tmp/mdslc-m2-review-6cb-release -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_BUILD_RUNTIME_BENCHMARKS=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/mdslc-m2-review-6cb-release -- -j2
ctest --test-dir /tmp/mdslc-m2-review-6cb-release \
  --output-on-failure -j1
```

Result at the final production tree, followed by the documentation-only final
head: 31/31 build steps, then a no-op rebuild, and **14/14 tests passed**. The
exact `bf79a6d` rerun completed in 61.23 s.

The exact final Debug rerun likewise passed **14/14** in 61.23 s, including
native/parity/adversarial frontend, driver pipeline, integration matrix,
relocated installed consumer, IR, runtime, scoped vector object, and planner
CLI tests.

### Sanitizers

The Clang 21 Debug build used ASan+UBSan with leak detection and retained the
source-local RapidJSON pointer-overflow exception only where RapidJSON 1.1 is
instantiated. The supported focused set passed **9/9** in 8.61 s:

- native focused, primary, and adversarial suites;
- IR v1 core;
- benchmark support;
- runtime GEMM/planner execution;
- three planner CLI cases.

The sanitizer plan selected tiled execution and did not advertise vector
capabilities. A separately generated instrumented GEMM executable also printed
`MDSLC CPU GEMM PASS` with no sanitizer report, as recorded in the authoritative
milestone evidence.

### IR, planner, benchmark, and runtime

- `matcore_ir_v1_tests` reports **180 checks passed**.
- Invalid UTF-8, unknown versions, exact-member violations, shape-symbol
  mismatch, lossy projection, duplicate site IDs, malformed ranges, effects,
  aliases, policies, and unsupported semantics fail closed.
- The fixed registry and injected capability tests cover reference, tiled,
  vector, missing features, incomplete discovery, non-x86 records, bad widths,
  bad alignments, overflow saturation, forced illegal requests, and stable
  repeated selection.
- Five benchmark shapes (4x4x4, 16x16x16, 64x7x19, 33x35x37, and 128x128x128)
  ran every host-legal variant; each matched the forced reference and passed
  the generous absolute guard. Runtime tests separately use a double-precision
  independent oracle.
- Runtime dynamic exports are exactly
  `matcore_runtime_gemm_f32_v0` and
  `matcore_runtime_plan_gemm_f32_v1`.
- The runtime imports no `malloc`, `free`, C++ allocation, `memcpy`, or
  `memmove` symbol. Planning does not modify output; execution remains
  synchronous with no hidden allocation or migration.
- The installed `runtime_c.h` compiles as strict C11 with
  `-Wall -Wextra -Wpedantic -Werror`.

### Artifact, install, consumer, and hosted checks

The final driver emitted and verified Matcore IR v1 with producer
`clang-libtooling-v1`, generated host/sites/stubs/backend sources, and produced
an ordinary x86-64 ELF relocatable object. Ordinary `clang++-21` linked it
against the versioned runtime; execution printed:

```text
host-before
MDSLC CPU GEMM PASS
```

`file`, `readelf`, `nm -C`, and `ldd` confirmed the expected REL object,
generated site/backend symbols, unresolved C runtime boundary before final
link, and normal native dependencies. Installed binaries, headers, runtime,
and CMake files contain no source/build/home absolute path and no Python or
nanobind reference. The external `find_package(MatcoreDSL REQUIRED)` consumer,
dependency regeneration, and no-op rebuild are included in the passing full
suite.

At hosted head `4d98f57` (which contains final production commit `6cb2f3a`),
GitHub's `Clang 21.1.8 standalone Release` and legacy `build-and-test` jobs both
passed. `bf79a6d` changes authoritative documentation only; its exact local
Release, Debug, sanitizer, artifact, package, and consumer state was rechecked
before this report.

## Residual limitations

- Executable support remains synchronous host-resident rank-2 row-major f32
  GEMM. The typed vocabulary is broader than the currently legal executable
  subset.
- Native runtime and packed-object evidence are x86-64 AVX2/FMA. AArch64 is
  modeled and synthetically checked but lacks native runtime validation.
- The cost model is deterministic and deliberately small; it does not model
  threading, NUMA, cache topology, frequency, or global optimum and performs no
  autotuning.
- Sanitizer instrumentation deliberately makes the compiler-vectorized variant
  unavailable rather than mislabeling scalarized code.
- Direct extractor output remains v0 by default for compatibility; `mdslc++`
  explicitly requests v1. The existing execution ABI retains its v0 suffix.
- No BLAS, CUDA, HIP, Metal, NPU, MLIR lowering, GEMV, GEVM, fusion, or
  heterogeneous placement is claimed.

These are declared scope limits, not unresolved high or medium defects.

## Final recommendation

Accept the 43-commit Milestone 2 range and this report as the CPU-only
architecture checkpoint. Preserve the focused history for review. Do not infer
accelerator support, production-wide optimality, or operation coverage beyond
the validated native CPU GEMM slice.
