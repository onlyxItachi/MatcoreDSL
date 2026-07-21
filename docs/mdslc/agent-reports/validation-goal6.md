# Goal 6 standalone validation report

## Scope and baseline

- Role: validation and adversarial test owner (agent E).
- Worktree: `/home/hamza-usta/MatcoreDSL-wt-validation-final`.
- Branch: `mdslc/validation-v0-final`.
- Verified integration baseline:
  `e66c2ae6bb7e04312f3026ab562d77ee9cfd69ce`.
- File ownership was limited to `compiler/tests/integration/**` and this new
  report. No compiler, frontend, IR, codegen, driver, runtime, public header,
  CMake, packaging, example, fixture, or legacy production file was edited.

The version-2 manifest has 42 executable cases and 7 future-capability records.
Future records are printed separately and are never counted as passes.

## Executable coverage

The integration runner now checks the implemented Goal 1-4 surface rather
than recording it as pending:

- host-only valid C++ and ordinary host templates;
- direct qualified and namespace-aliased canonical GEMM recognition;
- explicit `target=cpu` and `fallback=error` policy capture;
- non-template free-function and class-method calls;
- two distinct site IDs that remain stable across repeated extraction;
- deterministic JSON IR, deterministic generated sources, and valid IR
  verification;
- rejection of textual lookalikes, policy variables, `std::mdsl`, unqualified
  calls, indirect references, templates, lambdas, macros, header-origin calls,
  constexpr calls, const/temporary outputs, syntactic aliasing, side effects,
  and CUDA policy;
- malformed and version-mismatched serialized IR rejection;
- direct-driver and CPU-pipeline input after `--`;
- direct and full CPU-pipeline metacharacter-path probes without shell
  evaluation;
- missing-input diagnostics, unsupported CUDA/no-fallback behavior, and source
  overwrite protection;
- complete saved Matcore artifact set, ELF relocatable form, stable generated
  symbols, and saved-IR verification;
- ordinary external `clang++` link, runtime SONAME resolution, and correct
  executable behavior;
- three CPU GEMM shapes against an independently ordered double oracle;
- runtime null, ABI, dtype, rank, shape, layout, mutability, residency, target,
  fallback, overlap, alignment, and overflow errors;
- generated-wrapper shape and storage-alias failures with original `.mdsl`
  source locations;
- strict C11 inclusion of `runtime_c.h`;
- runtime ELF/SONAME/dynamic-symbol inspection.

## Release build and tests

Fresh Release configuration and build:

```sh
cmake -S compiler -B /tmp/matcoredsl-goal6-release.UL6cEM -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcoredsl-goal6-release.UL6cEM -- -j2
ctest --test-dir /tmp/matcoredsl-goal6-release.UL6cEM \
  --output-on-failure -j1
```

Result: all 15 Ninja steps passed with Clang 21.1.8. Final CTest result was
3/3 passed in 30.98 seconds:

```text
frontend.bootstrap                 passed
integration.validation_matrix      passed (42/42 executable cases)
runtime.cpu.gemm_v0                passed
```

Direct narrow evidence:

```text
frontend tests: 32 checks passed
runtime CPU GEMM v0: all tests passed
```

## Debug build and blocking finding

Fresh Debug configuration and all 15 Ninja build steps passed:

```sh
cmake -S compiler -B /tmp/matcoredsl-goal6-debug.kmuKhx -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/matcoredsl-goal6-debug.kmuKhx -- -j2
ctest --test-dir /tmp/matcoredsl-goal6-debug.kmuKhx \
  --output-on-failure -j1
```

Debug CTest passed `frontend.bootstrap` and `runtime.cpu.gemm_v0`, but the
integration matrix passed only 41/42 cases. The intentionally strict runtime
ABI artifact check found this extra exported C++ symbol:

```text
0000000000001a60 W _ZNSt14numeric_limitsImE3maxEv
0000000000001120 T matcore_runtime_gemm_f32_v0
```

Release exports only `matcore_runtime_gemm_f32_v0`. The Debug-only weak
`std::numeric_limits<unsigned long>::max()` export leaks a C++ implementation
symbol from a library intended to expose a stable C ABI. The integration owner
classified this as a blocker and will fix production. The validation remains
strict: the intended GEMM entry must be the sole exported implementation
symbol. No production workaround was added in this branch.

## Sanitizer evidence

A dedicated Debug ASan/UBSan runtime build used:

```sh
cmake -S compiler/lib/runtime \
  -B /tmp/matcoredsl-goal6-sanitized-runtime.TZ4KNX -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Debug \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined' \
  '-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address,undefined'
cmake --build /tmp/matcoredsl-goal6-sanitized-runtime.TZ4KNX -- -j2
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ctest --test-dir /tmp/matcoredsl-goal6-sanitized-runtime.TZ4KNX \
  --output-on-failure -j1
```

Result: 1/1 runtime test passed with no sanitizer finding.

The generated host/stub/backend path was independently compiled with
`-fsanitize=address,undefined -fno-omit-frame-pointer` through `mdslc++` and
executed with leak detection and UBSan stack traces. It printed:

```text
host-before
MDSLC CPU GEMM PASS
```

No sanitizer diagnostic was emitted.

## Artifact and execution evidence

The required saved-object pipeline produced exactly:

```text
gemm_v0.backend.cpp
gemm_v0.backend.o
gemm_v0.host.cpp
gemm_v0.host.o
gemm_v0.matcore.json
gemm_v0.o
gemm_v0.sites.h
gemm_v0.stubs.cpp
gemm_v0.stubs.o
```

Inspection of `gemm_v0.o` reported:

```text
ELF 64-bit LSB relocatable, x86-64
Class: ELF64
Type: REL (Relocatable file)
Machine: Advanced Micro Devices X86-64
```

`nm -C` showed defined `main`, one generated C++ call-site function, one
generated `extern "C"` backend entry, and the expected unresolved
`matcore_runtime_gemm_f32_v0` reference. Ordinary `/usr/bin/clang++-21`
linked the object, `ldd` resolved `libmatcore_runtime.so.0` from the selected
Release build, and execution printed the two-line CPU pass output.

## Failures encountered and resolved in the test lane

The first expanded Release runner attempt passed 37/39 cases. Two generated
runtime-error fixtures contained an incorrectly escaped newline in a C++
character literal. That was a validation-template defect, not a production
failure. The template was fixed, both exact cases were rerun, and the complete
Release matrix then passed. A later strengthening added C-header, runtime
artifact, CPU metacharacter, and repeated-site-stability checks, bringing the
final executable count to 42.

## Explicitly unimplemented capabilities

The manifest truthfully records these as `future`, not pass, skip, or xfail:

- installed external consumer at baseline `e66c2ae` (owned by the packaging
  lane and to be validated after integration);
- CUDA/cuBLAS execution;
- AMD/HIP execution;
- f64 execution;
- noncontiguous execution;
- `relu_gemm` or another custom fused backend;
- `gemv` and `gevm`.

Runtime rejection of unsupported dtype, layout, residency, target, and fallback
is implemented and tested; that does not imply positive support.

## Legacy regression boundary

The standalone diff from `351075e` through `e66c2ae` is additive under
`compiler/`, `docs/`, and repository guidance. Root legacy CMake and the
Python/JIT/native-extension source paths were not modified or rebuilt for this
Goal 6 lane. No legacy runtime-regression claim is made from the standalone
suite; final integration should record either a feasible legacy smoke test or
the exact dependency blocker.

## Verdict at this handoff

The Release CPU architecture slice is validated end to end. The overall Goal 6
gate remains blocked solely by the Debug runtime ABI export leak described
above. The test branch is intentionally ready to prove the lead's production
fix rather than relaxing the ABI contract.
