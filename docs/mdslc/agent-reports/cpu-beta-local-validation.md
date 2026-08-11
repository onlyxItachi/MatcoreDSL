# CPU beta local Linux validation

Date: 2026-08-11

Owner: local Linux CPU-beta validation lane

Candidate: `69d099ed84871a6d2e448ea3b9105a497047ec28`

Branch: `mdslc/semantic-compiler-foundation-v1`

## Verdict

**Passed for the declared local Linux validation scope.** The candidate was
clean before validation and source provenance embedded by the benchmark
matched the exact candidate commit. No production source was edited by this
lane. Raw build, test, install, and benchmark logs remain outside the
repository under:

```text
/home/hamza-usta/.cache/mdslc-beta-validation-69d099e/
```

This report does not claim hosted CI, Windows execution, native/OpenBLAS
parity, general performance calibration, or a public API/ABI freeze.

## Authenticated environment

- CPU: AMD Ryzen AI 9 HX 370 with Radeon 890M
- Topology: 1 socket, 12 physical cores, 24 logical CPUs, SMT enabled,
  1 physical NUMA node
- Compiler: Ubuntu Clang/Clang++ 21.1.8 (`1:21.1.8-6ubuntu1`)
- LLVM: 21.1.8
- MLIR: isolated Ubuntu MLIR 21.1.8 package prefix at
  `/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`
- BLAS: OpenBLAS 0.3.32 pthread development provider discovered coherently
  through `pkg-config`; the linked runtime resolved
  `/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0`
- Generator: Ninja; every build used `nice -n 10` and at most two jobs;
  every CTest invocation used one job

The MLIR-enabled artifacts did not dynamically link the aggregate shared
`libMLIR`. OpenBLAS-disabled runtimes had no OpenBLAS dynamic dependency.

## Full configuration matrix

| ID | Build | MLIR | Configured default | OpenBLAS | Configure | Compile | Build | Tests | CTest |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| R1 | Release | ON | `matcore-mlir` | required | 1.63 s | 19.08 s | 128/128 | 63/63 | 180.66 s |
| R2 | Release | ON | `matcore-mlir` | OFF | 1.72 s | 20.08 s | 128/128 | 63/63 | 185.22 s |
| R3 | Release | OFF | `capture-v0` | required | 1.49 s | 6.66 s | 106/106 | 58/58 | 157.19 s |
| R4 | Release | OFF | `capture-v0` | OFF | 1.27 s | 5.98 s | 106/106 | 58/58 | 139.59 s |
| C1 | Release | ON | `capture-v0` | required | 1.69 s | 17.61 s | 128/128 | 63/63 | 180.88 s |
| D1 | Debug | ON | `matcore-mlir` | required | 1.71 s | 38.72 s | 128/128 | 63/63 | 286.38 s |

The MLIR-disabled cells included the fail-closed
`integration.semantic_mlir_unavailable` gate. MLIR-enabled cells included the
semantic dialect/verifier, map/domain, recovered-GEMM analysis, CPU runtime
dispatch lowering, CLI, real semantic CPU pipeline, installed consumer,
strict C17 ABI, and source-inaccessible installed-package tests.

The common configuration shape was:

```sh
nice -n 10 cmake -S compiler -B <external-build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=<Release-or-Debug> \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=<ON-or-OFF> \
  -DMDSLC_REQUIRE_OPENBLAS=<ON-or-OFF> \
  -DMDSLC_ENABLE_MATCORE_MLIR=<ON-or-OFF> \
  -DMDSLC_DEFAULT_SEMANTIC_PIPELINE=<matcore-mlir-or-capture-v0> \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DMLIR_DIR=<isolated-MLIR-21-prefix>/lib/cmake/mlir
nice -n 10 cmake --build <external-build-dir> --parallel 2
nice -n 10 ctest --test-dir <external-build-dir> \
  --output-on-failure -j1
```

`MLIR_DIR` was omitted when MLIR was disabled.

## Sanitizers

ASan+UBSan used Debug, `-O1 -g`,
`-fsanitize=address,undefined`, frame pointers, MLIR enabled, the semantic
default, and OpenBLAS disabled. The exact 20 in-process tests passed in 4.68 s:

```text
ir.v1.core
mlir.semantic.core
mlir.semantic.map-domain
mlir.cpu.runtime_dispatch_lowering_v1
mlir.semantic.recovered-gemm
platform.fp_environment.v1
runtime.cpu.reference_types
runtime.c_abi.typed_reference_v1
runtime.cpu.packed_avx2
runtime.cpu.packed_avx512
runtime.cpu.execution_context.v1
runtime.cpu.parallel_packed.v1
runtime.cpu.planner_v3_resources
runtime.cpu.workspace_v1
runtime.cpu.openblas_adapter
runtime.c_abi.compatibility_v1
runtime.c_abi.fp_environment_v1
runtime.cpu.variant_conformance.v1
runtime.c_abi.public_context_v1
runtime.cpu.gemm_v0
```

The environment used:

```text
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
```

The fixed allowlist command was:

```sh
sanitizer_tests='^(ir\.v1\.core|mlir\.semantic\.(core|map-domain|recovered-gemm)|mlir\.cpu\.runtime_dispatch_lowering_v1|platform\.fp_environment\.v1|runtime\.cpu\.(reference_types|packed_avx2|packed_avx512|execution_context\.v1|parallel_packed\.v1|planner_v3_resources|workspace_v1|openblas_adapter|variant_conformance\.v1|gemm_v0)|runtime\.c_abi\.(typed_reference_v1|compatibility_v1|fp_environment_v1|public_context_v1))$'
nice -n 10 ctest --test-dir <asan-ubsan-build-dir> \
  --output-on-failure -j1 -R "${sanitizer_tests}"
```

ASan+UBSan configured in 1.96 s and built 128/128 steps in 84.09 s.

TSan used Debug, `-O1 -g`, `-fsanitize=thread`, frame pointers, and disabled
MLIR, OpenBLAS, and the native frontend to isolate the shared runtime. Its
exact four-test scope passed in 7.87 s:

```text
runtime.cpu.execution_context.v1
runtime.cpu.parallel_packed.v1
runtime.cpu.planner_v3_resources
runtime.c_abi.public_context_v1
```

`TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1` was active.

The fixed TSan command was:

```sh
tsan_tests='^runtime\.(cpu\.(execution_context\.v1|parallel_packed\.v1|planner_v3_resources)|c_abi\.public_context_v1)$'
nice -n 10 ctest --test-dir <tsan-build-dir> \
  --output-on-failure -j1 -R "${tsan_tests}"
```

TSan configured in 1.35 s and built its four requested targets in 7.33 s.

## Install, package, ABI, and artifacts

A fresh R1 install prefix contained 15 files or symlinks:

```sh
nice -n 10 cmake --install <R1-build-dir> --prefix <external-install-prefix>
```

- `bin/mdslc++`
- `bin/matcore-extract`
- `bin/matcore-plan`
- `bin/matcore-bench`
- `bin/matcore-mlir`
- `lib/libmatcore_runtime.so` and its versioned links
- `include/matcore/mdsl.h`
- `include/matcore/runtime_c.h`
- relocatable `lib/cmake/MatcoreDSL/*` package files

The installed package reported:

```cmake
set(MatcoreDSL_MATCORE_MLIR_AVAILABLE "ON")
set(MatcoreDSL_DEFAULT_SEMANTIC_PIPELINE "matcore-mlir")
```

ELF headers and dynamic symbols were inspected. The unfiltered R1 suite
passed `consumer.installed`, `package.installed_c17_abi`,
`package.installed_source_inaccessible`, and the source-inaccessible safety
test. Therefore the semantic installed consumer was executed after the test
removed access to its staged source and build trees; a build-tree-only success
is not being inferred.

## Legacy and repository gates

- Legacy Python frontend contract: 14/14 passed in 0.18 s using
  `python3 -m pytest -q -p no:cacheprovider tests/test_frontend_contract.py`.
- `git diff --check`: passed.
- `tests/check_repository_hygiene.sh`: passed.
- `git fsck --full --strict`: exit 0. It reported only existing dangling
  trees/blobs, not object corruption.
- Non-ignored untracked files: 0.
- Final worktree before this report: clean at the exact candidate SHA.

The repository commands were:

```sh
git diff --check
tests/check_repository_hygiene.sh
git fsck --full --strict
git status --short --branch
git ls-files --others --exclude-standard
```

## Performance sanity boundary

One guarded functional timing sanity run used a build-idle host, inherited
affinity restricted to CPU 0, one requested thread, hot cache, included
packing, reused workspace, two warmups, five samples, and complete-call timing:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
nice -n 10 taskset -c 0 matcore-bench \
  --m 256 --n 256 --k 256 --threads 1 --variant auto \
  --warmup 2 --iterations 5 --hot-cache --include-packing \
  --reuse-workspace --alignment 64 --timer-floor-us 1000 --guard \
  --json-out <external-log-path>
```

The planner selected `cpu.external.openblas.f32.v1`; correctness passed,
timing was valid, median was 0.223964 ms, and the diagnostic throughput was
149.82 GFLOP/s. This single small run is only an execution/timer sanity proof.
It is not a native parity result, planner-regret calibration, frequency-stable
benchmark campaign, or general performance claim.

## Adversarial discovery resolved before the final run

The first R1 attempt at the preceding candidate
`6b828c9a733bba4d89d6adb7dec6a7d0b284d290` passed 62/63 and deterministically
failed `runtime.cpu.planner_v3_resources`. Investigation showed a stale test
oracle: numerical-conformance validation had expanded execution to six
unconditional serial submissions and four submissions per available parallel
ISA, while the test retained its older submission-count formula. The dedicated
planner-resource lane corrected the test-only oracle and an independent lane
reviewed it. All results above were then rebuilt from scratch at `69d099e`;
no binary from the failed candidate was reused.

During R3 correctness CTest, an unrelated llama/XDNA Ninja build began only
after the MDSLC R3 compilation had completed. R3 remained correctness-only;
no performance evidence was collected or claimed from that overlap. R4 and
the timing sanity check started only after the unrelated processes exited and
the host was build-idle.

## Supported and unsupported conclusions

Supported locally:

- the exact Linux CPU-beta candidate passes the declared Release toggle
  matrix, compatibility default, full Debug, focused ASan+UBSan, and focused
  TSan scopes;
- both semantic-default and capture-default packages behave correctly when
  MLIR is compiled in;
- MLIR and OpenBLAS absence are explicit and fail closed where required;
- installed semantic tooling and a real installed consumer do not require the
  source/build trees;
- the guarded benchmark executes a complete verified GEMM and emits valid
  provenance and timing metadata.

Not supported by this lane:

- hosted GitHub Actions or Windows results;
- native BLAS parity or a completed Milestone 7 envelope;
- multi-node NUMA validation;
- AMX runtime support;
- GPU/NPU execution;
- a frozen public API, ABI, or backend contract;
- a beta tag or published release.
