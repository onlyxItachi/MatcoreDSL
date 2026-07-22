# Milestone 4 independent build and package validation

Date: 2026-07-22

## Scope and checkpoint

This lane independently validated the Clang 21 Debug and Release build,
OpenBLAS-required package, installed external consumer, runtime ABI surface,
artifact dependencies, repository hygiene, and the explicit OpenBLAS-disabled
configuration. It did not author production code.

The isolated branch began at integration commit
`67531984a56dad61cdeccf4e92abc8baf65c87d0`. The first Debug run exposed two
integration regressions:

- the Debug runtime DSO exported a weak `std::min<size_t>` instantiation in
  addition to the intended C ABI; and
- the installed-consumer check required `status=selected`, while the new v2
  diagnostic encoded the status numerically.

The integration owner supplied focused fixes as commits `47f4085` and
`93f323c`. They were cherry-picked unchanged as `b253ab9` and `137527e` for
independent reruns. The former also added prepacked-storage alias regressions;
the latter restored textual v2 request/status diagnostics and hid the private
backend archive from the Linux DSO export surface. No unresolved build or
package finding remains at the validated checkpoint `137527e`.

## Toolchain and provider

- Clang/Clang++: Ubuntu 21.1.8 (`x86_64-pc-linux-gnu`)
- CMake: 4.3.2
- Ninja: 1.13.2
- OpenBLAS: 0.3.32, authenticated through pkg-config and the configure-time
  LP64 CBLAS probe
- Build parallelism: Ninja `-j2`; CTest `-j1`

All build trees, installations, consumer trees, JSON output, and logs were
created below `/tmp/matcore-m4-build-validation.OhwPMI`; none is tracked.

## Debug validation

Fresh configuration required native and bootstrap frontends plus coherent
Clang/LLVM 21 and OpenBLAS:

```sh
cmake -S compiler \
  -B /tmp/matcore-m4-build-validation.OhwPMI/debug -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=ON \
  -DMDSLC_REQUIRE_OPENBLAS=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/matcore-m4-build-validation.OhwPMI/debug -- -j2
ctest --test-dir /tmp/matcore-m4-build-validation.OhwPMI/debug \
  --output-on-failure -j1
```

Result after the fixes: **27/27 passed** in 65.71 seconds. This includes the
native frontend, driver pipeline, 63-case integration matrix, relocated
installed consumer, IR, platform record, packed AVX2, workspace/prepacked-B,
OpenBLAS, C ABI compatibility, exact object checks, planner v2, and benchmark
contract tests.

The Debug runtime now has exactly these seven defined dynamic symbols:

```text
matcore_runtime_gemm_f32_execute_prepacked_b_v1
matcore_runtime_gemm_f32_execute_v1
matcore_runtime_gemm_f32_prepack_b_v1
matcore_runtime_gemm_f32_prepacked_b_size_v1
matcore_runtime_gemm_f32_v0
matcore_runtime_gemm_f32_workspace_size_v1
matcore_runtime_plan_gemm_f32_v1
```

## Release, install, and consumer validation

A second, new Release tree and install prefix were configured after the fixes:

```sh
cmake -S compiler \
  -B /tmp/matcore-m4-build-validation.OhwPMI/release-final -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=ON \
  -DMDSLC_REQUIRE_OPENBLAS=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/tmp/matcore-m4-build-validation.OhwPMI/install-final
cmake --build /tmp/matcore-m4-build-validation.OhwPMI/release-final -- -j2
cmake --install /tmp/matcore-m4-build-validation.OhwPMI/release-final
ctest --test-dir /tmp/matcore-m4-build-validation.OhwPMI/release-final \
  --output-on-failure -j1
```

Results:

- fresh Release build: 55 build steps, success;
- full Release suite: **27/27 passed** in 63.99 seconds;
- relocated `consumer.installed`: passed, including its regeneration checks;
- independent fresh `find_package(MatcoreDSL REQUIRED)` consumer:
  configure/build/run **1/1 passed**;
- consumer output: `consumer-before`, `consumer-header=1`, `consumer-pass`;
- a subsequent Release build reported `ninja: no work to do.`

The install contains `mdslc++`, `matcore-extract`, `matcore-plan`,
`matcore-bench`, the versioned runtime DSO and symlinks, both public headers,
and the relocatable MatcoreDSL CMake package files.

## Artifact and dependency inspection

`matcore-bench` and `matcore-plan` are ordinary ELF64 x86-64 PIE executables.
`libmatcore_runtime.so.0.0.0` is an ELF64 x86-64 shared object with SONAME
`libmatcore_runtime.so.0` and exactly the seven C ABI exports listed above in
both Debug and Release.

The OpenBLAS-required benchmark, planner, and runtime resolve
`libopenblas.so.0` from `/usr/lib/x86_64-linux-gnu`. Their remaining direct
dependencies are the normal C/C++ runtime libraries; no Python or nanobind
dynamic dependency is present. `matcore-plan` defines no dynamic symbols.
`matcore-bench`, an executable rather than an ABI library, has five weak
libstdc++ string-template definitions; the runtime ABI remains constrained.

The installed planner reported the authenticated OpenBLAS 0.3.32 provider,
readable `request=force-reference status=selected` metadata, every candidate,
cost, workspace, alignment, and selection reason. The installed benchmark ran
a guarded `33x35x37` automatic-selection case through
`cpu.native-packed.avx2-fma.f32.v1`; correctness passed and its JSON parsed.
That single packaging smoke measurement is not performance evidence.

Literal scans found none of the source worktree, Release build, installation,
or `/home/hamza-usta` paths in any installed file. The installed external
consumer is an ordinary ELF64 PIE and resolves the versioned runtime from the
fresh prefix.

## OpenBLAS-disabled validation

A third fresh Release tree used:

```sh
cmake -S compiler \
  -B /tmp/matcore-m4-build-validation.OhwPMI/no-openblas-final -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=OFF \
  -DMDSLC_REQUIRE_OPENBLAS=OFF \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
```

Focused runtime/workspace/OpenBLAS/planner/benchmark targets built, and the
selected 11 tests passed **11/11**. `ldd` confirmed that the runtime, planner,
and benchmark have no OpenBLAS dependency. Automatic planning explicitly
reported `openblas-linked=false` and rejected the external candidate. A forced
OpenBLAS request exited 1 with
`status=forced-variant-illegal` and the actionable reason
`OpenBLAS CBLAS adapter is not linked`; it did not fall back.

## Repository gates and verdict

```sh
bash tests/check_repository_hygiene.sh
git diff --check
```

Both passed. No generated build, install, benchmark, JSON, binary, or log file
is tracked. For this lane's build/package scope, the Milestone 4 checkpoint is
**accepted with no unresolved high or medium finding**.
