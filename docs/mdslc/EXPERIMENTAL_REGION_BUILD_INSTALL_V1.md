# Experimental region build and installation boundary

This checkpoint promotes existing, reviewed compiler/runtime definitions out of
test-directory ownership. It does **not** add a source-driver option, authorize
imported IR, freeze a public/private ABI, or claim installed source-to-executable
compilation. See [frontend admission](EXPERIMENTAL_REGION_FRONTEND_V1.md) for the
source subset and its separate evidence.

## Configuration

`MDSLC_ENABLE_EXPERIMENTAL_REGIONS` defaults to `OFF`. Enabling it requires the
native frontend, Matcore MLIR, the exact LLVM/Clang/MLIR 21.1.8 product tuple,
the authenticated shared `clang-cpp` runtime, and native Linux x86-64 (64-bit
pointers). Cross compilation and the 22.1.8 compatibility experiment fail at
configuration rather than manufacturing product support.

```sh
cmake -S compiler -B build-regions -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-21 -DCMAKE_CXX_COMPILER=clang++-21 \
  -DLLVM_DIR=/path/to/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/path/to/llvm-21/lib/cmake/clang \
  -DMLIR_DIR=/path/to/exact-mlir-21.1.8/lib/cmake/mlir \
  -DMDSLC_ENABLE_MATCORE_MLIR=ON \
  -DMDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON -DBUILD_TESTING=OFF
cmake --build build-regions -j2
cmake --install build-regions --prefix /path/to/package
```

OpenBLAS remains optional. Enabling the feature neither fabricates provider
availability nor changes the existing runtime/provider selection policy.

## Single production ownership

`compiler/lib/regions/CMakeLists.txt` owns these existing target names:

| Target | Role |
| --- | --- |
| `matcore_closed_region_semantics` | Frontend-neutral closed-region semantics |
| `matcore_closed_region_admission` | Authenticated private/public source admission |
| `matcore_closed_host_emitter` | Verified straight-line host orchestration emitter |
| `matcore_cpu_gemm_candidate` / `matcore-cpu-gemm-candidate` | Closed-pipeline strict GEMM issuer |
| `matcore_closed_candidates_production_v1` | Native/generated/provider candidate runtime archive |

The same definitions build when the existing inspection/conformance tests are
enabled with the feature `OFF`; only feature `ON` installs experimental
artifacts. Tests retain their names. Injection and native-only omission controls
are separate test archives, constructed by the same private CMake helper. The
old standalone native adapter controls remain independent falsification targets.

The compiler-private build variables `MDSLC_CLOSED_GEMM_NORMAL_OBJECT` and
`MDSLC_CLOSED_GEMM_ASAN_OBJECT` locate the issued objects under
`build/lib/regions/strict-{normal,asan}.o`. The second exists only with tests.
Each has an explicit custom target in its owning directory, so cross-directory
test consumers cannot lose the generating rule. A production archive contains
the normal leaf exactly once; source-link consumers must not append it again.

An ASan compiler configuration instruments the actual production leaf as well
as its surrounding runtime. Selection considers global **and active
single-configuration** C++ flags; the repository supports Ninja, not a
multi-configuration generator. Test-only always-ASan execution and out-of-bounds
negative controls remain separate. A sanitized archive requires a matching
sanitizer runtime when linked; it is not an unsanitized distribution artifact.

## Installed boundary

Feature `OFF` installs only the established `matcore/mdsl.h` and
`matcore/runtime_c.h` headers. It does not recursively leak the experimental
facade or its detail directory into legacy or Windows packages.

Feature `ON` additionally installs:

| Relative to the selected prefix | Contract |
| --- | --- |
| `include/matcore/region.h` | Experimental ordinary-C++ source declarations |
| `include/matcore/detail/region_storage.h` | Opaque owning-result support declarations |
| `lib/mdslc/experimental-regions/include/closed_host_v1.h` | Compiler-private, revision-coupled host adapter |
| `lib/mdslc/experimental-regions/libmatcore_closed_candidates_production_v1.a` | Compiler-private candidate archive, including one issued leaf |

`include`/`lib` above follow `CMAKE_INSTALL_INCLUDEDIR`/`CMAKE_INSTALL_LIBDIR`.
Root build variables `MDSLC_EXPERIMENTAL_REGION_PRIVATE_LIBDIR` and
`MDSLC_EXPERIMENTAL_REGION_PRIVATE_INCLUDEDIR` expose these relative paths to the
future driver. `MatcoreDSL_EXPERIMENTAL_REGIONS_AVAILABLE` reports that support
was installed; it does not add a new `matcoredsl_add_executable` source mode.

The private archive is deliberately **not** an exported public CMake target.
Installed C++ consumers need the public include directory, the private adapter
include directory when deliberately testing compiler internals, this archive,
and the existing `matcore_runtime` DSO. They do not need LLVM/MLIR headers,
libraries, CMake packages or an installed issuer. The legacy DSO remains the
sole owner of OpenBLAS policy synchronization: no private backend archive or
duplicate provider lock is embedded into the region archive.

Ordinary Clang compilation still does not give undefined source intrinsics an
execution implementation. The separate authenticated source-driver integration
must preserve that authority boundary.

## Mechanical acceptance surfaces

`package.experimental_regions.configuration_rejections` rejects missing native
admission, missing MLIR, the newer compatibility tuple and cross compilation at
the feature gate. `package.experimental_regions.install_contract` uses a fresh
installation prefix, checks feature-OFF absence, checks exported dependency
isolation, counts the actual installed generated leaf, rejects test-injection
or embedded-provider exports, then compiles/runs existing owning-result and
candidate tests using only installed headers/archive/DSO. It is also runnable
directly with `cmake -P` against a `BUILD_TESTING=OFF` build.

This is bounded installed runtime/primitive execution evidence, not an
installed source-language compiler claim. Exact build/test outcomes are recorded
in the implementation handoff; no performance or BLAS-parity claim follows.
