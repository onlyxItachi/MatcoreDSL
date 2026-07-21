# Goal 5 install and consumer report

Date: 2026-07-19

Role: Driver/build packaging agent

Branch: `mdslc/install-consumer-v0`
Starting integration commit: `e66c2ae6bb7e04312f3026ab562d77ee9cfd69ce`

## Ownership

This task changed only standalone packaging and its external consumer test:

- `compiler/CMakeLists.txt`
- `compiler/lib/runtime/CMakeLists.txt`
- `compiler/cmake/MatcoreDSLConfig.cmake.in`
- `compiler/cmake/MatcoreDSLCompile.cmake`
- `compiler/tests/consumer/`
- this report

No root CMake, legacy Python/JIT/native-extension source, frontend, codegen, or
runtime implementation was changed.

## Commits

1. `d6d8f2b` `build(mdslc): install relocatable CMake package`
2. `4c2be9f` `test(mdslc): validate installed CMake consumer`

## Installed contract

`cmake --install` now creates the bootstrap prefix expected by the existing
self-locating driver:

```text
bin/mdslc++
bin/matcore-extract
include/matcore/mdsl.h
include/matcore/runtime_c.h
lib/libmatcore_runtime.so -> libmatcore_runtime.so.0
lib/libmatcore_runtime.so.0 -> libmatcore_runtime.so.0.0.0
lib/libmatcore_runtime.so.0.0.0
lib/cmake/MatcoreDSL/MatcoreDSLConfig.cmake
lib/cmake/MatcoreDSL/MatcoreDSLConfigVersion.cmake
lib/cmake/MatcoreDSL/MatcoreDSLTargets.cmake
lib/cmake/MatcoreDSL/MatcoreDSLCompile.cmake
```

The exported targets are `MatcoreDSL::Compiler`, `MatcoreDSL::Extractor`, and
`MatcoreDSL::Runtime`. Their installed locations and include directory are
derived from `_IMPORT_PREFIX`; the installed CMake files contain no repository
or producer-build absolute paths.

`matcoredsl_add_executable` accepts one valid-C++ `.mdsl` source, models its
generated relocatable object as an explicit custom-command output, and creates
a normal CMake executable target linked to `MatcoreDSL::Runtime`. It currently
rejects targets other than `cpu`.

## Fresh validation evidence

The independent Release proof used `/tmp/mdslc-goal5.BHvn1g` and Clang 21.1.8:

```sh
cmake -S compiler -B /tmp/mdslc-goal5.BHvn1g/producer -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DCMAKE_INSTALL_PREFIX=/tmp/mdslc-goal5.BHvn1g/install
cmake --build /tmp/mdslc-goal5.BHvn1g/producer -- -j2
cmake --install /tmp/mdslc-goal5.BHvn1g/producer

cmake -S /tmp/mdslc-goal5.BHvn1g/consumer-src \
  -B /tmp/mdslc-goal5.BHvn1g/consumer-build -G Ninja \
  -DCMAKE_PREFIX_PATH=/tmp/mdslc-goal5.BHvn1g/install \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21
cmake --build /tmp/mdslc-goal5.BHvn1g/consumer-build -- -j2
/tmp/mdslc-goal5.BHvn1g/consumer-build/matcore_consumer
```

The clean producer build completed all 15 Ninja steps. The external build had
exactly two steps:

```text
[1/2] Compiling MDSLC source consumer.mdsl
[2/2] Linking CXX executable matcore_consumer
```

Execution produced:

```text
consumer-before
consumer-pass
```

After adding a harmless comment to the copied `.mdsl`, the rebuild again had
only the same object-generation and final-link steps. The immediately following
build reported `ninja: no work to do.`

The complete fresh standalone suite passed:

```text
frontend.bootstrap                 Passed
integration.validation_matrix      Passed
consumer.installed                 Passed
runtime.cpu.gemm_v0                Passed
100% tests passed, 0 tests failed out of 4
```

The installed-consumer CTest independently checks the complete install tree,
configures and runs a copied external project, makes a real `.mdsl` edit,
requires exactly one regeneration plus one link, requires the next build to be
a no-op, and scans installed CMake files for local absolute-path leakage.

Artifact inspection found:

- generated consumer object: ELF 64-bit x86-64 relocatable;
- consumer: ELF 64-bit x86-64 PIE executable;
- dynamic dependency: `libmatcore_runtime.so.0` resolved from the fresh install
  prefix;
- `rg` found no worktree, repository, or producer-build path in installed CMake
  files (exit status 1, no matches).

## Failure found and corrected

The first rebuild test advanced the copied source timestamp relative to its old
source timestamp, which could still be earlier than the just-created object.
Ninja correctly reported no work. The test now performs a real source edit and
then verifies both the rebuilt object timestamp and exact two-step rebuild.

## Bootstrap limitations

- The installed driver continues to use the configured coherent compiler path
  `/usr/bin/clang++-21`; this package is intentionally Linux/Clang-first rather
  than a redistributable cross-toolchain SDK.
- Driver discovery currently requires the standard `bin`, `include`, and `lib`
  prefix directories. Overriding GNU install subdirectories is not supported.
- The helper exposes the implemented single-source, CPU-only v0 path. Multi-TU
  convenience APIs and GPU targets remain later work.
- Python is used only by the opt-in CTest harness; installed compiler/runtime
  execution and the consumer build path do not invoke or import Python.
