# Native driver and package agent report

## Ownership

This lane changed only the assigned driver/package surface:

- `compiler/tools/mdslc/main.cpp`
- `compiler/cmake/mdslc_config.h.in`
- `compiler/cmake/MatcoreDSLCompile.cmake`
- `compiler/tests/consumer/`
- `compiler/tests/driver_native/`

It did not edit the root standalone CMake project, frontend implementation,
extractor main, IR, runtime, or general validation suites.

## Interface frozen for integration

- `mdslc++` selects `native` by default for the CPU extraction pipeline.
- `--frontend=native` and `--frontend=ast-json-bootstrap` are the only accepted
  explicit values.
- The driver always passes exactly one corresponding `--frontend=...` argument
  to the single `matcore-extract` executable. It propagates failure and never
  retries through the compatibility frontend.
- `--frontend` is meaningful only with `--matcore-target=cpu`; a direct
  host-only compilation remains the existing ordinary Clang path.
- Tool/header/runtime discovery is self-relative and coherent. It supports the
  build layout and configured relative GNU install directories without an
  embedded install prefix.
- The trusted-prefix override is compiled out by default. It exists only when
  deliberately configured with `-DMDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE=ON`.
- The driver snapshots source device/inode, timestamps, size, and exact bytes.
  It rechecks after extraction, dependency scanning, each generated-source
  compile, and linking, rejecting source changes before publishing an object.
- Child processes use `fork` plus `execv` with argv vectors; shell-looking
  compiler arguments are never evaluated. Runtime rpath is forwarded as
  separate linker argv values, preserving spaces and commas.

## Commits

1. `00b4be9` `feat(driver): select native frontend without fallback`
2. `3378d80` `build(mdslc): package native frontend selection`
3. `86f6ff1` `fix(driver): verify source after every pipeline phase`
4. `6161def` `fix(package): preserve punctuation in relocated paths`
5. `c1038c5` `test(driver): execute relocated runtime link`

## Validation evidence

The following passed in this worktree with Clang 21.1.8:

```text
cmake -S compiler -B build-native-driver-v1 -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 -DBUILD_TESTING=OFF
cmake --build build-native-driver-v1 --target mdslc_driver -- -j2
/usr/bin/clang++-21 -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -Ibuild-native-driver-v1/generated -fsyntax-only \
  compiler/tools/mdslc/main.cpp
python3 compiler/tests/driver_native/run_driver_tests.py \
  --driver build-native-driver-v1/bin/mdslc++
```

Focused result:

```text
native driver: default/explicit/bootstrap/no-fallback/argv/relocation/runtime-link/source-snapshot PASS
```

The same focused test passed with the test-only override enabled, and with
configured directories `libexec/matcore`, `sdk/include`, and `lib64`. The
host-only proof also remained valid and printed `5`.

The package was installed, moved to a different prefix, and successfully found
by an external consumer `find_package` configure. The installed production
driver rejected `--tool-prefix-for-testing`, and installed CMake files contained
no source/build prefix leak.

## Integration dependency

This lane began from the bootstrap extractor, which does not yet accept the new
frontend flag. Therefore the real native consumer build/run was intentionally
not claimed here. It must be run after the native frontend commit is integrated.
The required extractor contract was coordinated with the frontend agent: one
binary, the two frontend values above, native default, and no internal fallback.
