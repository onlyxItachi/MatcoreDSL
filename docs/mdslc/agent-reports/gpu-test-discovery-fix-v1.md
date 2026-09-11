# GPU test interpreter discovery correction

## Observed counterexample

[Native CI run 34604186150](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34604186150)
at `d440b28a25a5b92b739ce730df4651b271bed9a3` failed Debug and three
MLIR-enabled Release jobs. Their only failed tests were
`generated_gpu.nvvm.source` and `generated_gpu.rocdl.source`, both **Not Run**:
`Unable to find executable:` had an empty executable. These were real missing
test executions, not GPU absence skips or numerical failures.

The error survived the ARM dependency merge and the CPU/GPU composition
`a67276a072e56531c3e4810f3d172f7782d72a7c`. GPU tests used
`${Python3_EXECUTABLE}` before the top-level `find_package(Python3)` ran;
sibling-directory discovery does not establish this directory's variable.

A fresh unseeded configure at `65b6e9eacc145bd514e416bced3513655ebd7db0`
reproduced it locally without building or loading vendor libraries:
configure returned 0, the generated `tests/generated_gpu/CTestTestfile.cmake`
contained `add_test(... "" "-B" ...)`, and running exactly the two source
tests returned 8 with both **Not Run**. The full 335-test local run at the same
source head had actually executed and passed, but its existing build cache
contained `Python3_EXECUTABLE:UNINITIALIZED=/usr/bin/python3`. That result
therefore did not establish fresh unseeded configuration correctness.

## Correction and direct regression

`compiler/tests/generated_gpu/CMakeLists.txt` now discovers its own required
Python interpreter immediately after checking that `mdslc-region` exists and
before registering tests. Missing Python fails at configuration; an empty
test command is not silently registered. This changes no compiler semantics,
runtime, GPU artifact, scheduling policy or execution authority.

Reproduce the qualification using a new empty build directory, the exact
LLVM/Clang/MLIR 21.1.8 paths documented in `AGENTS.md`, experimental regions ON,
both GPU execution options OFF, and no `Python3_EXECUTABLE` override:

```sh
cmake -S compiler -B NEW_BUILD -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_MATCORE_MLIR=ON \
  -DMDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON \
  -DMDSLC_ENABLE_EXPERIMENTAL_GPU_ISSUER=OFF \
  -DMDSLC_ENABLE_EXPERIMENTAL_NVVM=OFF \
  -DMDSLC_ENABLE_EXPERIMENTAL_ROCDL=OFF \
  -DMDSLC_ENABLE_OPENBLAS=OFF \
  -DLLVM_DIR=LLVM_21_1_8_CMAKE_DIR \
  -DClang_DIR=CLANG_21_1_8_CMAKE_DIR \
  -DMLIR_DIR=MLIR_21_1_8_CMAKE_DIR
cmake --build NEW_BUILD --target mdslc-region --parallel 1
ctest --test-dir NEW_BUILD --output-on-failure \
  -R '^generated_gpu\.(nvvm|rocdl)\.source$'
```

Inspect both generated CTest commands before building: their first argument
must be the resolved Python interpreter, not an empty string. The existing
source tests then exercise actual compiled-out candidate refusal through the
freshly built driver. These tests are the direct regression surface; no new
test-count-only framework is introduced. Hardware-executing GPU validation
remains a separate enabled-build obligation.
