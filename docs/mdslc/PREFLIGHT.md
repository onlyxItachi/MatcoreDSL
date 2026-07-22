# MDSLC preflight

Audit date: 2026-07-19

> Historical audit snapshot. The commit IDs, dirty-checkout observations, and
> remaining constraints below describe the pre-history-sanitation bootstrap
> environment. They are preserved as provenance and are not the current
> repository status. Current milestone state belongs in `STATUS.md` and the
> milestone-specific preflight/review reports.

## Repository state

The original repository audit observed:

- remote: `git@github.com:onlyxItachi/MatcoreDSL.git`;
- original branch/SHA: `feature/device-resident-tensors` at
  `351075e4d8af1880330b7c0474d701ca76776dfa`;
- no submodules;
- the original checkout had 50 pre-existing tracked deletions, later also an
  untracked top-level `CMakeFiles/`; neither was modified by MDSLC work.

Bootstrap v0 completed on `mdslc/bootstrap-v0`. The native frontend milestone
started from its clean head
`3e3fa5b2d1990e1c37870f8b2096fbda6128716b` and uses integration branch
`mdslc/native-libtooling-v1`. The independently reviewed native implementation
evidence head is `f71f1800a1ba70f2b363ff68ecd6632c7ae8fad1`.

All implementation is additive under `compiler/`, `docs/`, and repository
guidance. Legacy Python/JIT production sources were not migrated or rewritten.

## Host resources

- Ubuntu 26.04 LTS, kernel `7.0.0-27-generic`, x86-64, 64-bit.
- AMD Ryzen AI 9 HX 370, 12 cores/24 threads, one NUMA node.
- 14 GiB RAM and 32 GiB swap at audit time.
- CMake 4.3.2, Ninja 1.13.2, ccache 4.12.3, binutils 2.46.
- Python 3.14.3 was present for tests only; no Python is used by the installed
  compiler/runtime execution path.

Memory pressure remains a practical constraint. Use Ninja `-j2` for builds and
CTest `-j1`; use `-j1` for unusually large links. Do not build LLVM from source
for this project.

## Coherent Clang/LLVM 21 selection

The exact selected tuple is:

- `/usr/bin/clang-21`, `/usr/bin/clang++-21`: 21.1.8;
- `/usr/bin/llvm-config-21`: 21.1.8;
- headers: `/usr/lib/llvm-21/include`;
- libraries: `/usr/lib/llvm-21/lib`, including
  `libclang-cpp.so.21.1` and Clang component archives;
- LLVM CMake package: `/usr/lib/llvm-21/lib/cmake/llvm/LLVMConfig.cmake`;
- Clang CMake package: `/usr/lib/llvm-21/lib/cmake/clang/ClangConfig.cmake`;
- Ubuntu package revision: `1:21.1.8-6ubuntu1`.

The user explicitly approved:

```sh
sudo apt-get install --no-install-recommends \
  libclang-21-dev \
  libclang-cpp21-dev
```

The preceding APT simulation estimated a 28.81 MiB download and approximately
285.53 MiB installed size. At execution, APT reported both packages already at
the newest 21.1.8 revision and made **zero package changes**.

The post-command audit found the matching required surface:

- `clang/Tooling/Tooling.h`;
- `clang/Tooling/CommonOptionsParser.h`;
- `clang/ASTMatchers/ASTMatchFinder.h`;
- `clang/Rewrite/Core/Rewriter.h`;
- `clang/Frontend/FrontendActions.h`;
- `clang/Frontend/CompilerInstance.h`;
- `clang/Lex/Lexer.h`;
- `clang/AST/Attr.h`;
- `clang-cpp`, Tooling, ASTMatchers, Rewrite, AST, Lex, Frontend, and LLVM
  libraries/imported targets.

A native link probe compiled, linked, and ran; `ldd` resolved
`libclang-cpp.so.21.1` and `libLLVM.so.21.1`. The executable, headers,
libraries, and CMake package versions are therefore coherent. MLIR is not a
native frontend dependency.

Do not use PATH defaults for this build: PATH favors LLVM 22. Do not mix the
selected Clang 21 executable with Clang 22 headers or the legacy MLIR packages.

## Reproducible standalone configure

```sh
cmake -S compiler -B build-mdslc -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-mdslc -- -j2
ctest --test-dir build-mdslc --output-on-failure -j1
```

CMake requires exact LLVM/Clang 21.1.8 packages and imported `clang-cpp` and
`LLVM` targets when native support is enabled. Missing native dependencies are
a configuration/default-invocation error, never a bootstrap fallback.

## Accelerator inventory

- NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, 8,188 MiB,
  driver 610.43.02, CUDA 13.3, nvcc 13.3.73: detected only.
- AMD `gfx1150`, 16 compute units, wave size 32, HIP/ROCm 7.1: detected only.
- AMD/Xilinx `aie2p` NPU/DSP agent: detected only.
- OpenCL reported NVIDIA plus an unavailable Xilinx device.

No CUDA, cuBLAS, HIP, Metal, NPU, or other accelerator backend was compiled or
executed in native frontend v1.

## Constraints observed at this audit

There is no Clang/LibTooling dependency blocker for the standalone native CPU
slice. Two unrelated constraints remain:

1. The original checkout's user-owned deletions/untracked output must not be
   overwritten; isolated worktrees remain required.
2. Fresh root legacy CMake configuration still fails because the project asks
   for MLIR 18.1.3 while the available MLIR configuration is 22.1.2. This is a
   legacy-build issue, not a standalone frontend dependency, and was not
   repaired in this milestone.
