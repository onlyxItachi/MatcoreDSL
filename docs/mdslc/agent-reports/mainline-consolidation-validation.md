# MDSLC mainline consolidation validation

Date: 2026-07-21

## Repository and history

Verified remote tips before integration:

| Ref | Commit |
| --- | --- |
| `origin/main` | `7d2f8475ce3148658743ebc6a58be808a7b36423` |
| `origin/feature/device-resident-tensors` | `351075e4d8af1880330b7c0474d701ca76776dfa` |
| `origin/mdslc/native-libtooling-v1` | `993d1d3544bc69838f54747ea867ba4a54725dd0` |
| `origin/mdslc/matcore-ir-v1-cpu-planner` | `58d3e7c75eadf5a71416870ac4c99b4829db8566` |

`main` and the historical default branch diverged at
`1d5072306513d566b03879ccfb7890517ae8ffc9` with counts `45 37`.

The isolated branch `mdslc/mainline-integration-v2` contains these focused
integration commits:

1. `a66dbdf7a294021e93d6c4443cdea150a73cd4c2` — merge the native compiler
   history into the mainline base;
2. `6f68cde775803c2b594dfec5182fef06c43e717b` — preserve the final Milestone 2
   review lineage;
3. `0125cf28204e13edcd87d935347abffc7783c5de` — reconcile demonstrated runtime
   and source-level semantic merge defects; and
4. `0418b2f4f3b98412f0c7c688ace46927fc3c0d01` — repair device benchmark timing
   and checksum semantics.

All four remote tips above are ancestors of the integration branch. The
branch is 160 commits ahead and zero commits behind the fetched `origin/main`.
`git diff --quiet 58d3e7c -- compiler` succeeds, proving that the standalone
compiler tree is unchanged from the reviewed Milestone 2 checkpoint.

## Dirty-state handling

The original feature worktree contained only historical benchmark/report
deletions, review-cache metadata changes, and generated CMake detector output.
There were no dirty production source, public header, Python frontend, or
project CMake input files.

- tracked historical artifacts were restored in place;
- untracked generated detector output was moved to the recoverable quarantine
  `/home/hamza-usta/MatcoreDSL-dirty-quarantine.cJsLm6`;
- no commit, branch, tag, or tracked source file was deleted; and
- ignored outputs generated during integration validation were moved to the
  desktop trash after testing.

Both the original and integration Git worktrees were clean at the end of the
local checkpoint.

## Semantic merge review

An independent read-only overlap review audited all 15 paths changed on both
parents. It found partial auto-merges in plan target propagation, CUDA pointer
classification and directional copies, deterministic pipeline dumps, GPU
mapping helpers, transform parsing, cache-object retention, and a shape-aware
MLIR call. It also found CUDA-array residency and asynchronous benchmark
measurement gaps.

The repaired tree:

- carries the plan's normalized target into repeated tensor parsing;
- authenticates CUDA array residency without treating malformed interfaces as
  device buffers;
- avoids implicit NumPy copies during device validation;
- zeros device output through the device runtime;
- dispatches host/device CUDA copies by resolved pointer kind;
- restores missing mapping helpers and static launch metadata;
- retains deterministic NVIDIA pipeline dumps;
- preserves disk-cache object files only when explicitly requested;
- synchronizes device benchmark samples and uses device-aware checksums; and
- retains the feature/default branch's validated lowering order instead of
  applying an incompatible experimental async transform from `main`.

No high- or medium-severity overlap finding remains.

## Standalone MDSLC validation

Final committed-tree reruns:

```sh
cmake --build /tmp/mdslc-mainline-release -- -j2
ctest --test-dir /tmp/mdslc-mainline-release --output-on-failure -j1

cmake --build /tmp/mdslc-mainline-debug -- -j2
ctest --test-dir /tmp/mdslc-mainline-debug --output-on-failure -j1
```

Results:

- Release: 14/14 passed in 65.52 seconds;
- Debug: 14/14 passed in 66.06 seconds;
- both runs include the native frontend, adversarial frontend suite, driver
  pipeline, integration matrix, installed consumer, IR v1 core, CPU runtime,
  vectorized object, and planner CLI checks.

## Legacy compatibility validation

The source was configured and built with coherent Clang 21.1.8, LLVM/MLIR
18.1.8, Python 3.14, CMake, and Ninja:

```sh
cmake -S . -B /tmp/matcore-mainline-legacy18-py314 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=/usr/bin/python3.14 \
  -DPython_ROOT_DIR=/usr \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
cmake --build /tmp/matcore-mainline-legacy18-py314 -- -j2
```

The complete 42-step extension build linked successfully. Runtime results
before restoring the original system packages:

- focused frontend/fusion/tracer regression: 32 passed;
- pytest-compatible legacy suite: 68 passed;
- CUDA graph regression: 4 passed;
- CPU matmul script: f32, f16, AVX2, AVX-512, int8/i32, BF16-storage/f32,
  rectangular and capability-rejection cases passed;
- GPU elementwise script: 24 cases passed;
- GPU softmax script: 7 cases passed.

A first broad `pytest tests` invocation additionally exposed three script-like
GPU functions that require positional arguments and therefore are not pytest
fixtures. They were excluded from pytest and their files were executed through
their intended script entry points, where all cases passed. The same broad run
found a merge-only CUDA graph lowering failure; removing the incompatible
main-only async-transform invocation restored the feature/default lowering
sequence and made all four graph cases pass.

The linker emitted pre-existing text-relocation warnings for some legacy JIT
cache objects. They were not hidden and are not represented as fixed here.

## System package restoration

The legacy build required a temporary `libmlir-18-dev` installation. Apt
removed the installed MLIR 22 libraries during that coherent-version swap.
After all legacy validation, the following approved command restored the
machine's original MLIR 22 development surface:

```sh
sudo apt-get install --no-install-recommends -y \
  libmlir-22 libmlir-22-dev mlir-22-tools
```

Final package verification shows `libmlir-22`, `libmlir-22-dev`, and
`mlir-22-tools` at `1:22.1.2-1ubuntu1`; the temporary `libmlir-18` and
`libmlir-18-dev` packages are absent. Clang development packages remain
coherent at 21.1.8.

## Local verdict

The history-preserving mainline consolidation and compatibility proof passed
locally. Remote publication, hosted milestone creation, pull-request checks,
and final merge remain separate gates.

