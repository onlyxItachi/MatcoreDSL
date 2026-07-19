# MDSLC preflight

Audit date: 2026-07-19
Audit mode: read-only against the original checkout

## Repository baseline

- Remote: `git@github.com:onlyxItachi/MatcoreDSL.git`
- Actual branch: `feature/device-resident-tensors`
- Actual SHA: `351075e4d8af1880330b7c0474d701ca76776dfa`
- Prompt baseline discrepancy: none for branch or SHA.
- Original checkout: `/home/hamza-usta/MatcoreDSL`
- Audit worktree: `/home/hamza-usta/MatcoreDSL-wt-audit`
- Audit branch: `mdslc/audit-and-adr`
- Submodules: none reported by `git submodule status --recursive`.

The original checkout was not clean. `git status --porcelain=v1` reported 50
tracked deletions and no other status class. `git diff --stat` reported 50 files
and 15,466 deleted lines. Deleted paths included benchmark reports, cache
metadata, and the tracked `build-review` CMake/Ninja output. The deletions were
not altered, restored, staged, or copied by this audit. The isolated worktree
started clean from the recorded SHA.

Existing worktrees observed before mutation:

- `/home/hamza-usta/MatcoreDSL` at the baseline SHA;
- `/home/hamza-usta/MatcoreDSL_safe_1d50723`, detached and prunable;
- `/tmp/matcore_commit_check`, detached and prunable.

No pre-existing `mdslc/*` branch was found. The repository had no `AGENTS.md`.
The requested depth-three build-file scan found only `context.md` and the root
`CMakeLists.txt`. `git clean -ndX` emitted no ignored-file candidates, the
checkout occupied 7.0 MiB, and no extant `build*` directory was available for
size reporting because `build-review` was among the tracked deletions.

## Operating system and resources

- OS: Ubuntu 26.04 LTS (Resolute Raccoon)
- Kernel: `7.0.0-27-generic`, x86_64, 64-bit
- CPU: AMD Ryzen AI 9 HX 370 with Radeon 890M
- CPU topology: 12 cores, 24 online hardware threads, one NUMA node
- Relevant CPU features: AVX2 and AVX-512 families are advertised by `lscpu`
- Memory at audit: 14 GiB total, 6.9 GiB available
- Swap: 32 GiB total, approximately 319 MiB used
- Root filesystem: 937 GiB total, 303 GiB available
- `/tmp`: 7.2 GiB total, 6.2 GiB available
- Open-file limit: 524,288
- Stack limit: 8 MiB

Because this host has 14 GiB of RAM and previous repository builds experienced
memory pressure, standalone builds default to Ninja `-j2`. Large link steps and
tests use `-j1`. ccache is already installed and should be used when detected.
Do not start a full LLVM source build.

## Compiler and build tools

| Tool | Observed version |
| --- | --- |
| Default `clang`, `clang++` | Ubuntu Clang 22.1.6 |
| `gcc`, `g++` | Ubuntu GCC 15.2.0 |
| Default `llvm-config` | 22.1.6 |
| CMake | 4.3.2 |
| Ninja | 1.13.2 |
| ccache | 4.12.3 |
| LLD | 22.1.6 |
| GNU `ld`, `ar`, `nm`, `readelf` | binutils 2.46 |
| Python | 3.14.3 |

LLVMConfig and ClangConfig packages were found for versions 17, 18, 20, 21,
and 22. MLIRConfig was found only under `/usr/lib/llvm-22`.

## Coherent Clang/LibTooling selection

No complete LibTooling development tuple is installed today. For every
installed Clang version checked (17, 18, 20, 21, and 22), the executable,
ClangConfig, and monolithic `libclang-cpp` were present, but these required
development headers were absent:

- `clang/Tooling/Tooling.h`
- `clang/ASTMatchers/ASTMatchers.h`
- `clang/Rewrite/Core/Rewriter.h`

No `libclang-*-dev` package is installed, and a filesystem search under `/usr`
and `/opt` found no alternate copies of the required headers. Therefore the
post-Sema `matcore-extract` frontend cannot honestly be configured or compiled
yet.

For driver-only Goal 2 work, select `/usr/bin/clang-21` and
`/usr/bin/clang++-21`, version 21.1.8. A read-only syntax proof passed:

```sh
printf '%s\n' 'int main() { return 0; }' \
  | /usr/bin/clang++-21 -x c++ -std=c++20 -fsyntax-only -
```

LLVM 21.1.8, Clang 21.1.8, `libclang-cpp21` 21.1.8, LLVMConfig, and ClangConfig
are installed. The package metadata candidate for `libclang-21-dev` is also
21.1.8, making LLVM/Clang 21 the preferred future coherent frontend tuple if
the user explicitly approves installation. No package was installed during
this audit.

Do not use the default LLVM/Clang 22 tuple for LibTooling now. The installed
Clang/LLVM packages and ClangConfig are 22.1.6, while the available
`libclang-22-dev` candidate and installed MLIR package are 22.1.2. Combining
them would violate the same-version rule.

The legacy root CMake requests MLIR 18.1.3, but
`/usr/lib/llvm-18/lib/cmake/mlir/MLIRConfig.cmake` is absent. The only installed
MLIRConfig identifies MLIR 22.1.2 and requests LLVM 22.1.2 exactly, whereas the
installed LLVMConfig is 22.1.6. The legacy project therefore cannot be assumed
to reconfigure successfully. MDSLC bootstrap v0 avoids this issue by not
requiring MLIR.

After explicit approval and installation of the matching Clang 21 development
headers, the expected standalone configure shape is:

```sh
cmake -S compiler -B build-mdslc -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang
cmake --build build-mdslc -- -j2
ctest --test-dir build-mdslc --output-on-failure -j1
```

These are expected commands, not recorded passing frontend-build evidence.

## Accelerator inventory

### NVIDIA

- Device: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute capability: 8.9
- Memory: 8,188 MiB
- Driver: 610.43.02
- CUDA UMD: 13.3
- CUDA compiler: nvcc 13.3.73
- `/usr/local/cuda` resolves to `/usr/local/cuda-13.3`

The NVIDIA device and compiler are detected, but no MDSLC CUDA path was
compiled or executed. CUDA/cuBLAS is optional only after CPU Goals 1-6 pass.

### AMD and other accelerators

- `hipcc`: HIP 7.1.52801 using Clang 21.1.8
- `rocminfo`: functional; ROCk module loaded
- AMD GPU agent: `gfx1150`, Radeon Graphics, 16 compute units, wave size 32,
  64 KiB group memory
- AMD NPU/DSP agent: `aie2p`, marketed as RyzenAI-npu4
- `rocm-smi`: not installed
- OpenCL: NVIDIA platform/device available; an Xilinx platform is present but
  its reported device is unavailable and emits query errors

No AMD, HIP, Metal, Xilinx, or NPU backend is in bootstrap-v0 scope.

## Relevant environment

Only these requested environment classes were populated:

```text
LD_LIBRARY_PATH=/usr/local/cuda/lib64
PATH=...:/usr/lib/llvm-22/bin:/usr/lib/llvm-20/bin:/opt/rocm/bin:...:/usr/local/cuda/bin:...
```

No `CC`, `CXX`, `CMAKE*`, `LLVM*`, `CLANG*`, `MLIR*`, `CUDA_HOME`,
`ROCM_PATH`, or `HIP*` variable was set. PATH prioritizes LLVM 22, so the
standalone configuration and spawned compiler command must use explicit
Clang 21 paths rather than inherit an ambiguous default.

## Blockers requiring user action

1. Acknowledge or resolve the original checkout's 50 tracked deletions before
   any workflow that could overwrite them. Isolated worktrees preserve them.
2. Approve installation of the exact matching Clang 21 LibTooling development
   package before Goal 3 frontend compilation.
3. Treat legacy MLIR reconfiguration as separately blocked; do not install or
   change MLIR merely to prove the standalone CPU compiler path.

Goal 1 documentation and Goal 2 driver-only work can proceed in isolated
worktrees. Goal 3 post-Sema extraction cannot pass until the LibTooling header
blocker is resolved.
