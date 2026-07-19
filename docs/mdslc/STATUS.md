# MDSLC bootstrap status

Status date: 2026-07-19
Integration baseline: `351075e4d8af1880330b7c0474d701ca76776dfa`

## Current verdict

The architecture and environment have been audited, and the standalone design
is frozen in ADR 0001. No standalone compiler implementation has been
integrated yet. The architecture proof is therefore **not yet passed**.

The original checkout contained 50 tracked deletions. Work proceeds only in
isolated worktrees so those changes remain untouched.

## Acceptance status

| Goal | State | Evidence or remaining gate |
| --- | --- | --- |
| Preflight and baseline | Complete | Branch/SHA, dirty state, tools, memory, devices, and dependency gaps recorded in `PREFLIGHT.md` |
| Architecture freeze | Complete in docs branch | Valid-C++ source model, post-Sema recognition, JSON IR v0, C ABI, CPU-first path, and future capability model recorded in ADR 0001 |
| Standalone CMake skeleton | Not started in this branch | Must configure under `compiler/` without Python, nanobind, or MLIR |
| Goal 2 `.mdsl` host compile | Not yet accepted | Raw `/usr/bin/clang++-21 -x c++` syntax test passed; no `mdslc++`, executable, or relocatable-object proof exists yet |
| Goal 3 post-Sema extraction | Blocked | Matching LibTooling development headers are absent |
| Goal 4 CPU GEMM vertical slice | Not started | Depends on verified extraction and rewrite |
| Goal 5 install/consumer | Not started | Depends on stable driver/runtime targets |
| Goal 6 validation matrix | Not started | Must validate positive, negative, clean, sanitizer, artifact, and runtime cases |
| Goal 7 CUDA/cuBLAS | Not attempted | Optional only after Goals 1-6 pass |

## Selected bootstrap toolchain

For driver-only work, use Clang 21.1.8 explicitly:

```text
/usr/bin/clang-21
/usr/bin/clang++-21
/usr/bin/llvm-config-21
```

This selection avoids PATH's default experimental Clang 22.1.6 and aligns with
the installed LLVM/Clang 21.1.8 packages. It does **not** make LibTooling
available: `libclang-21-dev` and its Tooling/ASTMatcher/Rewriter headers remain
missing and require explicit installation approval.

Build concurrency is capped at Ninja `-j2`, with `-j1` for large links and
ctest. ccache 4.12.3 is available.

## Device status

- CPU execution target: available, not yet validated through MDSLC.
- NVIDIA RTX 4060 Laptop GPU, compute capability 8.9, CUDA 13.3: detected only;
  no MDSLC compile or runtime validation.
- AMD `gfx1150`, HIP 7.1: detected only and out of v0 implementation scope.
- RyzenAI `aie2p`: detected only and out of scope.

## Command log

Commands completed for this documentation milestone:

```sh
git remote -v
git status --short --branch
git branch --show-current
git rev-parse HEAD
git log --oneline --decorate -20
git worktree list
git submodule status --recursive

uname -a
cat /etc/os-release
getconf LONG_BIT
lscpu
nproc
free -h
swapon --show
df -h
ulimit -a

clang --version
clang++ --version
gcc --version
g++ --version
llvm-config --version
cmake --version
ninja --version
ccache --version
ld.lld --version
ld --version
ar --version
nm --version
readelf --version
python3 --version

find /usr/lib /usr/local/lib -path '*cmake/llvm/LLVMConfig.cmake'
find /usr/lib /usr/local/lib -path '*cmake/clang/ClangConfig.cmake'
find /usr/lib /usr/local/lib -path '*cmake/mlir/MLIRConfig.cmake'
ls -d /usr/lib/llvm-*

nvidia-smi
nvcc --version
hipcc --version
rocminfo
rocm-smi
clinfo
```

`rocm-smi` was the expected nonfatal missing command. No build, test, runtime,
artifact, sanitizer, or legacy regression result is claimed by this docs-only
milestone.
