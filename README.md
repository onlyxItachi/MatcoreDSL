# MatcoreDSL / MDSLC

MatcoreDSL is a mathematical/HPC compiler project. Its experimental `mdslc-region`
compiler admits closed mathematical regions inside ordinary C++, preserving
immutable values, explicit host-storage effects and checked failure semantics.
It has a validated, bounded Linux x86-64 generated CPU execution path alongside
native and numerically legal provider choices.

Start with the compact [current checkpoint](docs/mdslc/CURRENT_STATE.md), then the
[build/install and language guide](docs/mdslc/REGION_COMPILER_V1.md) and
[executable two-GEMM example](compiler/examples/experimental/two_gemm.mdsl).
The [foundation evidence](docs/mdslc/CPU_FOUNDATION_CHECKPOINT_V1.md) distinguishes
what was actually executed from inspection, compatibility and future work.

The new route is opt-in, uses coherent Clang/LLVM/MLIR 21.1.8, and has no frozen
source API or ABI. Whole-region transformation/fusion, automatic storage reuse,
GPU/NPU execution and BLAS parity are not established. Only the strict GEMM
primitive currently traverses the generated MLIR/LLVM pipeline; the region
orchestration is statically compiled from authenticated Matcore semantics.

The standalone compiler lives under `compiler/`. Existing mutating C++ GEMM,
CPU runtime/provider and Windows compatibility retain their separate contracts.
Root Python/JIT/native-extension code remains a separate Linux-tested compatibility surface;
[context.md](context.md) contains historical background, not the current compiler
checkpoint. Generated region execution is not claimed on Windows or accelerators.
