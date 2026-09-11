# MatcoreDSL / MDSLC

MatcoreDSL is a mathematical/HPC compiler project. Its experimental `mdslc-region`
compiler admits closed mathematical regions inside ordinary C++, preserving
immutable values, explicit host-storage effects and checked failure semantics.
Its bounded generated CPU path runs on native Linux x86-64 and ARM64. Explicit
x86 AVX/AVX2/AVX512F choices preserve strict arithmetic; optional staged GPU
candidates run on the qualified RTX 4060 Laptop (`sm_89`) and Radeon 890M
(`gfx1150`) tuples. Native and numerically legal x86 provider choices coexist.

Start with the compact [current checkpoint](docs/mdslc/CURRENT_STATE.md), then the
[build/install and language guide](docs/mdslc/REGION_COMPILER_V1.md) and
[executable two-GEMM example](compiler/examples/experimental/two_gemm.mdsl).
The [foundation evidence](docs/mdslc/CPU_FOUNDATION_CHECKPOINT_V1.md) distinguishes
what was actually executed from inspection, compatibility and future work.

The new route is opt-in, uses coherent Clang/LLVM/MLIR 21.1.8, and has no frozen
source API or ABI. Whole-region transformation/fusion, automatic storage reuse,
device-resident intermediates, NPU/Metal execution and BLAS parity are not
established. Compiler-issued isolated GEMM primitives traverse MLIR/LLVM;
region orchestration is statically compiled from authenticated Matcore semantics.
GPU allocations/transfers are an explicit property of the forced staged
candidate, not zero-copy or automatic target selection. See the
[target contracts and campaign evidence](docs/mdslc/MULTITARGET_CORRECTNESS_CAMPAIGN_V1.md).

The standalone compiler lives under `compiler/`. Existing mutating C++ GEMM,
CPU runtime/provider and Windows compatibility retain their separate contracts.
Root Python/JIT/native-extension code remains a separate Linux-tested compatibility surface;
[context.md](context.md) contains historical background, not the current compiler
checkpoint. Generated region execution is not claimed on Windows. Other devices,
toolchain tuples and ISA extensions require separate qualification; successful
lowering alone does not establish target support or competitive performance.
