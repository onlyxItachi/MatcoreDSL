# Private output alias-fact experiment

Research only; neither the MLIR fixture nor the LLVM editing instrument confers
source admission or product execution authority. Starting product checkpoint:
`58193878394c774c3365dbf746d1622e951b0813` (subsequently merged by PR60).

## Question and observed result

The private generated GEMM contract already requires C data disjoint from both
immutable inputs, but permits A and B to alias. Does stock dynamic-MemRef lowering
preserve that fact when `llvm.noalias` is placed on C's MemRef argument?

**Observed on coherent21.1.8:** no. The fixture's argument attribute reaches
the C-wrapper descriptor pointer, but does not mark the flattened function's
aligned C data pointer. Those are different objects. This is demonstrated by
actual lowering, not an inference from an attribute spelling.

`annotate_output.cpp` then uses LLVM's existing API to mark only the exact
specimen's aligned C data argument. The instrument requires the pinned21-argument
rank2x3 descriptor signature before touching index15. It adds no alias property
to A/B, no nonnull, dereferenceability, alignment or numerical assumption.
This is a supplied contract for a trusted experiment, not verification that
arbitrary imported IR has truthful runtime storage.

Actual baseline-x64 Clang21 `-O2 -ffp-contract=off` objects contain2125 and1403
text bytes respectively. The annotated object has separate SSE `mulps`/`addps`.
This demonstrates different code generation, **not a measured performance win**.

The broad research oracle from
[`6ce4b3f`](https://github.com/onlyxItachi/MatcoreDSL/blob/6ce4b3f6035f1a4ae8f5395753f9e0202eabdffc/compiler/experiments/mlir_hpc_v1/execution.cpp)
passed **77,985 checks** with the annotated ordinary object and another77,985
with actual generated ASan plus harness ASan/UBSan. Its invalid-capacity control
exited1 with a heap-buffer-overflow READ4 and generated wrapper in frame0.
That establishes instrumentation, not a wide SIMD read claim or capacity guard.

## Reproduction

Use exact21.1.8 `mlir-opt`, `mlir-translate`, `llvm-config-21`, Clang and `opt-21`.
The initial local output directory was `/tmp/mdslc-gemm-alias-contract.Ov0Z3d`.

1. Lower `row_output_noalias.mlir` with `convert-linalg-to-loops`, `lower-affine`,
   `convert-scf-to-cf`, `convert-arith-to-llvm`, `finalize-memref-to-llvm`,
   `convert-func-to-llvm`, `convert-cf-to-llvm`, `reconcile-unrealized-casts`.
2. Translate using `mlir-translate --mlir-to-llvmir`.
3. Compile the instrument with LLVM core/irreader flags from `llvm-config-21`,
   C++20, and invoke it with input LLVM and a fresh output LLVM path.
4. Compile both LLVM files with Clang `-O2 -ffp-contract=off`; link the cited
   ordinary C++ oracle without changing its source. For actual generated ASan,
   add `sanitize_address` to generated functions using the research `forceattrs`
   pass, compile that IR with ASan and link the oracle with ASan/UBSan.

The relevant pinned upstream boundary is
[FuncToLLVM](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp):
wrapper argument attributes and unpacked data-pointer attributes are not the
same contract. `memref.distinct_objects` is absent from the installed21 headers
and present in the separate22 tuple; do not assume moving documentation proves
availability in the product toolchain. No new tiler, optimizer or backend was
needed for this experiment.

Next acceptance boundary: a compiler-owned, structurally checked preservation
of this existing output contract, actual source execution and alias/failure
regressions. Never annotate arbitrary external Storage or infer disjoint inputs
from different descriptors. This experiment does not implement that integration.
