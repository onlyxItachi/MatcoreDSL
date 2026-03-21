# x86 CPU Backend Notes

## Recommended Route

- `target="x86-avx2"`: `linalg` matmul -> bufferization -> vectorization -> `convert-vector-to-llvm` -> LLVM codegen.
- `target="x86-avx512"`: same base route, but prefer `x86vector` lowering where BF16/F16 patterns exist, then lower to LLVM.
- For both targets, keep the CPU path bufferized and rely on `vector.transfer_read` / `vector.transfer_write` plus `vector.contract` or `vector.fma` where the rewrite is available.

## Half Precision

- MLIR `vector` supports floating-point element types, and LLVM lowering on CPU turns `n-D` vectors into LLVM-compatible vector/array forms.
- For BF16 on x86, MLIR’s `x86vector` dialect has explicit AVX/AVX-512 BF16 ops, including BF16/F16 broadcast-to-F32 conversions and an AVX-512 BF16 dot product that can lower to `llvm.dpbf16ps`.
- For FP16, the MLIR 18 docs I found expose F16 broadcast/convert forms in `x86vector`, but not a full, first-class x86 FP16 compute path comparable to BF16.
- Practical rule: treat BF16 as the preferred half-precision CPU type on AVX-512; treat FP16 on x86 as backend-dependent and allow upcast-to-F32 fallback when legalization does not select native instructions.

## Passes And Dialects

- Required dialects: `linalg`, `vector`, `memref`, `scf`, `arith`, `func`, `cf`, `llvm`.
- Useful x86-specific dialect: `x86vector`.
- Likely pass sequence:
  - `bufferization` / memref conversion before vector lowering
  - `convert-linalg-to-loops` or linalg vectorization rewrites, depending on the kernel shape
  - `lower-vector-to-from-elements-to-shuffle-tree`
  - `lower-vector-mask` and `lower-vector-multi-reduction` if masks or reductions appear
  - `convert-vector-to-llvm` with x86 enabled
  - `finalize-memref-to-llvm`
  - `convert-scf-to-cf`
- Use `vector.contract` / `vector.fma` as the main bridge for matmul-like accumulation when the rewrite path is available.

## LLVM 18 Constraints

- LLVM’s CPU lowering still treats `vector` lowering as a structured `vector -> LLVM` rewrite; 1-D vectors are native, while `n-D` vectors become LLVM aggregates/arrays.
- AVX2 has no native half-precision compute path equivalent to AVX-512 BF16/FP16 features, so BF16/FP16 will usually need widening or legalization help.
- AVX-512 BF16 support is the most concrete MLIR 18 half-precision CPU path I found.
- I did not find an MLIR 18 doc page showing a complete native AVX-512 FP16 compute pipeline on the x86 side, so treat that as a host/backend capability rather than an assured MLIR abstraction.

## Sources

- https://mlir.llvm.org/docs/Dialects/Linalg/
- https://mlir.llvm.org/docs/Dialects/Vector/
- https://mlir.llvm.org/docs/Dialects/X86Vector/
- https://mlir.llvm.org/docs/Passes/
- https://mlir.llvm.org/docs/Dialects/LLVM/
- https://releases.llvm.org/18.1.8/tools/clang/docs/UsersManual.html
