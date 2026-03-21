# NVIDIA Backend Notes

- `target="nvidia-dgpu"` should be treated as a GPU offload path, not a host-CPU vectorization path.
- The practical MLIR 18 route is `linalg`/`tensor` -> bufferization -> tiling/packing -> `vector`/`nvgpu` -> `nvvm` -> `llvm` -> PTX/CUBIN.
- Do not expect plain `linalg.matmul` to become Tensor Core code automatically. You need explicit tiling/packing and a contraction-style lowering step that exposes the right fragment shapes.

# Recommended Lowering Route

- Start from `linalg.matmul` or `linalg.generic` on ranked tensors.
- Bufferize before GPU lowering.
- Apply matmul tiling/packing so the contraction is visible at warp-friendly granularity.
- For Tensor Core candidates, rewrite to `vector.contract` and then to `nvgpu.mma.sync` or `nvgpu.warpgroup.mma` when the target architecture supports it.
- Use `convert-nvgpu-to-nvvm` and `convert-gpu-to-nvvm` for the device side, then `convert-nvvm-to-llvm` and LLVM NVPTX codegen for final PTX emission.
- If packaging a full GPU module, use `gpu-kernel-outlining`, `nvvm-attach-target`, and `gpu-module-to-binary` as the final serialization stage.

# Dialects And Passes

- Core dialects to keep in the pipeline: `linalg`, `tensor`, `bufferization`, `scf`, `vector`, `gpu`, `nvgpu`, `nvvm`, `llvm`.
- Useful MLIR passes and transforms: `-linalg-block-pack-matmul`, `-convert-vector-to-gpu`, `-convert-nvgpu-to-nvvm`, `-convert-gpu-to-nvvm`, `-convert-nvvm-to-llvm`, `-gpu-lower-to-nvvm-pipeline`.
- Transform-dialect helpers worth using for NVIDIA matmul: `nvgpu_rewrite_matmul_as_mma_sync`, `nvgpu_pipeline_shared_memory_copies`, `nvgpu_create_async_groups`, and `nvgpu_rewrite_copy_as_tma`.

# FP16 And BF16

- FP16 is the safest Tensor Core input type on NVIDIA Volta/Ampere-class parts and is the canonical path in MLIR contraction lowering.
- BF16 Tensor Cores are supported on Ampere-class devices and later; CUDA documents that `__nv_bfloat16` uses the same shapes/operations as `__half`, with `float` accumulation.
- In MLIR, the Tensor Core friendly mixed-precision pattern is the one where `linalg.matmul` or `vector.contract` folds extra `extf` operations into contraction form, enabling native instructions like `mma.sync.*.f32.f16.f16.f32` and `mma.sync.*.f32.bf16.bf16.f32`.
- For MatCore, prefer FP32 accumulation for both FP16 and BF16 inputs unless a target-specific benchmark proves a narrower accumulator is worthwhile.

# LLVM 18 Constraints And Gaps

- LLVM 18 exposes the NVIDIA lowering pieces, but it does not auto-select Tensor Cores from high-level matmul code.
- `gpu-lower-to-nvvm-pipeline` expects already-parallel IR and does not perform GPU parallelization for you.
- `nvgpu.mma.sync` is the clean intermediate for warp-level MMA, while `nvgpu.warpgroup.mma` is the newer warpgroup route for architectures that support it.
- A robust MatCore implementation should keep an explicit architecture gate, because `bf16` and warpgroup/TMA paths depend on compute capability and are not universally available.

# Sources

- https://mlir.llvm.org/docs/Dialects/GPU/
- https://mlir.llvm.org/docs/Dialects/NVGPU/
- https://mlir.llvm.org/docs/Dialects/NVVMDialect/
- https://mlir.llvm.org/docs/Passes/
- https://mlir.llvm.org/docs/Dialects/Vector/
- https://mlir.llvm.org/doxygen/structFoldArithExtIntoContractionOp.html
- https://mlir.llvm.org/python-bindings/autoapi/mlir/dialects/transform/nvgpu/index.html
- https://docs.nvidia.com/cuda/archive/11.5.0/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/archive/11.0/ampere-tuning-guide/
- https://docs.nvidia.com/cuda/archive/12.4.0/turing-tuning-guide/index.html
- https://docs.nvidia.com/cuda/pdf/Turing_Tuning_Guide.pdf
