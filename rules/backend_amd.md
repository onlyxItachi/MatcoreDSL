# AMD Backend Notes

## `target="amd-igpu"`

- Preferred route in MatCore: `linalg` / `vector` / `memref` lowering first, then map into `gpu` for workgroup/subgroup structure, then lower `gpu` to AMD-specific LLVM IR through `rocdl`.
- Use `gpu` as the middle layer and keep AMD-specific details late. MLIR documents `gpu` as the generic launch/kernel dialect, `rocdl` as the 1:1 wrapper layer for AMDGPU LLVM intrinsics, and `amdgpu` as the higher-level AMD wrapper dialect for hardware-specific matrix and shuffle operations.
- For matrix math, prefer generic `vector.contract`/`linalg` forms until you know the chip. Promote to `amdgpu.mfma` or `amdgpu.wmma` only when the target chipset is known and the tile shape matches the hardware.
- Practical lowering sequence for AMD iGPU:
  - tile/fuse `linalg`
  - bufferize as needed
  - map loops to `gpu.launch`
  - `-convert-vector-to-gpu`
  - `-gpu-decompose-memrefs` if the GPU path needs explicit size/stride metadata
  - `-convert-gpu-to-rocdl -chipset=<gfx...>`
  - `-rocdl-attach-target`
  - `-gpu-to-llvm`
- Keep chipset selection explicit. MLIR’s AMDGPU passes are chipset-aware, and the docs state that `convert-math-to-rocdl`, `convert-gpu-to-rocdl`, and the AMDGPU dialect itself can depend on the chosen `gfx` target.

## `target="amd-npu"`

- On this host, the AMD NPU route is not an in-tree LLVM 18 lowering path.
- `mlir-aie` / `llvm-aie` are separate AMD/Xilinx toolchains for AI Engine targets; MLIR core only references MLIR-AIE as a user project, not a built-in LLVM 18 dialect stack.
- Treat `amd-npu` as scaffold-only unless the separate AIE/XDNA toolchain is vendored or installed. The compiler should emit an explicit unsupported/backend-missing error rather than silently falling back to CPU.
- If MatCore later vendors AIE tooling, the likely route is an AIE-oriented IR path rather than `gpu`/`rocdl`: tile decomposition, stream/DMA setup, and AIE core kernels compiled through the AIE-specific LLVM fork/toolchain.

## FP16 / BF16 Notes

- AMDGPU MLIR has first-class support for half-precision matrix math in the `amdgpu` dialect. `amdgpu.wmma` explicitly documents `f16` and `bf16` matrix multiply forms on newer gfx targets.
- The docs also show `amdgpu.mfma` / `amdgpu.wmma` as wrappers around hardware matrix intrinsics, which is the right place for fast FP16/BF16 tensor-core style lowering on AMD GPUs.
- Keep generic `fp16` / `bf16` arithmetic in `vector` or `arith` until a target chip is known. Do not assume every AMDGPU target supports the same matrix instruction shapes.
- Be careful with chipset-specific edge cases. The `amdgpu.wmma` docs note gfx11/gfx12 differences, including `subwordOffset` behavior and partially valid result lanes on gfx11.

## Constraints For MatCore

- Implement an explicit backend router keyed on `amd-igpu` and `amd-npu`.
- For `amd-igpu`, require a target chip attribute or configuration knob so the compiler can select the correct AMDGPU/ROCDL patterns.
- For `amd-npu`, fail closed unless a separate AIE/XDNA stack is available.
- Do not encode AMD matrix ops as a generic CPU fallback. The whole point of the AMD path is to preserve target-specific lowering until the backend can legally select `amdgpu`/`rocdl` or AIE-specific codegen.

## Sources

- https://mlir.llvm.org/docs/Dialects/GPU/
- https://mlir.llvm.org/docs/Dialects/ROCDLDialect/
- https://mlir.llvm.org/docs/Dialects/AMDGPU/
- https://mlir.llvm.org/docs/Passes/
- https://mlir.llvm.org/users/
- https://github.com/Xilinx/mlir-aie
- https://github.com/Xilinx/llvm-aie
