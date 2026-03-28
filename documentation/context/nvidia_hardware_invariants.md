# NVIDIA Hardware Invariants For Dynamic Macro-Topology

## Purpose

MatCore must separate:

- micro-topology: fixed hardware-facing constants chosen at compile time
- macro-topology: runtime launch geometry derived from actual tensor sizes

For NVIDIA tensor-core lowering, the micro-topology side must stay static. The GPU ISA, fragment layouts, and scheduler contracts are defined around fixed warp and warpgroup semantics. Only the overall number of blocks should vary with runtime `m`, `n`, and `k`.

## Why Micro-Topology Must Stay Static

### 1. Warp-level tensor core instructions encode fixed participant counts

`nvgpu.mma.sync` is the warp-level bridge to `nvvm.mma.sync`. In MLIR 18, the op itself is modeled around a static `mmaShape` such as `[16, 8, 16]`. That is not a runtime quantity. The fragment types, register layout, and lane ownership all depend on that exact shape.

- Warp size is fixed at 32 threads for CUDA tensor core warp-level execution.
- A warp-level `mma.sync` lowering assumes the compiler knows, ahead of codegen, which lane owns which fragment registers.
- Changing the tile shape dynamically would change fragment packing and lane-to-element mapping, which MLIR/NVVM/PTX do not model as runtime parameters.

Relevant local MLIR 18 evidence:

- [`/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPU.td`](#/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPU.td) documents `nvgpu.mma.sync` as the intermediate op for lowering `vector.contract` to `nvvm.mma.sync`, with explicit `mmaShape = [16, 8, 16]` in the example.
- [`/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.td`](#/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.td) describes `transform.nvgpu.rewrite_matmul_as_mma_sync` as a one-to-one rewrite to tensor-core-compatible ops, not a dynamic-shape runtime selection mechanism.

### 2. WGMMA also uses fixed hardware-defined tile families

For Hopper/Blackwell-class warpgroup MMA, MLIR 18 exposes NVGPU warpgroup types and ops that are still statically shaped:

- `mlir::nvgpu::WarpgroupMatrixDescriptorType`
- `mlir::nvgpu::WarpgroupAccumulatorType`
- `mlir::nvgpu::WarpgroupMmaOp`

These are typed around specific descriptor and accumulator layouts. The local MLIR headers also hardcode a WGMMA invariant:

- [`/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPUDialect.h`](#/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPUDialect.h) defines `constexpr int kWgmmaSizeM = 64;`

That is the opposite of a runtime-selected arbitrary tile. The instruction family is defined around legal fixed shapes, shared-memory layout rules, and fixed fragment contracts.

### 3. Register allocation is shape-sensitive

Tensor-core kernels win only if the compiler can allocate fragments without spilling. The fragment count is a function of:

- instruction family: `mma.sync` vs `wgmma`
- operand dtype: `f16`, `bf16`, `s8`, `e4m3`, etc.
- tile shape: for example `m16n8k16`, `m16n16k16`, `m64nNk16`, `m64nNk32`

If those are allowed to float dynamically, LLVM cannot form a stable fragment/register allocation plan. The likely outcomes are:

- fallback away from tensor-core lowering
- inflated register pressure
- reduced active warps per SM
- spills to local memory

That directly harms occupancy and can erase the whole point of using tensor cores.

### 4. Shared-memory promotion also assumes fixed per-block tile footprints

Workgroup/shared-memory promotion is only safe and performant when the compiler knows:

- the per-block tile sizes
- the number of warps or warpgroups per block
- the staging footprint for A and B tiles
- the swizzle or descriptor layout required by the instruction family

For `mma.sync`, the compiler wants static workgroup tile shapes.
For `wgmma`, PTX requires specific shared-memory layouts and descriptor encodings for the operand tiles.

So the amount of data each block stages into shared memory is a compile-time design choice. The number of blocks covering the full matrix is the runtime part.

### 5. Occupancy is controlled by fixed kernel shape, not by dynamic grid math

To approach full SM utilization, the kernel must expose a stable per-block resource model:

- threads per block
- registers per thread
- shared memory per block
- MMA fragment footprint per warp or warpgroup

Those determine how many resident blocks and warps can live on an SM. If these micro-parameters are stable, runtime launch geometry can scale block count to the matrix size without forcing LLVM/PTX to re-solve the instruction layout problem.

This is the right split:

- static micro-topology:
  - warp size = 32
  - warpgroup size = 128 when using WGMMA
  - legal MMA/WGMMA tile families
  - block thread shape
  - per-block shared-memory promotion pattern
- dynamic macro-topology:
  - `grid_x = ceil_div(n, block_tile_n)`
  - `grid_y = ceil_div(m, block_tile_m)`
  - edge behavior for partial tiles

## What Must Remain Static In MatCore

For the current NVIDIA path, these should remain compile-time constants in C++ and IR generation:

- warp size: `32`
- warp-level MMA tile families such as `16x8x16` and `16x16x16`
- WGMMA tile family selection for future `sm_90+`
- thread mapping inside a block
- per-block tile sizes for `M`, `N`, `K`
- block thread dimensions
- shared-memory promotion policy
- vector fragment shapes and NVGPU descriptor types

## What Must Become Dynamic

These must come from runtime `memref.dim` or `tensor.dim`, not host-side C++ division on actual matrix sizes:

- number of blocks in `x`, `y`, `z`
- effective loop bounds for macro tiles
- edge predicates / partial-tile guards

## Architectural Implication For Phase 4.3

MatCore should stop computing `grid_x`, `grid_y`, and `grid_z` in host C++ for NVIDIA lowering.

Instead:

1. Keep the tile family and thread/block micro-shape static.
2. Build GPU macro-launch bounds inside the MLIR module with runtime `memref.dim`.
3. Construct `gpu.launch` using those runtime-computed grid values.
4. Preserve fixed block dimensions so the kernel remains tensor-core legal and occupancy-friendly.

## Source Notes

Local MLIR / LLVM 18 sources:

- [`/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPUDialect.h`](#/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPUDialect.h)
- [`/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPU.td`](#/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPU.td)
- [`/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.td`](#/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.td)

NVIDIA reference material:

- PTX ISA, WGMMA and tensor-core layout sections: https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA C++ Programming Guide, warp-level matrix operations: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
