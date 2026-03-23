# Hardware Command Set Lookup for MatCore

Local inspection date: 2026-03-23

Scope:
- Read-only inspection of `/usr/lib/llvm-18/include` and `/usr/include/llvm-18`
- Web cross-checks against vendor docs and product pages
- Focus: matrix/math acceleration paths that matter for MatCore lowering, not general SIMD coverage

High-confidence local conclusions:
- Upstream LLVM/MLIR 18 on this host has strong first-class surface area for `NVGPU`/`NVVM`, `AMDGPU`/`ROCDL`, `AMX`, and generic `vector`
- Upstream LLVM/MLIR 18 on this host has builtin FP8 types `f8E4M3FN`, `f8E5M2`, `f8E4M3FNUZ`, `f8E5M2FNUZ`
- Upstream LLVM/MLIR 18 on this host does not expose builtin FP4 or FP6 types, and I did not find Blackwell-specific FP4/FP6 MLIR/NVVM ops in the inspected headers
- Upstream LLVM/MLIR 18 on this host does not ship `mlir/Dialect/AIE`, `AIEX`, `XeGPU`, `XeVM`, `VPU`, or `NPU` dialect headers
- On x86, explicit MLIR dialect coverage is strongest for `AMX`; `X86Vector` exists but does not model BF16/VNNI matmul directly in the inspected headers

## Coverage Summary

| Area | Local LLVM/MLIR 18 status | Practical meaning for MatCore |
|---|---|---|
| NVIDIA warp-level tensor core | Present | `vector.contract -> nvgpu.mma.sync -> nvvm.mma.sync -> llvm.nvvm.*` is a real path |
| NVIDIA warpgroup tensor core | Present | Hopper-style `nvgpu.warpgroup.mma -> nvvm.wgmma.mma_async` exists locally |
| NVIDIA FP8 builtin types | Present | Builtin MLIR FP8 types exist; NVVM WGMMA also has FP8 enum surface |
| NVIDIA FP4 / FP6 builtin types | Not found | Blackwell FP4/FP6 is vendor-real, but not represented in this LLVM/MLIR 18 install |
| AMD CDNA MFMA | Present | `amdgpu.mfma` and `rocdl.mfma.*` exist locally, including BF16 and gfx940 FP8/BF8 forms |
| AMD RDNA3 WMMA | Present | `amdgpu.wmma` and `rocdl.wmma.*` exist locally |
| Intel AVX-512 BF16 / VNNI | LLVM intrinsic surface present | Lower via generic vector/LLVM/x86 features; no dedicated BF16/VNNI matmul MLIR dialect op found |
| Intel AMX | Present | Best explicit MLIR x86 matrix path in this install |
| Intel AMX-FP16 | LLVM intrinsics present, MLIR AMX user ops absent | Backend can know `tdpfp16ps`, but the inspected MLIR AMX dialect does not expose a user-facing FP16 tile matmul op |
| AMD/XDNA or Intel NPU dialects in upstream LLVM 18 | Not found locally | Treat as external toolchains, not stock LLVM/MLIR 18 lowering targets on this host |

## Lookup Table

| Vendor / family | ISA / instruction set | Native math dtypes | Likely MLIR 18 path | LLVM / IR / intrinsic path | Local header / file refs | External sources / notes |
|---|---|---|---|---|---|---|
| NVIDIA Ada / Lovelace, 4th-gen Tensor Cores | Warp-level Tensor Core `mma.sync` / WMMA | Publicly: `TF32`, `BF16`, `FP16`, `FP8`, `INT8`, `INT4`; local `nvvm.mma.sync` surface explicitly covers `f16`, `bf16`, `tf32`, `u8/s8`, `u4/s4`, `b1` | `linalg.matmul -> vector.contract -> nvgpu.mma.sync -> nvvm.mma.sync`; older WMMA path also exists | `llvm.nvvm.mma.*`, `llvm.nvvm.wmma.*`, PTX `mma.sync` / `wmma.*` | `mlir/Dialect/NVGPU/IR/NVGPU.td` (`nvgpu.mma.sync`), `mlir/Dialect/LLVMIR/NVVMOps.td` (`nvvm.mma.sync`, `nvvm.wmma.*`), `llvm/IR/IntrinsicsNVVM.td` | NVIDIA Ada architecture pages say 4th-gen Tensor Cores support FP8/BF16/FP16/INT8/INT4: <https://www.nvidia.com/en-us/design-visualization/ada-lovelace-architecture/>, <https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf>, <https://www.nvidia.com/en-us/data-center/l40s/> |
| NVIDIA Hopper / H100 / SM90 | Warpgroup Tensor Core `wgmma.mma_async` plus warp-level `mma.sync` | Local WGMMA enum surface covers `f16`, `tf32`, `u8`, `s8`, `b1`, `bf16`, `e4m3`, `e5m2`, `f32`, `s32`; vendor docs emphasize FP8 TE on Hopper | `linalg.matmul -> vector.contract -> nvgpu.warpgroup.generate.descriptor / nvgpu.warpgroup.mma / nvgpu.warpgroup.mma.store -> nvvm.wgmma.*`; fallback warp-level route still valid | `llvm.nvvm.wgmma.*` and PTX `wgmma.*`, plus `llvm.nvvm.mma.*` | `mlir/Dialect/NVGPU/IR/NVGPU.td` (`nvgpu.warpgroup.mma`), `mlir/Dialect/LLVMIR/NVVMOps.td` (`nvvm.wgmma.mma_async`, `wgmma.fence`, `wgmma.commit`, `wgmma.wait`) | Hopper pages and H100 product pages call out FP8 Tensor Cores / Transformer Engine: <https://www.nvidia.com/en-gb/technologies/hopper-architecture/>, <https://www.nvidia.com/en-us/data-center/h100/>, PTX ISA: <https://docs.nvidia.com/cuda/parallel-thread-execution/> |
| NVIDIA Blackwell / 5th-gen Tensor Cores | Public Blackwell Tensor Core path; near-term likely WGMMA-family evolution | Public docs mention `FP4`, `FP6`, `FP8`, `BF16`, `FP16`, `INT8`; local LLVM/MLIR 18 only exposes FP8 builtin types and Hopper-style WGMMA enums | In this LLVM/MLIR 18 install there is no clear Blackwell-specific MLIR lowering surface beyond existing `NVGPU`/`NVVM` WGMMA; treat Blackwell FP4/FP6 as future extension work | Builtin MLIR types: `f8E4M3FN`, `f8E5M2`; no local FP4 / FP6 builtin types found; no inspected `nvvm` op names mentioning Blackwell FP4 | `mlir/IR/BuiltinTypes.td` (`Builtin_Float8E4M3FN`, `Builtin_Float8E5M2`), absence of FP4/FP6 in inspected builtin type headers, `mlir/Dialect/LLVMIR/NVVMOps.td` still Hopper-shaped | NVIDIA Tensor Core pages now mention FP6/FP4 on Blackwell-class hardware: <https://www.nvidia.com/en-us/data-center/tensor-cores/>, <https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf>. For MatCore, this is vendor-real but not stock-LLVM-18-local yet. |
| AMD CDNA2 / CDNA3 / Instinct MI200 / MI300 | MFMA / XDLOPS matrix core path | Local `amdgpu.mfma` and `rocdl.mfma.*` cover `f32`, `f64`, `f16`, `bf16`, `i8`; local gfx940-only rows add `fp8` / `bf8` combos | `linalg.matmul -> vector.contract or explicit gpu/amdgpu matmul rewrite -> amdgpu.mfma -> rocdl.mfma.* -> LLVM AMDGPU` | `llvm.amdgcn_mfma_*` via ROCDL translation | `mlir/Dialect/AMDGPU/IR/AMDGPU.td` (`amdgpu.mfma`), `mlir/Dialect/LLVMIR/ROCDLOps.td` (`rocdl.mfma.*`), `mlir/Dialect/LLVMIR/ROCDLConversions.inc`, `mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h` | AMD MI300X public material highlights FP8/FP16/BF16 AI throughput: <https://www.amd.com/content/dam/amd/en/documents/partner-hub/instinct/instinct-mi300x-acceleration-power-ai-hpc-infographic.pdf>. Local LLVM 18 directly confirms MFMA BF16 and gfx940 FP8/BF8 op surface. |
| AMD CDNA3 gfx940-specific FP8/BF8 | FP8/BF8 MFMA | Local ROCDL op names explicitly include `bf8_bf8`, `bf8_fp8`, `fp8_bf8`, `fp8_fp8`; AMDGPU dialect uses AMD-specific `f8E4M3FNUZ` and `f8E5M2FNUZ` types | Usually same MFMA lowering route as CDNA matmul, but type selection and chipset gating matter | `llvm.amdgcn_mfma_f32_*_bf8_*`, `llvm.amdgcn_mfma_f32_*_fp8_*` through ROCDL | `mlir/Dialect/LLVMIR/ROCDLOps.td` rows marked `// fp8, only on gfx940`, `mlir/Dialect/AMDGPU/IR/AMDGPU.td` FP8/FNUZ types and FP8 pack/unpack ops | Important mismatch: NVIDIA-style `f8E4M3FN` is not the same as AMD’s local `f8E4M3FNUZ`; cross-vendor FP8 needs an explicit type policy. |
| AMD RDNA3 | WMMA | Local `amdgpu.wmma` input types: `f16`, `bf16`, `i8`, `si8`, `ui8`; outputs: `f32`, `i32`, `f16`, `bf16`; ROCDL also has `iu4` | `linalg.matmul -> vector.contract or explicit rewrite -> amdgpu.wmma -> rocdl.wmma.* -> LLVM AMDGPU` | `llvm.amdgcn_wmma_*` via ROCDL | `mlir/Dialect/AMDGPU/IR/AMDGPU.td` (`amdgpu.wmma`), `mlir/Dialect/LLVMIR/ROCDLOps.td` (`rocdl.wmma.*`), `mlir/Dialect/LLVMIR/ROCDLConversions.inc` | rocWMMA docs explicitly show both CDNA and RDNA support in the API tables: <https://rocm.docs.amd.com/projects/rocWMMA/en/docs-6.4.3/API_Reference_Guide.html>. LLVM 18 local headers clearly treat `amdgpu.wmma` as an RDNA3 wrapper. |
| AMD CDNA4 / MI350 near-term | Public matrix-core generation after MI300 | Vendor marketing is public around `FP8`, `BF16`, `INT8`, and low-bit formats such as `MXFP4` / `MXFP6`; local LLVM/MLIR 18 inspected here does not expose MX-format builtin types or ROCDL op names | No stock local LLVM 18 path found beyond existing `AMDGPU` / `ROCDL` FP8/BF8 MFMA surface | No inspected local FP4 / FP6 / MX-format AMDGPU or ROCDL types | Absence in `mlir/IR/BuiltinTypes.td`, `mlir/Dialect/AMDGPU/IR/AMDGPU.td`, `mlir/Dialect/LLVMIR/ROCDLOps.td` | Treat as future AMDGPU/ROCDL extension work, not current upstream LLVM 18 reality on this host. |
| Intel x86 AVX-512 BF16 | `VDPBF16PS` and BF16 conversion family | `bf16` inputs, `f32` accumulate / output in the core dot-product instruction set | `linalg.matmul -> vector.contract -> generic vector lowering with x86 features`; optionally mixed with `enable-x86vector`, but no BF16-specific x86vector matmul op was found | `llvm.x86.avx512bf16.dpbf16ps.*`; Clang builtins `__builtin_ia32_dpbf16ps_{128,256,512}` | `llvm/IR/IntrinsicsX86.td` lines for `int_x86_avx512bf16_dpbf16ps_*`, `llvm/TargetParser/X86TargetParser.def` feature `avx512bf16` | Intel AMX overview and Intel AVX-512 FP16/BF16 materials: <https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-amx.html>, <https://builders.intel.com/docs/networkbuilders/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide-1651874188.pdf>. For LLVM/MLIR 18, BF16 matrix semantics are mostly intrinsic/backend-driven, not X86Vector-op-driven. |
| Intel x86 AVX-512 VNNI and AVX-VNNI | `VPDPBUSD`, `VPDPBUSDS`, `VPDPWSSD`, `VPDPWSSDS`; AVX2/AVX-VNNI variants also exist in LLVM feature plumbing | `i8/u8` inputs, `i32` accumulation | `linalg.matmul -> vector.contract -> generic vector lowering -> LLVM x86 dot-product intrinsics or backend combine`; no inspected dedicated MLIR VNNI matmul dialect op | `llvm.x86.avx512.vpdp*`, `llvm.x86.avx2.vpdp*`; feature flags `avx512vnni`, `avxvnni`, `avxvnniint8`, `avxvnniint16` | `llvm/IR/IntrinsicsX86.td` VNNI section, `llvm/TargetParser/X86TargetParser.def` | Same caution as BF16: this is real LLVM/x86 lowering surface, but not represented by a rich first-class MLIR matrix dialect in the inspected headers. |
| Intel x86 AMX (BF16 / INT8) | AMX tiles + `TDPBF16PS`, `TDPBSSD`, `TDPBSUD`, `TDPBUSD`, `TDPBUUD` | `bf16 -> f32`, `i8/u8 -> i32` | `linalg.matmul -> vector/tile transform -> amx.tile_load / amx.tile_mulf or amx.tile_muli / amx.tile_store -> AMX to LLVM IR` | `llvm.x86.tdpbf16ps_internal`, `llvm.x86.tdpbssd_internal`, etc. | `mlir/Dialect/AMX/AMX.td`, `mlir/Dialect/AMX/AMXDialect.h`, `mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h`, `mlir/Dialect/AMX/AMXConversions.inc` | This is the cleanest explicit x86 matrix path in stock MLIR 18 on this host. `createConvertVectorToLLVMPass` has `enable-amx`; see `mlir/Conversion/Passes.td` and `Passes.h.inc`. |
| Intel x86 AMX-FP16 | `TDPFP16PS`, `TCMMIMFP16PS`, `TCMMRLFP16PS` | `fp16` tile math | No user-facing MLIR AMX FP16 matmul op was found in the inspected AMX dialect; likely requires direct LLVM intrinsic route or newer MLIR surface | `llvm.x86.tdpfp16ps`, `llvm.x86.tcmmimfp16ps`, `llvm.x86.tcmmrlfp16ps`; feature `amx-fp16` | `llvm/IR/IntrinsicsX86.td` AMX-FP16 section, `llvm/TargetParser/X86TargetParser.def` feature `amx-fp16`; notable absence from `mlir/Dialect/AMX/AMX.td` user-facing ops | Important gap: LLVM 18 backend knows AMX-FP16, but the inspected MLIR AMX dialect still only exposes BF16 and INT8 user-facing tile matmul ops. |
| Intel x86 AVX-512 FP16 | AVX-512 half-precision scalar/vector arithmetic and FMA | `fp16` | `linalg/arith/vector -> generic vector lowering -> LLVM x86 AVX-512 FP16 intrinsics`; not a dedicated matrix-core path by itself | `llvm.x86.avx512fp16.*`; feature `avx512fp16` | `llvm/IR/IntrinsicsX86.td` `int_x86_avx512fp16_*`, `llvm/TargetParser/X86TargetParser.def` feature `avx512fp16` | Useful for elementwise or packing paths; not a first-class tile/TensorCore analogue in MLIR 18. |
| AMD Zen 5 public x86 path | Same x86 ISA family, vendor-specific core generation | Public AMD product pages I checked clearly expose `AVX512`; local LLVM target parser exposes `znver4`, not a distinct `znver5` subtype; BF16/VNNI lowering still uses the same x86 intrinsic/feature machinery as Intel | Same as x86 generic route: `vector.contract -> LLVM/x86 features`, optionally AMX where hardware supports it | `avx512bf16`, `avx512vnni`, `avxvnni`, `avx512fp16`, AMX features when present; no Zen-specific LLVM intrinsic family exists | `llvm/TargetParser/X86TargetParser.def` has `znver4` but no `znver5`; all relevant math features live in the shared x86 feature table | AMD Ryzen 9000 product pages show AVX512 support, but I did not find a stock LLVM/MLIR 18 Zen-only matrix dialect path: <https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html> |
| AMD / Xilinx AI Engine, Ryzen AI, XDNA NPU-style path | AI Engine / XDNA compiler path | Vendor stacks target NPU data types and graph ops; exact low-level matmul ISA exposure is not represented in stock upstream LLVM 18 headers on this host | No stock upstream LLVM 18 dialect found locally; if MatCore wants this, it should treat it as an external backend integration rather than a normal in-tree MLIR target | No inspected `AIE`, `AIEX`, `AIEVec`, or `XDNA` headers in local LLVM 18 include trees | Absence confirmed in `/usr/lib/llvm-18/include/mlir` and `/usr/include/llvm-18`; no `mlir/Dialect/AIE` directory on this host | There is an external MLIR-AIE ecosystem and AMD Ryzen AI stack, but that is not the same as stock upstream LLVM/MLIR 18 on this host: <https://ryzenai.docs.amd.com/_/downloads/en/1.1/pdf/>. Use with an out-of-tree toolchain assumption, not with `find_package(MLIR 18)` alone. |
| Intel NPU / VPU-style path | OpenVINO / vendor compiler path | Vendor NPU datatypes and graph kernels | No clear upstream LLVM/MLIR 18 dialect mapping was found in local headers; treat as vendor-runtime / vendor-compiler integration | Not represented as local MLIR dialects in the inspected include trees | No local `XeGPU`, `XeVM`, `VPU`, or `NPU` dialect headers found under `/usr/lib/llvm-18/include/mlir` or `/usr/include/llvm-18` | Intel’s NPU story is exposed through OpenVINO and vendor software stacks rather than stock local LLVM 18 headers on this machine: <https://downloadmirror.intel.com/849225/NPU_Win_Release_Notes_v3764.pdf> |

## Local File Reference Index

Core local files that matter most for MatCore target routing:

- NVIDIA:
  - `/usr/lib/llvm-18/include/mlir/Dialect/NVGPU/IR/NVGPU.td`
  - `/usr/lib/llvm-18/include/mlir/Dialect/LLVMIR/NVVMOps.td`
  - `/usr/include/llvm-18/llvm/IR/IntrinsicsNVVM.td`
  - `/usr/lib/llvm-18/include/mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h`
  - `/usr/lib/llvm-18/include/mlir/Conversion/VectorToGPU/VectorToGPU.h`
- AMD:
  - `/usr/lib/llvm-18/include/mlir/Dialect/AMDGPU/IR/AMDGPU.td`
  - `/usr/lib/llvm-18/include/mlir/Dialect/LLVMIR/ROCDLOps.td`
  - `/usr/lib/llvm-18/include/mlir/Dialect/LLVMIR/ROCDLConversions.inc`
  - `/usr/lib/llvm-18/include/mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h`
  - `/usr/lib/llvm-18/include/mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h`
- x86 / Intel / AMD CPU:
  - `/usr/lib/llvm-18/include/mlir/Dialect/AMX/AMX.td`
  - `/usr/lib/llvm-18/include/mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h`
  - `/usr/lib/llvm-18/include/mlir/Dialect/X86Vector/X86Vector.td`
  - `/usr/include/llvm-18/llvm/IR/IntrinsicsX86.td`
  - `/usr/include/llvm-18/llvm/TargetParser/X86TargetParser.def`
  - `/usr/lib/llvm-18/include/mlir/Conversion/Passes.td`
  - `/usr/lib/llvm-18/include/mlir/Conversion/Passes.h.inc`
- Builtin low-precision type system:
  - `/usr/lib/llvm-18/include/mlir/IR/BuiltinTypes.td`
  - `/usr/lib/llvm-18/include/mlir/IR/BuiltinTypes.h`
  - `/usr/lib/llvm-18/include/mlir/IR/Builders.h`

## MatCore Design Notes

1. Use builtin MLIR FP8 types when the backend truly supports them:
   - NVIDIA path: `f8E4M3FN`, `f8E5M2`
   - AMD path: local AMDGPU ops also expose `f8E4M3FNUZ`, `f8E5M2FNUZ`
   - Do not silently alias NVIDIA FP8 and AMD FNUZ FP8; they are different encodings

2. Distinguish "compile target capability" from "host execution capability":
   - Example: LLVM/MLIR 18 can represent Hopper WGMMA FP8 even if the current host GPU is Ada
   - Example: vendor-public Blackwell FP4 exists, but this local LLVM/MLIR 18 install does not expose FP4 types or ops

3. On x86, do not equate `enable-x86vector` with a full matmul story:
   - `X86Vector` is real, but the inspected ops are mostly mask/scalef/rsqrt/intersect/dot style helpers
   - `AMX` is the only clearly explicit matrix dialect path in the local MLIR 18 install
   - AVX-512 BF16/VNNI/FP16 matmul is mostly an LLVM-intrinsic / target-feature path in the inspected headers

4. AMD BF16 requires care:
   - `ROCDLOps.td` and `AMDGPU.td` clearly expose BF16 matrix instructions
   - `AMDGPUToROCDL.h` still states that ROCDL does not support LLVM `bfloat` natively and rewrites it through `i16`
   - MatCore should keep this as an explicit lowering policy choice, not an accidental cast

5. NPU routes should start as external backend integrations, not in-tree MLIR assumptions:
   - Nothing in the inspected stock LLVM/MLIR 18 include tree justifies claiming first-class local `AIE`, `XDNA`, or Intel `NPU/VPU` dialect coverage
   - Capability registry entries for these should say "external toolchain required" unless the project vendors the extra dialect stack
