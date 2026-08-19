# Explicit Evidence Gaps Log

This document explicitly catalogues all boundaries where physical evidence was limited, hardware execution was unavailable, or external toolchain dependencies were required.

---

## 1. Documented Gaps & Status

| Gap ID | Description | Classification | Reason & Disposition |
| :--- | :--- | :--- | :--- |
| **GAP-0001** | AMD GPU Physical Execution | `hardware-unavailable` | The host Windows machine does not have an active AMD CDNA/RDNA GPU runtime. AMD lowering evidence is gathered via compiler target code generation (`-target amdgcn-amd-amdhsa`), LLVM AMDGPU tests, and MLIR `amdgpu` dialect sources. |
| **GAP-0002** | NVIDIA CUDA Toolkit Linking on Windows | `toolchain-unavailable` | Prebuilt LLVM provides the NVPTX backend (`-target nvptx64-nvidia-cuda`), but full link/execution requires the host NVIDIA CUDA Toolkit (NVRTC/CUDA Driver). Code generation to PTX is verified; physical execution on GPU is marked deferred. |
| **GAP-0003** | Proprietary cuBLAS / oneMKL Internals | `unsupported` | Proprietary BLAS library binaries are opaque. They serve strictly as API/ABI/dispatch performance oracles; internal lowering sequences are not fabricated. |
| **GAP-0004** | Standalone `mlir-opt.exe` in Windows Prebuilt Archives | `toolchain-unavailable` | Upstream official Windows LLVM releases do not bundle standalone `mlir-opt.exe` binaries. MLIR analysis is conducted via MLIR C++ headers, TableGen specifications, and MDSLC's internal `matcore-mlir` tool. |
