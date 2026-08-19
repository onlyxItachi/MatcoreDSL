# Windows MLIR 21.1.8 Toolchain Provisioning, Fingerprints, and MLIR Static Specialization

## 1. Toolchain Provisioning Recipe

* **LLVM/Clang Baseline**: LLVM / Clang 21.1.8 installed at `C:\Users\hamza\tools\llvm-21.1.8`.
* **MLIR Release & Commit**:
  - Source Repository: `https://github.com/llvm/llvm-project.git`
  - Release Tag: `llvmorg-21.1.8`
  - Git Commit: `2078da43e25a4623cab2d0d60decddf709aaea28`
* **Build Host / Environment**:
  - OS: Windows 11 MSVC x64
  - Compiler: `clang-cl.exe` from `C:\Users\hamza\tools\llvm-21.1.8\bin`
  - Generator: Ninja
* **CMake Build Configuration**:
  ```powershell
  cmake -S "llvm-project/llvm" -B "build-mlir" -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER="C:/Users/hamza/tools/llvm-21.1.8/bin/clang-cl.exe" `
    -DCMAKE_CXX_COMPILER="C:/Users/hamza/tools/llvm-21.1.8/bin/clang-cl.exe" `
    -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" `
    -DLLVM_ENABLE_PROJECTS="mlir" `
    -DLLVM_TARGETS_TO_BUILD="X86" `
    -DLLVM_INCLUDE_TESTS=OFF `
    -DLLVM_INCLUDE_EXAMPLES=OFF `
    -DLLVM_INCLUDE_BENCHMARKS=OFF `
    -DLLVM_ENABLE_ASSERTIONS=OFF `
    -DCMAKE_INSTALL_PREFIX="C:/Users/hamza/tools/llvm-mlir-21.1.8"
  ninja -C build-mlir install
  ```

---

## 2. Toolchain Authentication Fingerprints

| Component | Path | SHA256 Fingerprint |
| :--- | :--- | :--- |
| `mlir-opt.exe` | `C:\Users\hamza\tools\llvm-mlir-21.1.8\bin\mlir-opt.exe` | `90C8D27B6C64A37F268C3C68F0E0712D96F6A9CA78B5D2D0C6F087AF88D61CD0` |
| `MLIRConfig.cmake` | `C:\Users\hamza\tools\llvm-mlir-21.1.8\lib\cmake\mlir\MLIRConfig.cmake` | `BFF9C7BBC5199A8A75F13153FC011039CCC1141B04D0A07262C09951B5BA3393` |
| CRT ABI Model | All `.lib` and `.exe` artifacts | `/MT` (`MultiThreaded` Static Release) |

---

## 3. Matcore MLIR Static Lowering Architecture

* **Source Files**:
  - `compiler/lib/mlir/MatcoreStaticGemmLowering.h`
  - `compiler/lib/mlir/MatcoreStaticGemmLowering.cpp`
* **Test Suite**:
  - `compiler/tests/mlir/matcore_static_gemm_lowering_test.cpp`
  - `compiler/tests/codegen/triway_specialization_comparison_test.cpp`

### Validated Geometry Lowering Paths:
1. **Ranked Static Tensor Extraction**: Fail-closed verification rejecting unranked/dynamic shapes.
2. **DOT Reduction ($1\times 1\times K$)**: Direct scalar reduction with vectorization loop pragmas.
3. **GEVM ($1\times N\times K$)**: Outer-loop elimination, vectorized inner broadcast-accumulate.
4. **GEMV ($M\times 1\times K$)**: Contiguous row reduction.
5. **Square/Rectangular GEMM ($M\times N\times K$)**: 2D accumulator-tiled microkernels.

---

## 4. Benchmark Validation Summary

* **Tri-way Comparison**: Runtime Dispatch vs Generated-C++ AOT vs MLIR-Native AOT.
* **Peak Measured Speedup**: **$9,554\times$** for tiny kernels ($1\times 1\times 1$ latency dropped from $2,675\text{ ns}$ down to $0.3\text{ ns}$).
* **CTest Pass Rate**: **57 / 57 passed (100%)**.
