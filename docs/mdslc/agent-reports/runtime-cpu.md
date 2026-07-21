# Runtime and ABI agent report

## Ownership and result

This agent changed only the v0 C runtime header, runtime subdirectory, runtime
tests, and this report. Legacy Python/JIT/MLIR sources were not modified.

The slice provides a versioned, fixed-layout C ABI and exported
`matcore_runtime_gemm_f32_v0` symbol. The synchronous implementation accepts
only positive rank-2, row-major contiguous f32 host tensors with `target=cpu`
and `fallback=error`. It performs no allocation, copying, fallback, or device
migration. Failures return static-lifetime diagnostics and no C++ exception
crosses the ABI.

Validation covers three GEMM shapes against an independently ordered
double-accumulation oracle and the major ABI, pointer, dtype, rank, shape,
layout, mutability, residency, target, fallback, overflow, alignment, and
output/input-alias error paths. Alias validation compares complete touched byte
ranges rather than only pointer identity.

## Evidence

Direct Clang 21 warning-clean build and execution:

```sh
/usr/bin/clang++-21 -std=c++20 -Wall -Wextra -Werror -fPIC -shared \
  compiler/lib/runtime/cpu_runtime.cpp -Icompiler/include \
  -o build-runtime-direct/libmatcore_runtime.so
/usr/bin/clang++-21 -std=c++20 -Wall -Wextra -Werror \
  compiler/tests/runtime/runtime_cpu_test.cpp -Icompiler/include \
  -Lbuild-runtime-direct \
  -Wl,-rpath,/home/hamza-usta/MatcoreDSL-wt-runtime/build-runtime-direct \
  -lmatcore_runtime -o build-runtime-direct/runtime_cpu_test
build-runtime-direct/runtime_cpu_test
```

Result: `runtime CPU GEMM v0: all tests passed`.

Standalone CMake/CTest:

```sh
cmake -S compiler/lib/runtime -B build-runtime-cmake -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 -DCMAKE_BUILD_TYPE=Release
cmake --build build-runtime-cmake -- -j2
ctest --test-dir build-runtime-cmake --output-on-failure -j1
```

Result: 1/1 tests passed.

ASan/UBSan Debug build used `-fsanitize=address,undefined
-fno-omit-frame-pointer`; CTest passed 1/1 with leak detection and UBSan stack
traces enabled. A Clang 21 C11 `-fsyntax-only -Wall -Wextra -Werror` probe also
included `runtime_c.h` successfully.

Artifact inspection identified an x86-64 ELF shared object with SONAME
`libmatcore_runtime.so.0` and exactly the intended public implementation symbol:

```text
0000000000001110 T matcore_runtime_gemm_f32_v0
```

The first direct test run exposed unsigned underflow in test-data generation;
the test generator was corrected to subtract through `int`, and the exact build
and all CMake/sanitizer validations then passed.

## Deliberate limits

- GEMM is assignment semantics (`out = lhs * rhs`) with f32 accumulation.
- Only host CPU execution is implemented; CUDA, AMD, mixed residency, and
  permissive fallback are explicit errors.
- Only rank-2 positive contiguous row-major matrices are legal.
- Runtime validation cannot diagnose source locations; generated host wrappers
  and the frontend retain that responsibility.
