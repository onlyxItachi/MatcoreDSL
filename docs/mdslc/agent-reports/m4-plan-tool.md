# Milestone 4 planner CLI lane

## Ownership

This lane changed only `compiler/tools/matcore-plan/` and this report. It did
not change planner policy, runtime/backends, the public C ABI, or root CMake.

## Result

`matcore-plan` now consumes the shared resource-aware CPU planner v2 and the
shared compiled-backend registry. Shape planning accepts `--threads` and all
five stable variant IDs:

- `cpu.reference.f32.v1`
- `cpu.tiled.f32.v1`
- `cpu.compiler-vectorized.avx2-fma.f32.v1`
- `cpu.external.openblas.f32.v1`
- `cpu.native-packed.avx2-fma.f32.v1`

Legacy short names remain command-line aliases only. The emitted planner v2
diagnostic lists every candidate's legality or rejection reason, estimated
cost, workspace/alignment, and actual thread count. A separate implementation
resource record reports CPU discovery, the authenticated OpenBLAS provider,
and native-packed build/runtime/workspace availability. `--platform-info`
continues to emit the versioned platform record.

An explicit unavailable OpenBLAS request remains a recognized request, exits
nonzero, and reports `OpenBLAS CBLAS adapter is not linked`; it never selects a
different implementation.

## Validation

Clang 21.1.8 Release with OpenBLAS 0.3.32 required:

```text
cmake -S compiler -B /tmp/matcore-m4-plan-release -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=ON -DMDSLC_REQUIRE_OPENBLAS=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-m4-plan-release --target matcore-plan -- -j2
ctest --test-dir /tmp/matcore-m4-plan-release --output-on-failure -j1 \
  -R '^planner\.(platform\.cli|cpu\.cli\.)'
7/7 passed
```

Clang 21.1.8 Release with `MDSLC_ENABLE_OPENBLAS=OFF`:

```text
cmake -S compiler -B /tmp/matcore-m4-plan-no-openblas -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=OFF \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-m4-plan-no-openblas --target matcore-plan -- -j2
ctest --test-dir /tmp/matcore-m4-plan-no-openblas --output-on-failure -j1 \
  -R '^planner\.(platform\.cli|cpu\.cli\.)'
7/7 passed
```

The focused suite covers the platform record, explicit reference selection,
automatic selection, all five registered IDs, byte-for-byte deterministic
diagnostics, zero-thread rejection, and invalid alignment. Forced OpenBLAS in
the provider-disabled build exited 1 with a complete five-candidate diagnostic
and no fallback.

## Limitations

The CLI reports the planner's current static cost model; this lane does not
calibrate or alter that policy. Thread counts above one are currently usable by
the OpenBLAS candidate only, as reflected by candidate diagnostics.
