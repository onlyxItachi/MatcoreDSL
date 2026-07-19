# MDSLC bootstrap roadmap

The roadmap is gate-driven. Higher layers do not proceed on unverified lower
layers, and every implementation milestone remains additive to the legacy
Python/JIT path.

## 1. Standalone build skeleton

Create `compiler/` with independent CMake targets for `mdslc++`,
`matcore-extract`, IR support, runtime support, examples, and tests. The first
configure must not import Python, find nanobind, require MLIR, or modify root
CMake.

Gate:

- clean standalone CMake configure;
- skeleton targets compile with Ninja `-j2`;
- no legacy source changes;
- generated files remain in `build-mdslc/`.

## 2. Valid-C++ driver proof

Implement the small C++ `mdslc++` process driver. It accepts normal compiler
arguments, forces `.mdsl` inputs through `-x c++`, invokes the selected Clang
without a shell, preserves exit status and output, and supports `-c`, final
link, `--verbose`, and `--save-temps`.

Gate:

- `hello_host.mdsl` prints `5` through `mdslc++`;
- the same source produces an ELF relocatable object;
- `file`, `readelf`, and `nm` confirm ordinary object shape;
- no Python, nanobind, Matcore operation, or legacy build is involved.

This work can use `/usr/bin/clang++-21` now.

## 3. Public header and post-Sema capture

After matching Clang 21 development headers are available, add the minimal
`<matcore/mdsl.h>` API and LibTooling extractor. Recognition uses only direct
canonical annotated declarations after Sema. Emit deterministic, verified
Matcore JSON IR v0 with complete source and descriptor information.

Gate:

- qualified and namespace-alias GEMM calls resolve to one canonical operation;
- unrelated calls are ignored;
- deterministic JSON matches a reviewed golden fixture;
- unsupported indirect, template, lambda, macro, header, and constexpr sites
  fail with original-source diagnostics;
- extraction does not rewrite source.

## 4. C ABI, rewrite, and CPU GEMM

Define the stable v0 C descriptors and status model. Add exact-source-range
rewrite, deterministic call-site symbols, generated C ABI stubs, and a
synchronous reference `f32` GEMM for rank-2 contiguous row-major host views.
Reject shape, alias, lifetime, dtype, layout, and residency violations without
hidden allocation, copy, migration, or fallback.

Gate:

- `gemm_v0.mdsl` produces saved host/IR/site/stub/backend artifacts;
- generated objects are partially linked into a normal relocatable `.o`;
- ordinary `clang++` links the object against `libmatcore_runtime`;
- execution matches an independent GEMM oracle for multiple small shapes;
- diagnostics still point to the original `.mdsl` source.

## 5. Installation and external consumer

Install the drivers, public headers, runtime library, and relocatable CMake
package. Provide an external `find_package(MatcoreDSL REQUIRED)` consumer with
a helper that models `.mdsl` generation dependencies explicitly.

Gate:

- clean install tree contains no repository-local absolute paths;
- clean external configure/build/link/run succeeds;
- editing one `.mdsl` source regenerates only its necessary artifacts.

## 6. Adversarial validation

Implement the full positive and negative compile/runtime matrix. Include two
stable call-site IDs, ordinary C++ around calls, non-template functions and
methods, safe host templates, malformed/version-mismatched IR, all initially
forbidden contexts, clean Debug and Release builds, and ASan/UBSan where
supported.

Independently review the integrated diff for textual matching, source-range
loss, ABI leakage, unstable names, multi-TU collisions, hidden allocation or
copy, silent fallback, local paths, LLVM-version mixing, Python leakage,
generated artifacts, and legacy regression.

Gate:

- all declared tests pass from a fresh build directory;
- runtime and artifact checks pass, not compilation alone;
- every high-severity review finding is resolved.

## 7. Optional CUDA/cuBLAS library backend

Only after Goals 1-6 pass, add explicit device-resident descriptors and a
synchronous `target=cuda`, `backend=cublas` path. It must reject host/mixed
residency and unavailable runtime/backend requests when fallback is `error`.

No custom PTX, WGMMA, HIP, ROCm, or Metal implementation belongs in this
milestone. Compile-only results must remain labeled compile-only.

## Future structured compiler bridge

After the bootstrap contract is stable, define one explicit conversion from
Matcore JSON IR v0 into the high-level Matcore IR/MLIR dialect. Adapt useful
legacy verifier, capability, lowering, and runtime concepts behind new
interfaces rather than coupling the frontend to JIT or Python details.

Then implement capability-aware planners and legalizers independently for CPU,
NVIDIA, AMD, and future targets. A backend becomes supported only after normal
artifact production, runtime execution, correctness validation, and declared
performance tests pass on a device covered by its capability model.

## Next three exact engineering tasks

1. Add `compiler/CMakeLists.txt`, empty standalone targets, and one configure
   test that proves Python, nanobind, and MLIR are absent from the new build.
2. Implement the C++ `mdslc++` process driver around `/usr/bin/clang++-21` and
   pass executable plus relocatable-object tests for `hello_host.mdsl`.
3. Obtain explicit approval for matching `libclang-21-dev`, verify its installed
   version and headers, then configure a minimal LibTooling executable before
   writing operation extraction logic.
