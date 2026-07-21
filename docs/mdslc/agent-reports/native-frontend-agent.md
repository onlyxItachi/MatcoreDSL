# Native Frontend Agent Report

Date: 2026-07-19

Branch: `mdslc/native-frontend-v1`

Base: `3e3fa5b2d1990e1c37870f8b2096fbda6128716b`

## Ownership

This lane changed the standalone frontend implementation, Matcore IR producer
authentication, `matcore-extract` selection logic, Clang/LLVM discovery, and a
focused native test. It did not change the driver, runtime, package templates,
consumer project, legacy Python/JIT implementation, or root legacy build.

## Implementation

- Added a Clang 21.1.8 LibTooling frontend linked through the imported
  `clang-cpp` and `LLVM` targets.
- Used `PPCallbacks::InclusionDirective` to require a direct, non-macro,
  angle-bracket `<matcore/mdsl.h>` include whose resolved `FileEntry` unique
  identity matches the tool-owned header.
- Used AST matchers after parsing/Sema to collect calls and function
  references. Candidate calls are authenticated through `getDirectCallee()`,
  `getCanonicalDecl()`, the exact canonical signature, trusted declaration-file
  identity, and exactly one non-inherited
  `AnnotateAttr("matcore.op.gemm")`. The `out` wrapper is authenticated in the
  same way with `matcore.wrapper.out`.
- Rejected unqualified, indirect, template, lambda, constexpr, macro, header,
  unsafe-expression, unsupported-policy, and aliasing cases before IR output.
- Derived half-open call and argument byte ranges with `SourceManager` and
  `Lexer::getLocForEndOfToken`; no textual search or estimated token length is
  used.
- Preserved one shared stable site-ID implementation for native and bootstrap
  producers, and reused the existing IR verifier and code generator.
- Made `native` the extractor default. The bootstrap remains available only as
  `--frontend=ast-json-bootstrap`; a native-disabled build fails clearly rather
  than falling back.
- Rejected a `--clang` option or compiler placeholder that differs from the
  configured Clang 21.1.8 executable, since native parsing occurs in-process.

## Commits

1. `f69b300a8b454c691c1e2c8af8a72fd9de71aef1` —
   `feat(frontend): add native Clang 21 LibTooling extraction`
2. `86b47be12473133f8ee75992ee5b918c9e31e0fe` —
   `feat(extractor): make native frontend the explicit default`
3. The report, compiler-placeholder ordering fix, strict-warning cleanup, and
   macro-include hardening are in the lane's final commit.

## Validation evidence

Configured with:

```text
cmake -S compiler -B build-native-lane -G Ninja
  -DCMAKE_C_COMPILER=/usr/bin/clang-21
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-native-lane -- -j2
ctest --test-dir build-native-lane -R '^frontend.native.focused$'
  --output-on-failure -j1
```

Result: focused native suite passed, 16 checks. Native and explicit bootstrap
produced semantically identical IR for `gemm_capture.mdsl` after excluding the
intentional producer field; site ID and every source range matched.

A separate strict build used
`-Wall -Wextra -Wpedantic -Werror` and linked `matcore-extract` successfully.
The validation lane independently exercised 416 native/parity/adversarial
checks against this extractor and reported no frontend defect.

The existing CPU artifact pipeline was run through the native default:

```text
build-native-lane/bin/mdslc++ -std=c++20 --matcore-target=cpu
  --save-temps -c compiler/examples/gemm_v0.mdsl
  -o build-native-lane/gemm_v0.native.o
/usr/bin/clang++-21 build-native-lane/gemm_v0.native.o
  -Lbuild-native-lane/lib -lmatcore_runtime
  -Wl,-rpath,/home/hamza-usta/MatcoreDSL-wt-native-front-v1/build-native-lane/lib
  -o build-native-lane/gemm_v0.native
build-native-lane/gemm_v0.native
```

Result: `host-before` followed by `MDSLC CPU GEMM PASS`. `file` identified the
combined output as an ELF 64-bit relocatable object. `nm -C` showed the stable
generated C++ call-site wrapper and C backend entry, with only
`matcore_runtime_gemm_f32_v0` unresolved before the ordinary final link.

A bootstrap-only build with `MDSLC_ENABLE_NATIVE_FRONTEND=OFF` was also built.
Its default invocation failed with the required native-not-built diagnostic and
emitted no IR; explicit `--frontend=ast-json-bootstrap` succeeded.

## Review fixes

- Lambda calls were initially classified as constexpr because Clang marks a
  lambda call operator constexpr. Context precedence now reports the required
  lambda diagnostic first.
- Compiler coherence was initially checked before normalizing the compiler
  placeholder following `--`. The check now runs afterward, and both mismatch
  forms have tests.
- Direct include authentication now rejects macro-expanded include filenames,
  in addition to checking spelling, angle-bracket form, main-file origin, and
  resolved file identity.

## Integration notes and limitations

- The extractor remains one binary with two explicit modes; it never retries a
  failed native extraction with bootstrap.
- The package lane must retain the configured relative bindir/includedir macros
  when integrating non-default GNUInstallDirs support. The extractor has a
  build-tree `bin/../include` fallback and no embedded checkout path.
- This lane implements only the declared host-resident rank-2 f32 CPU GEMM
  capture contract. It adds no CUDA, BLAS, MLIR lowering, GEMV, GEVM, NPU, or
  capability-planner behavior.
