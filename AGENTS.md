# MatcoreDSL Repository Guidance

These instructions apply to the whole repository. The standalone MDSLC work is
additive to the existing Python/JIT/native-extension implementation described
in `context.md`.

## Architecture boundary

- Every `.mdsl` translation unit is valid C++ source. `mdslc++` must pass it to
  Clang with `-x c++`; do not introduce a second C++ parser or a C++-like
  language.
- Public matrix operations live only in `matcore::mdsl`. Never add declarations
  to `std` or `std::mdsl`.
- Users opt in through the `.mdsl` extension, the `mdslc++` driver,
  `<matcore/mdsl.h>`, and annotated canonical declarations.
- The supported frontend is the in-process Clang 21 LibTooling path. Recognize
  operations after Sema from `getDirectCallee()`, the canonical resolved
  `FunctionDecl`, its exact `clang::annotate` payload, and its trusted-header
  origin. Textual call matching, unqualified/ADL recognition, and arbitrary
  C++ capture are forbidden.
- Authenticate `<matcore/mdsl.h>` through the directly resolved `FileEntry`
  identity, a stable physical/parsed content snapshot, and the expected public
  ABI semantics. A copied, shadowed, macro-altered, or signature-compatible
  lookalike is not a trusted declaration.
- Output mutation is explicit through `matcore::mdsl::out(C)`. Preserve shape,
  dtype, layout, memory-space, alias, effect, policy, and source-location
  information at compiler boundaries.
- Bootstrap through deterministic, verified Matcore JSON IR v0. JSON is a
  versioned inspection boundary, not the permanent optimizer architecture.
- Clang is the C++ frontend and host compiler. MDSLC owns matrix semantics,
  planning, legalization, scheduling, and target lowering. Do not claim that
  Clang automatically lowers Matcore IR for every accelerator.
- The first execution backend is a synchronous, row-major, rank-2 `f32` CPU
  reference GEMM behind a C ABI. GPU work is not a prerequisite for the CPU
  architecture proof.
- Never silently fall back between targets, silently copy between host and
  device, hide allocation, or propagate C++ exceptions across the C ABI.

## Source and compatibility rules

- Keep the standalone project under `compiler/` independent of root CMake,
  Python, nanobind, and the legacy JIT cache.
- Do not delete, reset, broadly reformat, or mass-migrate legacy code. Treat the
  Python/JIT path as a compatibility and regression surface.
- Unsupported contexts must fail with a nonzero status and an actionable
  diagnostic tied to the original `.mdsl` file, line, and column when Clang
  provides them. Reject unsafe macro, template, lambda, header, indirect-call,
  unevaluated, side-effect, alias, layout, dtype, and residency cases before
  rewriting.
- Rewrite only the exact validated `CallExpr` source range. Never rewrite macro
  expansions or source ranges not owned by the main `.mdsl` file.
- Do not accept user-controlled VFS overlays, precompiled headers, or module
  injection in the v1 frontend. Freeze the main source and its dependency
  closure before extraction and recheck both after every generated compile,
  link, and dependency publication phase.
- Generated host, IR, site, stub, backend, object, and executable files belong
  in the build tree. Do not commit them. Commit deterministic golden fixtures
  only when a test intentionally reviews their complete contents.
- Generated identifiers must be deterministic and collision-safe across call
  sites and translation units. Equivalent generated site wrappers/backends use
  weak definitions so deterministic IDs can safely co-link across independent
  source roots.

## Toolchain and build discipline

The 2026-07-19 audit selected one coherent native-frontend tuple: Clang,
Clang development libraries, and LLVM 21.1.8 from Ubuntu package revision
`1:21.1.8-6ubuntu1`. The user approved the exact
`libclang-21-dev libclang-cpp21-dev` install command; when run, APT reported
both packages already at the newest version and made zero package changes.
Configure the standalone project with that tuple:

```sh
cmake -S compiler -B build-mdslc -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang
cmake --build build-mdslc -- -j2
ctest --test-dir build-mdslc --output-on-failure -j1
```

The AST-JSON bootstrap remains an explicitly selected compatibility and
differential-testing path. It may configure without LLVM/Clang development
CMake packages only when native support is deliberately disabled:

```sh
cmake -S compiler -B build-mdslc -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-mdslc -- -j2
ctest --test-dir build-mdslc --output-on-failure -j1
```

Native is the supported default. Missing native dependencies must fail
configuration or default invocation clearly; they must never trigger a silent
bootstrap fallback. Keep the compatibility producer label and its limitations
explicit. Differential parity is a regression oracle; native Clang declaration
and source-manager semantics remain authoritative.

For the driver-only language proof, use the selected executable explicitly:

```sh
/usr/bin/clang++-21 -x c++ -std=c++20 source.mdsl -o program
/usr/bin/clang++-21 -x c++ -std=c++20 -c source.mdsl -o source.o
```

- Default to Ninja `-j2`; use `-j1` for memory-heavy links. Use ccache when the
  standalone CMake target detects the already-installed tool.
- Do not start an LLVM source build. Additional system package changes require
  the user's authorization and must stay within the task scope.
- Do not mix compiler executables, headers, libraries, or CMake packages from
  different LLVM/Clang releases in one frontend binary.
- Validate incrementally after every meaningful change. Before handoff, run the
  narrow tests, then the complete standalone suite, and repeat from a clean
  build directory when the milestone requires it.
- A successful compile is not enough for an execution milestone: run the
  program, compare GEMM with an independent oracle, inspect the relocatable
  object, and verify the ordinary final link.

## Commits and agent handoff

- Keep commits focused, independently reviewable, and free of unrelated
  cleanup. Do not squash the implementation history during active development.
- Each specialized agent owns non-overlapping files, records tests and failures
  under `docs/mdslc/agent-reports/`, commits only its scope, and supplies its
  commit SHA to the integration owner. Agents do not merge their own branches.
- Never claim a test, artifact, target, or device path that was not actually
  compiled and, where required, executed.
