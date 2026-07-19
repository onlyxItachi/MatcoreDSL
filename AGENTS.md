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
- Recognize operations after Clang Sema from the canonical resolved
  `FunctionDecl` and its `clang::annotate` metadata. Textual call matching,
  unqualified/ADL recognition, and arbitrary C++ capture are forbidden.
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
  side-effect, alias, layout, dtype, and residency cases before rewriting.
- Rewrite only the exact validated `CallExpr` source range. Never rewrite macro
  expansions or source ranges not owned by the main `.mdsl` file.
- Generated host, IR, site, stub, backend, object, and executable files belong
  in the build tree. Do not commit them. Commit deterministic golden fixtures
  only when a test intentionally reviews their complete contents.
- Generated identifiers must be deterministic and collision-safe across call
  sites and translation units.

## Toolchain and build discipline

The 2026-07-19 audit selected `/usr/bin/clang++-21` 21.1.8 for the
driver-only valid-C++ proof. A complete LibTooling build is currently blocked
because matching Clang development headers are not installed. Do not describe
the frontend as buildable until that dependency is resolved. Once the matching
21.1.8 development package is available, configure the standalone project with
one coherent tuple:

```sh
cmake -S compiler -B build-mdslc -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang
cmake --build build-mdslc -- -j2
ctest --test-dir build-mdslc --output-on-failure -j1
```

For the driver-only language proof, use the selected executable explicitly:

```sh
/usr/bin/clang++-21 -x c++ -std=c++20 source.mdsl -o program
/usr/bin/clang++-21 -x c++ -std=c++20 -c source.mdsl -o source.o
```

- Default to Ninja `-j2`; use `-j1` for memory-heavy links. Use ccache when the
  standalone CMake target detects the already-installed tool.
- Do not start an LLVM source build or install packages without explicit user
  approval.
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
