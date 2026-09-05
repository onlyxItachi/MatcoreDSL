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
- Existing per-call native and compatibility execution capture first produce
  the verified Matcore JSON IR v0 compatibility DTO, then cross an explicit,
  mandatory boundary into typed,
  verified Matcore IR v1. `mdslc++` artifacts are v1. Existing rewrite/codegen
  is reached only through a loss-checked v1-to-v0 projection; do not fork the
  schema, verifier, or code generator. JSON remains a versioned inspection
  boundary, not the permanent optimizer architecture.
- Matcore IR v1 remains the deterministic capture/provenance DTO for the
  explicit GEMM surface. The compositional optimizer representation is the
  internal `mdsl` MLIR dialect. Bridge v1 into that dialect through one
  documented, verified conversion boundary; do not mutate v1 in place or add
  another overlapping JSON optimizer schema.
- The opt-in two-GEMM region inspection route uses sealed native declaration
  bindings alongside the per-call contracts. Its descriptor handles are not
  physical no-alias proofs; committed tensor values and ordered observable
  effects are distinct. Preserve every call's failure frontier and host
  barriers. This route must never enter the v1-to-v0 execution projection,
  discharge runtime predicates by assertion, or acquire execution authority
  from serialized attributes. See `docs/mdslc/TWO_GEMM_REGION_V1.md`.
- Preserve semantic information until the final optimization that can use it.
  Matcore semantic operations describe WHAT. Legality, planning, structured
  upstream dialects, scheduling, and library/generated-code selection describe
  HOW. LLVM, target-specific dialects, and platform/vendor toolchains describe
  MACHINE. A lower representation may consume a fact only after encoding its
  meaning structurally or completing every decision that needs it.
- Numerical behavior, effects, aliasing, mutation, synchronization, execution
  intent, and target policy are legality contracts, not optimization hints.
  A required alignment or no-alias field is a precondition rather than proof
  about concrete runtime values; consume it only after static proof or a
  dominating fail-closed guard that runs before output mutation.
- Recognition of an ordinary C++ compute idiom is not permission to replace
  it. Recovered operations require separate dependence, alias/effect,
  numerical, barrier, and source-region legality proofs. If any proof fails,
  leave ordinary C++ behavior untouched and report the rejection reason.
- Explicit and conservatively recovered operations may converge into the same
  Matcore MLIR semantics, but recovered loops must not forge explicit-header
  provenance or inherit the explicit eDSL numerical profile. Keep host C++
  control flow in Clang/source rewriting unless a later milestone explicitly
  proves a larger-region capture.
- Clang is the C++ frontend and host compiler. MDSLC owns semantic admission,
  legality, retained-fact consumption, target policy and planning decisions,
  and orchestration. Upstream MLIR owns its canonical structured, tensor,
  buffer, transform, and vector machinery. LLVM and target backends own
  instruction selection, register allocation, target scheduling, and machine
  lowering. Do not claim that Clang automatically lowers Matcore IR for every
  accelerator.
- The synchronous, row-major, rank-2 `f32` CPU runtime validates descriptors,
  discovers capabilities, and deterministically selects among the registered
  reference, 32x32x64 tiled, compiler-vectorized AVX2/FMA, optional OpenBLAS,
  and native packed AVX2/FMA variants. Forced illegal variants fail without
  fallback. The generated one-shot execution ABI retains its v0 name for
  compatibility; additive v1 APIs expose caller-owned workspace and prepacked
  B storage. Packed execution must never hide allocation or packing.
- OpenBLAS is an optional authenticated CBLAS provider, never a requirement or
  a silent fallback. Control its thread count locally, report the actual
  provider policy, and prevent a nominal single-thread comparison from using
  an uncontrolled provider pool.
- Reference, tiled, and compiler-vectorized v1 execute with and report one
  actual thread even when the request permits more; native packed v1 requires
  exactly one requested thread. None is a parallel implementation.
- Advanced CPU dispatch must distinguish hardware support, OS-enabled extended
  state, compiler support, implementation availability, and physical runtime
  validation. CPUID alone never authorizes AVX-512 or AMX execution. Keep ISA
  requirements on isolated functions or translation units; never compile the
  complete runtime with AVX2, AVX-512, or AMX enabled globally.
- Persistent parallel variants use explicit execution contexts, bounded worker
  counts, deterministic output-tile assignment, caller-visible per-worker
  workspace, and mutually exclusive native/OpenBLAS pools. Do not create
  workers per GEMM, hide nested provider threads, or infer physical NUMA
  validation from synthetic topology tests.
- BF16 and INT8 acceleration follows typed reference semantics. Optimized
  variants may be advertised only when their exact instructions are present,
  legality is fail-closed, and compatible physical hardware has executed the
  correctness suite. Compile-only and synthetic status must remain explicit.
- Platform diagnostics use the versioned Linux/Windows/Unknown record. Shared
  code must not acquire scattered POSIX-only process, path, dynamic-library,
  or object-format assumptions. A platform is supported only after its native
  compiler, linker, runtime, package, and generated artifacts execute in that
  environment.
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
- The product, release, and default Matcore MLIR component must use MLIR
  21.1.8 coherently with the selected LLVM/Clang 21.1.8 tuple. The advanced
  internal compatibility selector may admit only the reviewed coherent
  LLVM/Clang/MLIR 22.1.8 Linux x64 tuple; that path is compatibility evidence,
  not product support or an installed consumer contract. Mixed distro-22 or
  cross-version components remain forbidden. MLIR is an internal build
  dependency and must not leak absolute development-prefix paths or MLIR
  targets into the installed consumer package. Do not couple standalone MDSLC
  to the legacy root project's MLIR 18 requirement.
- Validate incrementally after every meaningful change. Before handoff, run the
  narrow tests, then the complete standalone suite, and repeat from a clean
  build directory when the milestone requires it.
- A successful compile is not enough for an execution milestone: run the
  program, compare GEMM with an independent oracle, inspect the relocatable
  object, and verify the ordinary final link.
- Runtime correctness and `matcore-bench` use an independent double-precision
  oracle. Benchmark modes must distinguish allocation, transient packing,
  reused workspace, prepacked B, cache state, and compute-only diagnostics.
  Raw runs stay under ignored `benchmark_reports/`; only reviewed summaries
  belong under `docs/performance/cpu/`.
- Physical CPU capability discovery is independent from per-build
  implementation availability. Sanitizer builds may report real AVX2/FMA
  hardware while rejecting an instrumented/scalarized compiler-vectorized
  implementation. Never label scalarized work as an ISA implementation.

## Commits and agent handoff

- After each successfully merged engineering boundary, update
  `docs/mdslc/CURRENT_STATE.md`: at most about 20 architecture lines, the material
  change, unsupported/unproven boundaries, exact canonical engineering merge
  SHA/PR, and exactly one justified next boundary. Link detailed evidence;
  do not duplicate it. A documentation-only follow-up records the engineering
  merge, not a self-referential future documentation commit.
- Keep commits focused, independently reviewable, and free of unrelated
  cleanup. Do not squash the implementation history during active development.
- Each specialized agent owns non-overlapping files, records tests and failures
  under `docs/mdslc/agent-reports/`, commits only its scope, and supplies its
  commit SHA to the integration owner. Agents do not merge their own branches.
- Never claim a test, artifact, target, or device path that was not actually
  compiled and, where required, executed.
