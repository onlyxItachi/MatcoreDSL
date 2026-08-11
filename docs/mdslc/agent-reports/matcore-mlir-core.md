# Matcore MLIR core lane report

Date: 2026-08-11

## Ownership and result

The lane owned `compiler/lib/mlir/`, `compiler/tools/matcore-mlir/`,
`compiler/tests/mlir/`, and the narrow opt-in/install wiring in
`compiler/CMakeLists.txt`. Commit `e0dee79` adds a real TableGen `mdsl` dialect,
a destination-aware and effectful `mdsl.gemm`, a strict Matcore IR v1 bridge,
the internal `matcore-mlir` inspection CLI, deterministic goldens, and negative
verification tests. Review fixes `339ff7b` and `a66ade8` complete the
floating-point environment, semantic-root liveness, recovered-source
extensibility, and strict-versus-relaxed numerical contracts.

The existing standalone build remains unchanged by default. The new surface is
enabled only with `MDSLC_ENABLE_MATCORE_MLIR=ON`; enabling it without an
explicit audited `MLIR_DIR` fails configuration rather than falling back.

## Audited toolchain tuple

- C/C++ compiler: `/usr/bin/clang-21` and `/usr/bin/clang++-21`, version
  21.1.8.
- LLVM CMake package: `/usr/lib/llvm-21/lib/cmake/llvm`, version 21.1.8.
- Clang CMake package: `/usr/lib/llvm-21/lib/cmake/clang`, version 21.1.8.
- MLIR CMake package:
  `/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir`,
  version 21.1.8.

Configuration requires exact 21.1.8 LLVM, Clang, and MLIR packages and the
specific IR, Func, Parser, DestinationStyle, and SideEffect component targets.
A configure without `MLIR_DIR` and a configure against system MLIR 22 were both
confirmed to fail with actionable diagnostics.

## Implemented semantic boundary

- Each verified capture operation becomes one independent public semantic-entry
  function, so operation-local dynamic dimension symbols do not acquire
  cross-site identity and standard SymbolDCE cannot erase unconsumed semantic
  sites. This visibility is internal-IR liveness, not a native/exported ABI
  promise. Milestone E must consume or demote these functions with
  translation-unit-safe names before machine emission.
- Ranked tensor types carry rank, static/dynamic shape, and element dtype.
- Verified attributes preserve accumulation dtype, shape symbols, strides,
  layout, memory space, alignment and alias preconditions, mutability, effects,
  synchronization, policy, source expressions, source ranges, and source
  locations.
- `origin` is a version-extensible dictionary. The v1 bridge authenticates
  `kind = explicit_call` and `canonical_callee = matcore::mdsl::gemm`. The
  core dialect also verifies two fail-closed recovered-loop states without a
  forged callee: an analysis-only strict increasing-K form whose rewrite is
  rejected, and a relaxed source-proven form that still requires a dominating
  pre-mutation guard. Permission/profile cross-combinations are rejected.
- `mdsl.gemm` implements both `DestinationStyleOpInterface` and
  `MemoryEffectOpInterface`. It reports lhs/rhs reads and an observable output
  write. Its SSA result is the post-overwrite value tied to the explicit
  destination, not an independent allocation.
- The bridge requires the named `explicit-gemm-f32-v1` context and encodes all
  reviewed numerical and floating-point-environment fields. It never invents
  permissions from the target. Alignment and no-alias metadata remain required
  preconditions rather than optimizer facts.
- The module records `mdsl.execution_intent = "generic"`. Inference and training
  are enumerated at the bridge boundary but rejected for v1 until their
  semantics are validated; no intent silently grants caching, immutability, or
  numerical permission.
- `verifyMatcoreV1BridgeModule` deliberately verifies the explicit v1 capture
  envelope, not every future compositional dialect module. General dialect
  operations use normal MLIR verification; tests separately prove recovered
  `mdsl.gemm` parse/verification and explicit-envelope rejection.

Encoding this profile does not prove runtime conformance. Floating-point
environment enforcement and lowering guards remain a Milestone E obligation.

## Commands and evidence

The opt-in build was configured with:

```sh
cmake -S compiler -B /home/hamza-usta/.cache/mdslc-semantic-mlir-build -G Ninja \
  -DBUILD_TESTING=ON \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_MATCORE_MLIR=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DMLIR_DIR=/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /home/hamza-usta/.cache/mdslc-semantic-mlir-build -- -j2
ctest --test-dir /home/hamza-usta/.cache/mdslc-semantic-mlir-build \
  --output-on-failure -j1
```

Results from committed source trees:

- opt-in complete build at `339ff7b`: passed;
- opt-in CTest at `339ff7b`: 52/52 passed in 117.20 seconds;
- focused semantic executable at `a66ade8`: 204 checks, 0 failures;
- focused CLI contract at `a66ade8`: 9 checks, 0 failures;
- fresh default-OFF build: passed;
- unchanged default CTest surface: 50/50 passed in 125.32 seconds;
- installed CLI emitted bytes identical to the reviewed semantic MLIR golden.

An in-flight full run after `a66ade8` passed 51/52 tests. The shared branch
advanced to documentation commit `7021338` after benchmark provenance was
embedded, and only the authenticated native-BLAS-parity runner rejected that
stale source commit. No semantic, frontend, integration, package, runtime, or
planner test failed. The lead intentionally deferred the final full rerun until
all review/status documentation settles, so one clean provenance refresh can
validate the final tree.

The installed `matcore-mlir` is an ordinary x86-64 ELF PIE. `readelf`, `ldd`,
and `strings` showed no extracted MLIR toolchain path or source/build path in
the installed executable, no RUNPATH, and no MLIR dependency leaked into the
installed CMake package files.

## Initial failures and corrections

Incremental compilation exposed and corrected the following before acceptance:

1. The tool subdirectory was initially absent from the build graph; the
   opt-in subdirectory and leaf install rule were added.
2. MLIR TableGen did not inherit external include directories; each generation
   command now receives the exact MLIR include directory, while compilation
   receives both LLVM and MLIR includes.
3. Generated interfaces required bytecode/side-effect interface headers and
   produced upstream unused-parameter warnings; required headers/components
   were linked and `-Werror` was not imposed on generated code.
4. MLIR 21 type and `OwningOpRef` API differences were corrected after the
   first narrow compile.
5. Default attribute/type parser hooks were removed because this dialect has no
   custom attribute or type definitions; keeping those hooks produced undefined
   link symbols.
6. One CLI negative expected older v1-version diagnostic wording; the test was
   corrected to the actual strict v1 parser diagnostic and rerun.
7. Four initial full-suite failures were authenticated benchmark guards
   correctly rejecting a dirty implementation worktree. After committing and
   refreshing exact source provenance, all four passed without weakening the
   guards.
8. Independent review found that infinity behavior and execution intent were
   not explicit, per-site private functions could be erased by SymbolDCE, the
   core verifier was coupled to explicit-call provenance, and wide textual
   integer attributes could reach unsafe casts. The reviewed fixes add exact
   contracts, public semantic liveness roots with real SymbolDCE coverage,
   versioned origin/provenance branches, source-file consistency, and exact
   integer-width/range checks.
9. A second independent pass found recovered ordinary C++ was still forced to
   inherit the relaxed explicit-eDSL tuple. The core now uses a closed numerical
   vocabulary and exact origin/permission/profile cross-products, including an
   analysis-only strict increasing-K profile. Positive textual round-trip and
   malformed-cross-product tests close that gap.

## Linkage and installation limits

The semantic implementation links narrow static MLIR component archives,
including IR, Func, DestinationStyle, SideEffect, and Parser where needed. The
final executable uses the coherent system `libLLVM.so.21.1`; it does not use an
aggregate shared MLIR library or a private extracted-toolchain RUNPATH.

`matcore_mlir_semantics` is deliberately internal and is neither installed nor
exported. Only the leaf `matcore-mlir` executable is installed. Existing
`find_package(MatcoreDSL)` consumers therefore acquire no MLIR headers,
libraries, or CMake dependency. Current semantic support is limited to verified
explicit rank-2 F32 GEMM capture and inspection plus core-only recovered-loop
representation tests. It does not yet recognize recovered C++ loops, establish
their guards, lower or execute through MLIR, or claim runtime
floating-point-environment conformance.
