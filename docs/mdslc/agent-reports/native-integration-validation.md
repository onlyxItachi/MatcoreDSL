# Native LibTooling v1 integration validation

Date: 2026-07-19

Integration branch: `mdslc/native-libtooling-v1`

Milestone base: `mdslc/bootstrap-v0` at
`3e3fa5b2d1990e1c37870f8b2096fbda6128716b`

Independently reviewed implementation evidence head:
`f71f1800a1ba70f2b363ff68ecd6632c7ae8fad1`

## Scope and ownership

This report records lead-integration evidence for the native frontend, driver,
package, CPU artifact pipeline, and selected legacy regression checks. The
independent adversarial review and its closure are recorded separately in
`native-frontend-final-review.md`.

No CUDA, BLAS, MLIR lowering, GEMV, GEVM, NPU, or capability-planner
implementation was attempted.

## Toolchain gate

The user approved and the lead ran exactly:

```sh
sudo apt-get install --no-install-recommends \
  libclang-21-dev \
  libclang-cpp21-dev
```

APT reported both packages already newest at Ubuntu revision
`1:21.1.8-6ubuntu1` and made zero changes. The preceding simulation estimated
28.81 MiB download and about 285.53 MiB installed size.

The post-command audit verified a coherent 21.1.8 tuple: `clang-21`,
`clang++-21`, `llvm-config-21`, Tooling, CommonOptionsParser, ASTMatchers,
Rewriter, Lexer, AST, Attr, Frontend, and CompilerInstance headers,
`libclang-cpp.so.21.1`, `libLLVM.so.21.1`, component libraries, and exact LLVM
and Clang CMake packages. A native probe compiled, linked, ran, and resolved
only the matching Clang/LLVM 21 shared libraries.

## Integrated implementation commits

1. `e9d6c65` `docs(mdslc): record native frontend milestone and toolchain gate`
2. `e663e73` `feat(frontend): add native Clang 21 LibTooling extraction`
3. `795c072` `feat(extractor): make native frontend the explicit default`
4. `7749d87` `fix(frontend): harden native authentication boundaries`
5. `0bd611a` `feat(driver): select native frontend without fallback`
6. `db9e14f` `build(mdslc): package native frontend selection`
7. `bb16002` `fix(driver): verify source after every pipeline phase`
8. `f06ae5e` `fix(package): preserve punctuation in relocated paths`
9. `0417629` `test(driver): execute relocated runtime link`
10. `0186c6b` `docs(mdslc): report native driver package lane`
11. `4255870` `test(frontend): add native parity and adversarial validation`
12. `eee446e` `docs(mdslc): report native validation lane evidence`
13. `8cb8080` `fix(frontend): authenticate resolved direct includes`
14. `17e65b0` `fix(package): preserve no-op rebuilds after relocation`
15. `4670566` `test(frontend): make native frontend the primary contract`
16. `e207619` `fix(frontend): reject side-effectful policies`
17. `306c6c0` `fix(frontend): close semantic authentication bypasses`
18. `7c2c282` `fix(mdslc): preserve compilation phase identity`
19. `791e2cb` `fix(codegen): tolerate stable site symbol collisions`
20. `f6d24b3` `fix(frontend): authenticate parsed ABI and evaluated call sites`
21. `6b54482` `fix(driver): freeze dependency and target context`
22. `e30bb6d` `fix(frontend): reject unevaluated builtin operands`
23. `d461407` `fix(driver): publish complete dependency closures`
24. `5bcb7e2` `fix(driver): preserve dependency resolution identity`
25. `aa41c22` `fix(driver): freeze generated compilation inputs`
26. `f71f180` `fix(driver): recheck generated include resolution`

## Native frontend evidence

The default extractor runs ClangTool in-process with PPCallbacks, parse/Sema,
ASTMatcher/ASTConsumer, canonical `FunctionDecl` authentication, exact
`AnnotateAttr` checks, and SourceManager/Lexer ranges. It authenticates the
resolved trusted header's unique file identity, physical/parsed bytes, and
public ABI semantics. Native output uses producer `clang-libtooling-v1`.

Explicit `--frontend=ast-json-bootstrap` remains available with producer
`clang-ast-json-bootstrap-v0`. Missing native support fails default invocation;
there is no automatic retry or fallback. Native/bootstrap parity removes only
the producer field and compares all semantic IR and generated files.

Post-hardening focused results:

```text
native frontend focused: 16 checks passed
native frontend primary: 44 checks passed
native frontend core/parity/adversarial: 536 checks passed
native frontend driver: 72 checks passed
```

Coverage includes trusted-header shadow/copy/remap attacks, annotation and
signature spoofing, overloads, namespace alias success, unqualified/indirect
calls, templates, lambdas, macros, header sites, constexpr and unevaluated
contexts, token edges, CRLF/no-final-newline/UTF-8, multiline/two-call sites,
comments/strings, flag parity, source and included-header races, ABI-altering
macros, dependency escaping, target context, and cross-root co-linking.

## Fresh Release and Debug

Final Release tree: `/tmp/matcore-native-v1-final5-release.1R1Ecm`

```sh
cmake -S compiler -B /tmp/matcore-native-v1-final5-release.1R1Ecm -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-native-v1-final5-release.1R1Ecm -- -j2
ctest --test-dir /tmp/matcore-native-v1-final5-release.1R1Ecm \
  --output-on-failure -j1
```

Ninja built 19/19 steps. Result: 8/8 CTest tests passed in 61.55 seconds. The
integration matrix reported 63 pass, 0 fail, and 6 future capabilities
explicitly excluded from the pass count.

Final Debug tree: `/tmp/matcore-native-v1-final5-debug.YZDSYz`, configured
identically with `-DCMAKE_BUILD_TYPE=Debug`. Ninja built 19/19 steps and the
full suite passed 8/8 CTest tests in 62.94 seconds.

A strict `-Wall -Wextra -Wpedantic -Werror` build at the final implementation
head passed, followed by the full standalone CTest suite at 8/8.

Configuration-mode proofs also passed:

- bootstrap-only `/tmp/matcore-native-v1-bootstrap-only.NXg4Ku`: 3/3;
- native-only `/tmp/matcore-native-v1-native-only.d54uGa`: 4/4.

The bootstrap-only default invocation failed as required; explicit compatibility
mode succeeded.

## Sanitizer evidence

Sanitizer tree: `/tmp/matcore-native-v1-final5-sanitize.WdaSNP`

```text
CMAKE_CXX_FLAGS=
  -fsanitize=address,undefined
  -fno-sanitize=pointer-overflow
  -fno-omit-frame-pointer
```

With leak detection enabled, the supported focused set passed 4/4 in 7.99
seconds: native focused, native primary, native core, and runtime. The
separately instrumented generated program in
`/tmp/matcore-native-v1-final5-artifacts.NM2iTj` linked the sanitizer runtime
explicitly and printed `MDSLC CPU GEMM PASS` without a report. `nm` confirmed
ASan and UBSan instrumentation symbols.

Driver/package child links do not automatically inherit configure-only
sanitizer flags, so the full sanitizer CTest driver/package group is not counted
as passed. Pointer-overflow is excluded because explicit bootstrap mode uses
Ubuntu RapidJSON 1.1, whose compatibility implementation triggers that
instrumentation; address and remaining undefined-behavior checks stay enabled.

## Install, consumer, and artifact evidence

Final fresh prefix
`/tmp/matcore native v1 final5 direct.wK4ayb/install prefix` contains both
tools, public headers, the versioned runtime, and relocatable CMake package
files. The installed native suite passed 7 checks. Scans found no source/build
absolute path, Python, or nanobind leakage.

Final external build
`/tmp/matcore native v1 final5 direct.wK4ayb/consumer build` used
`find_package(MatcoreDSL REQUIRED)`, then configured, built, ran, and returned
to a no-op build. Touching the `.mdsl`, its user header, or the installed
runtime header independently rebuilt exactly the MDSLC object and final
executable. The space-bearing relocated prefix validates runtime/header/tool
discovery without an embedded development path.

Final artifact tree `/tmp/matcore-native-v1-final5-artifacts.3aSn01` retains:

```text
gemm_v0.host.cpp
gemm_v0.host-overlay.yaml
gemm_v0.matcore.json
gemm_v0.sites.h
gemm_v0.stubs.cpp
gemm_v0.backend.cpp
gemm_v0.host.o
gemm_v0.stubs.o
gemm_v0.backend.o
gemm_v0.o
gemm_v0
```

`file`, `readelf`, `nm -C`, and `ldd` verified an ordinary ELF64 relocatable
object, expected C++ site/backend symbols, the unresolved C runtime boundary
before link, an ordinary final PIE, and resolution through
`libmatcore_runtime.so.0`. Execution printed:

```text
host-before
MDSLC CPU GEMM PASS
```

Verbose driver evidence invoked the native extractor and ordinary Clang
compilations; no AST-dump invocation or bootstrap marker appeared.

A second build at `/tmp/matcore-native-v1-final5-repeat.tmb1BX` produced
byte-identical JSON, host, sites, stubs, and backend files. The deterministic
JSON SHA-256 is
`afd693d72e2574d27aae53a8ccd50e975404bee6cf19425cc05f64739016d480`.

## Legacy and device checks

```sh
pytest -q \
  tests/test_frontend_contract.py \
  tests/test_fusion_analysis.py \
  tests/test_fusion_contracts.py \
  tests/test_tracer.py
```

Result: 26 passed; 3 failed solely at the missing legacy `_matcore_native`
import boundary. Fresh root CMake at
`/tmp/matcore-native-v1-legacy-cmake.oUU93d` failed at the already documented
MLIR requirement mismatch: requested 18.1.3 versus available 22.1.2. Neither is
represented as a standalone native frontend failure or a green legacy build.

CPU GEMM is runtime-validated. NVIDIA/CUDA 13.3, AMD/ROCm 7.1, and `aie2p` NPU
were detected only; no accelerator MDSLC backend was attempted.

## Integration verdict

The native default meets the declared Clang frontend, authenticated source,
verified IR, normal artifact, installation, and CPU execution gates. The
integration verdict is:

**Architecture proof passed for the standalone native CPU frontend/runtime
vertical slice.**
