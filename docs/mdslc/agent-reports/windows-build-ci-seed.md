# Windows build, package, and CI seed

Date: 2026-07-22

Integration base: `091d74072a710389b4a8e9d51f696ad9773021e6`

Lane commits:

- `f0c3532` — `build(windows): seed clang-cl package and hosted validation`
- `67eacea` — `fix(windows): package generated MDSLC objects as archives`

This report records a portability and hosted-validation seed. It does **not**
claim a successful Windows build or runtime execution. The workflow must run on
native Windows after the driver, process, runtime, and platform lanes are
integrated; until that happens every Windows status remains unvalidated.

## Owned scope

This lane changed only standalone CMake/build policy, package/consumer probes,
Windows-specific validation scripts, and the Windows GitHub Actions workflow.
It did not edit driver, frontend, runtime, planner, kernel, or platform C++.

The implementation provides:

- a centralized compiler-frontend policy that distinguishes clang-cl/MSVC ABI
  flags from the GNU Clang command-line;
- supported Clang CMake-target discovery through `clang-cpp` where present or
  audited Tooling/ASTMatcher/Rewrite/Frontend component targets otherwise;
- exact LLVM/Clang `21.1.8` equality and required-header checks;
- Windows DLL/import-library output and installation conventions for the stable
  runtime C ABI;
- Windows OpenBLAS disabled by default and rejected when explicitly enabled,
  because no authenticated Windows CBLAS provider has been selected;
- `.mdsl` consumer generation as a normal static `.lib` containing the host,
  stub, and backend COFF objects. No ELF-style partial link is emulated and no
  archive is mislabeled as one `.obj`;
- clang-cl compile/dependency options, normal `.lib` final linking, and
  relocated install/consumer probes under whitespace and Unicode paths;
- strict C17 DLL/import-library ABI validation with `llvm-readobj`;
- an explicit native-only Windows frontend test and a COFF archive/PE pipeline
  test replacing only the ELF/rpath/POSIX-only Linux integration gates;
- a fail-closed Windows distribution validator for the required executables,
  runtime DLL/import library, C exports, imported LLVM DLL closure, platform
  diagnostics, package path leakage, and exact AVX2/AVX-512 microkernel symbols;
- an authenticated `windows-2025` workflow that builds Release, runs the full
  applicable CTest surface, performs a focused Debug run, installs, reruns the
  external consumer, emits an artifact report, and uploads a generated ZIP.

`matcore-bench.exe` is installed by `compiler/tools/matcore-bench/CMakeLists.txt`
and is a mandatory input to both the installed consumer and Windows
distribution validator. The root install block does not duplicate that rule.

## Reproducible third-party inputs

The workflow does not trust the runner's preinstalled LLVM version. It obtains
one official LLVM project development archive and authenticates it before use:

- target: `x86_64-pc-windows-msvc`
- LLVM/Clang: `21.1.8`
- URL: `https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/clang%2Bllvm-21.1.8-x86_64-pc-windows-msvc.tar.xz`
- bytes: `942572476`
- SHA-256: `749d22f565fcd5718dbed06512572d0e5353b502c03fe1f7f17ee8b8aca21a47`

RapidJSON is acquired from one immutable upstream commit archive:

- upstream version: `1.1.0`
- commit: `f54b0e47a08782a6131cc3d60f94d038fa6e0a51`
- URL: `https://github.com/Tencent/rapidjson/archive/f54b0e47a08782a6131cc3d60f94d038fa6e0a51.tar.gz`
- bytes: `1020103`
- SHA-256: `4a76453d36770c9628d7d175a2e9baccbfbd2169ced44f0cb72e86c5f5f2f7cd`

The workflow verifies the LLVM compiler, target triple, `llvm-config` version,
Tooling/ASTMatcher/Rewriter/Lexer/AST/Frontend headers, and LLVM/Clang CMake
package files. Visual Studio Build Tools and the Windows SDK are located via
`vswhere`; the environment comes from `VsDevCmd.bat`; `lld-link` and
`llvm-lib` are selected explicitly. OpenBLAS is intentionally off.

## Test-registration boundary

The compatibility AST-JSON bootstrap frontend remains off in the Windows
distribution. Native/bootstrap parity continues to run on Linux; Windows does
not silently re-enable bootstrap.

Two Linux-only tests are excluded on Windows for explicit reasons:

- `driver.native.selection` models ELF/rpath relocation and POSIX executable
  shims;
- `integration.validation_matrix` authenticates ELF relocatable objects,
  SONAME/rpath, `readelf`, `nm`, and the Linux partial-link contract.

Their Windows replacement coverage is registered as:

- `frontend.native.windows`: native clang-cl LibTooling extraction, typed IR
  v1, canonical declaration identity, source locations, and untrusted-header
  rejection;
- `integration.windows.native_pipeline`: `.mdsl` to COFF object set/static
  `.lib`, save-temp object inventory, normal clang-cl PE link, runtime DLL
  discovery, and numerical GEMM execution.

The remaining IR, capability, topology, runtime, planner, benchmark, package,
and installed-consumer tests stay registered. Linux exact-ISA artifact tests
remain Linux gates; the Windows distribution validator supplies exact-symbol
`llvm-nm`/`llvm-objdump` coverage rather than pretending GNU/ELF inspection is
portable.

## Local Linux regression evidence

The code commits were validated from a clean worktree with:

```text
cmake -S compiler \
  -B /home/hamza-usta/.codex-builds/matcore-win-seed-clean-release \
  -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=ON \
  -DMDSLC_REQUIRE_OPENBLAS=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /home/hamza-usta/.codex-builds/matcore-win-seed-clean-release --parallel 2
ctest --test-dir /home/hamza-usta/.codex-builds/matcore-win-seed-clean-release \
  --output-on-failure -j1
```

Results:

- configure: passed; coherent `clang-cpp` link model; OpenBLAS `0.3.32`;
- build: `89/89` actions passed;
- CTest: `42/42` passed in `67.32` seconds;
- relocated install/consumer, strict C17 ABI, benchmark provenance, native
  frontend, end-to-end GEMM, exact AVX2/AVX-512 artifacts, planner, runtime,
  platform, and repository tests were all included in that count.

A second feature-selection build matching Windows's frontend/provider policy
also passed configuration and all `87/87` build actions with:

```text
-DMDSLC_ENABLE_NATIVE_FRONTEND=ON
-DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=OFF
-DMDSLC_ENABLE_OPENBLAS=OFF
```

It registered 38 Linux-applicable tests. This is configuration evidence only;
it is not a substitute for native clang-cl execution.

Static validation:

- all changed Python scripts passed `python3 -m py_compile`;
- `.github/workflows/mdslc-windows.yml` passed actionlint `1.7.7`;
- `git diff --check` passed;
- `bash tests/check_repository_hygiene.sh` passed.

An earlier pre-commit CTest run reported `40/42`; its only two failures were
the benchmark's deliberate `source_worktree_dirty` provenance guard. The
post-commit clean run above resolved both without weakening the guard.

## Mandatory integration handoffs

The current configured header still encodes
`MDSLC_CLANGXX_EXECUTABLE` as an absolute build-time path. On a hosted Windows
runner that path is beneath `RUNNER_TEMP`, so it is not a relocatable installed
driver contract. The driver/platform lane must replace the installed behavior
with authenticated adjacent/PATH discovery of `clang-cl.exe` while retaining
exact version/provider checks. The distribution validator deliberately scans
installed binaries and CMake files for the source root, build root, and LLVM
extraction root; it will fail rather than allow this leakage.

The existing C++ process, executable-discovery, file-identity, and temporary
workspace implementation was POSIX-specific at this lane's base. The platform
and driver lanes must complete those ports before the hosted pipeline can pass.
Similarly, Windows CPU capability/topology/affinity and runtime code must be
integrated before runtime tests can establish support.

The exact imported-target shape and LLVM DLL closure of the official Windows
archive have not yet been exercised by CMake on native Windows. The build
supports `clang-cpp` or the audited component-target set and the workflow will
fail closed if neither is provided.

## Current Windows status

- Windows frontend: **not executed / unvalidated**
- Windows runtime DLL: **not executed / unvalidated**
- Windows CPU planner: **not executed / unvalidated**
- Windows native variants: **not runtime-validated**
- Windows OpenBLAS: **intentionally omitted**
- Windows parallel runtime: **not executed / unvalidated**
- Windows NUMA: **not executed; no physical claim**
- Windows package and external consumer: **not executed / unvalidated**
- Windows distribution ZIP: **workflow defined; not produced**
- GitHub-hosted workflow: **not run from this unpushed agent branch**

No Windows performance, ISA-runtime, NUMA, packaging, or support claim is made
by this seed.

## CMake integration addendum

Follow-up date: 2026-07-22

Follow-up base: `3741511b72eea256cc759049e89f99a8fd4b57f4`

Follow-up implementation: `8fb3a49ca787fc4d0fd3bd956509ff38a2193dc1`

After the portable support and topology lanes were integrated, a CMake-only
follow-up closed the remaining build-graph gap:

- `lib/support` is added before the IR, frontend, driver, extractor, runtime,
  planner, and benchmark consumers;
- both `mdslc++` and `matcore-extract` link
  `MatcoreDSL::PlatformSupportV1` explicitly;
- the standalone frontend test project also adds support before its extractor;
- `support.platform.v1` is now part of the root CTest surface;
- support, platform, and benchmark warning policy now flows through
  `MatcoreDSLCompilerOptions.cmake`, whose selection is based on the MSVC ABI
  frontend variant and therefore handles clang-cl without GNU flag leakage;
- standalone support and benchmark configurations load the same helper when a
  parent project has not already provided it.

The only direct MSVC/frontend-variant conditional remaining in the reviewed
CMake surface is `MatcoreDSLCompile.cmake`'s installed-consumer artifact mode.
That conditional is intentional: it selects clang-cl arguments and the normal
Windows static `.lib` output. It still invokes:

```text
/c <source.mdsl> -o <generated-static-library>.lib
```

and retains constituent `.obj` files only for save-temps validation. No
partial-link emulation or archive-as-object labeling was introduced.

Fresh clean Linux Release validation after the follow-up used the same exact
Clang/LLVM 21.1.8 and required OpenBLAS 0.3.32 configuration documented above:

- configure: passed;
- build: `95/95` actions passed;
- CTest: `43/43` passed in `63.86` seconds;
- `support.platform.v1` passed as test 9;
- relocated consumer, package ABI, frontend, integration, CPU runtime,
  planner, benchmark, and ISA artifact tests remained green.

A second fresh build with native frontend enabled, bootstrap disabled, and
OpenBLAS disabled completed all `93/93` actions. Its focused
`support.platform.v1` execution passed. Repository hygiene and `git diff
--check` also passed.

This addendum is Linux regression evidence for the integrated CMake graph. It
does not change the unvalidated Windows status above; only a native hosted
Windows run can do that.
