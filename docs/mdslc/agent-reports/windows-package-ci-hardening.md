# Windows Package and CI Hardening Follow-up

Date: 2026-07-22

Lane: Windows x64 package, consumer, driver-boundary, and CI validation

Base: `c1b316eda640d5d9a49b6e57d417304d3f59101a`

## Scope and ownership

This lane changed only the standalone compiler CMake configuration, installed
consumer/package validation, Windows-native validation scripts, and the
Windows GitHub Actions workflow. It did not change the compiler driver,
frontend, runtime, CPU kernels, IR, or planner implementation.

## Changes

- Kept `MDSLC_CLANGXX_EXECUTABLE` absolute and version-authenticated at CMake
  configure time, but configured installed Windows tools with the relocatable
  runtime name `clang-cl.exe`. Linux retains its audited absolute driver.
- Replaced clang-cl escape-form dependency flags with the public driver
  contract `-MD -MF <combined-depfile>`. The existing installed-consumer test
  proves one generated artifact, one combined depfile, source/header/runtime
  header regeneration, and a final no-op rebuild.
- Split Windows installed-consumer environments into an isolated runtime PATH
  and an intentional compiler PATH. `LLVM_ROOT` is removed rather than being
  inherited accidentally.
- Made Windows distribution validation recursively traverse the complete PE
  import graph. Every non-system DLL must be present in the install tree, and
  every third-party bundled DLL must be byte-identical to its source in the
  SHA-256-authenticated official LLVM archive.
- Made CI materialize the same transitive non-system dependency closure from
  the authenticated archive. It records provider, archive hash, file hashes,
  and available LLVM notices alongside the distribution licenses.
- Kept runner-specific environment and artifact evidence outside the ZIP. The
  JSON report is uploaded beside the ZIP as a separate workflow artifact file.
- Preserved the existing Windows native frontend matrix of three positive and
  seventeen negative cases. Extended the driver pipeline with three positive
  paths (default native, explicit native, and shell-metacharacter single-argv)
  and eleven fail-closed guards: `/LD`, direct `@response`, `/link` with `/c`,
  wrong `.lib` output, input/output overwrite, CUDA without fallback, `/TC`,
  unknown frontend, unavailable bootstrap, production test-prefix override,
  and source mutation between rewrite and generated compilation.
- Added a focused clang-cl x64 AddressSanitizer lane. `/fsanitize=address /Oy-`
  is forwarded through `mdslc++` so the generated host, stub, and backend
  objects are instrumented; `/fsanitize=address` is also used for the ordinary
  final PE link. The workflow locates the authenticated ASan runtime DLL before
  executing both the focused runtime test and generated GEMM pipeline. No
  Windows UBSan claim is made.

## Commits

1. `6a2ca1a` — `build(windows): keep installed compiler discovery relocatable`
2. `b567e2e` — `test(windows): harden driver and distribution validation`

## Validation

Fresh Linux Release configuration used Clang/LLVM 21.1.8 and required the
coherent OpenBLAS 0.3.32 provider:

```text
cmake -S compiler -B /tmp/matcore-win-hardening-release.JUJd5Z -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_REQUIRE_OPENBLAS=ON
cmake --build /tmp/matcore-win-hardening-release.JUJd5Z --parallel 2
ctest --test-dir /tmp/matcore-win-hardening-release.JUJd5Z \
  --output-on-failure -j1
```

Results:

- configure: passed;
- compilation and links: 95/95 steps passed;
- CTest: 43/43 passed;
- installed consumer: relocated configure/build/run, source edit, included
  header edit, installed `runtime_c.h` edit, and no-op rebuild passed;
- repository hygiene: passed;
- Python validation scripts: `py_compile` passed;
- workflow: actionlint 1.7.7 passed;
- `git diff --check`: passed.

The first link attempt used `/tmp`, whose 7.2 GiB tmpfs had only 1.5 GiB free,
and `ld.bfd` reported `No space left on device`. Re-running with `TMPDIR` on the
345 GiB-free root filesystem completed successfully. The first CTest run was
made before committing and correctly triggered two benchmark dirty-provenance
guards; rebuilding after the focused commits produced the clean 43/43 result.

## Status and remaining gate

The Linux regression evidence is complete for this lane. The Windows workflow
definition is statically validated but has not yet executed on GitHub-hosted
Windows hardware in this lane. Therefore Windows Release, Debug, ASan, COFF/PE,
DLL closure, installed consumer, and ZIP status remain **not yet hosted-run**,
not passed. The next integration step is to run the workflow, reconcile any
driver diagnostic spelling with these fail-closed assertions, and fix only
reproducible Windows failures before making a Windows support claim.
