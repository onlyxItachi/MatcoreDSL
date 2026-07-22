# Milestone 5 package and CI hardening report

## Scope and ownership

Branch: `mdslc/m5-package-ci-hardening`

Base: `37dc93838f0d8c953599560dece31ca407bb9902`

This lane owned the strict installed C ABI probe under `compiler/tests/package`,
its top-level CTest registration, and `.github/workflows/mdslc-native.yml`. It
did not modify benchmark implementation, `cpu_runtime.cpp`, or planner/runtime
core files.

## Changes

- Added a relocated-install probe compiled as strict C17 with
  `-pedantic-errors -Wall -Wextra -Werror`.
- Compared the installed header's complete `MATCORE_RUNTIME_API` export set to
  the probe's direct call set, then verified that every call remains an
  undefined dynamic reference in the linked executable.
- Compiled, linked, and executed positive F32, capability-v2, context-v1,
  context-plan-v2/v3, BF16/F32, and I8/I32 public C ABI paths.
- Directly called every legacy, workspace, prepacked-B, context, planning,
  capability, and typed-reference export. The tested installed header exposed
  exactly 15 public functions.
- Replaced stale standalone-workflow push branch filters with path-scoped
  validation on all branches.
- Added two hosted Release configurations: coherent OpenBLAS required and
  OpenBLAS explicitly disabled. The workflow authenticates the configure
  decision and resulting dynamic linkage before running CTest.

## Validation evidence

OpenBLAS-required Release:

```text
cmake ... -DMDSLC_ENABLE_OPENBLAS=ON -DMDSLC_REQUIRE_OPENBLAS=ON
MDSLC OpenBLAS variant: 0.3.32 via pkg-config
cmake --build ... --parallel 2
88/88 build steps completed
ctest -R '^package\.installed_c17_abi$' -V
1/1 passed; compile/link/run PASS; authenticated 15 public exports
```

Focused installed/runtime checks:

```text
package.installed_c17_abi             passed
runtime.c_abi.typed_reference_v1      passed
runtime.cpu.openblas_adapter          passed
runtime.c_abi.compatibility_v1        passed
runtime.c_abi.public_context_v1       passed
```

`ldd` authenticated the required build's provider:

```text
libopenblas.so.0 => /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0
```

OpenBLAS-disabled Release:

```text
cmake ... -DMDSLC_ENABLE_OPENBLAS=OFF -DMDSLC_REQUIRE_OPENBLAS=OFF
MDSLC OpenBLAS variant: unavailable (optional)
cmake --build ... --parallel 2
88/88 build steps completed
focused package/runtime tests: 5/5 passed
ldd: no OpenBLAS DT_NEEDED entry
```

The OpenBLAS-required full standalone run excluding the known installed
consumer failure passed 38 of 40 tests. The two failures were
`benchmark.cpu.contract` and `benchmark.cpu.cli_json`; both report that the
forced reference implementation is not runtime-validated. The installed
consumer fails for the same benchmark-evidence reason. This behavior exists at
the integration base and belongs to the separately active benchmark-evidence
lane; no benchmark workaround was added here.

`git diff --check` passed. `actionlint` and a local GitHub Actions runner were
not installed, so the workflow has not yet been hosted-executed in this lane.

## Commits

- `cd8a7e1` `test(package): authenticate installed C ABI from C17`
- `720382c` `ci(mdslc): validate linked and absent OpenBLAS`

## Verdict

The installed strict-C ABI and dual OpenBLAS package configuration checks pass.
Final hosted CI and the full consumer result remain integration gates after the
benchmark-evidence lane lands.
