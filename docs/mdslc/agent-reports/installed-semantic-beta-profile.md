# Installed semantic beta profile

- Date: 2026-08-11
- Implementation commit: `a97f8bc`
- Diagnostic-test hardening: `c4f459d`
- Integrated validation head: `6fb8731`
- Scope: compiler default selection, installed CMake package capability,
  relocated consumer, and source/build-inaccessible package validation
- Verdict: accepted for the Linux CPU beta profile; no public API or ABI
  freeze is claimed

## Product boundary

The source-tree default remains deliberately conservative:

```text
MDSLC_ENABLE_MATCORE_MLIR=OFF
MDSLC_DEFAULT_SEMANTIC_PIPELINE=capture-v0
```

`MDSLC_DEFAULT_SEMANTIC_PIPELINE` is a validated cache string with exactly two
values: `capture-v0` and `matcore-mlir`. Selecting `matcore-mlir` as the
configured default requires `MDSLC_ENABLE_MATCORE_MLIR=ON`; the invalid and
unavailable combinations stop during CMake configuration.

The Linux MLIR-enabled CPU beta profile deliberately configures:

```text
MDSLC_ENABLE_MATCORE_MLIR=ON
MDSLC_DEFAULT_SEMANTIC_PIPELINE=matcore-mlir
```

The configured choice is compiled into `mdslc++` and is used when the caller
does not supply `--semantic-pipeline`. An explicit selector still overrides
the configured default. Windows and MLIR-disabled configurations remain
explicitly `capture-v0`; this work does not claim Windows Matcore MLIR support.

The `matcore-mlir` route remains restricted to the authenticated native
frontend. In a package whose default is `matcore-mlir`, compatibility use of
the AST-JSON bootstrap frontend must explicitly pair it with `capture-v0`.
Bootstrap plus `matcore-mlir` is rejected without fallback.

## Installed package contract

`MatcoreDSLConfig.cmake` publishes two queryable values:

- `MatcoreDSL_MATCORE_MLIR_AVAILABLE`, normalized to exact `ON` or `OFF`
  regardless of the caller's original CMake boolean spelling; and
- `MatcoreDSL_DEFAULT_SEMANTIC_PIPELINE`, equal to the configured package
  default.

`matcoredsl_add_executable` accepts an optional `SEMANTIC_PIPELINE`. Omission
uses the package default. Invalid values, a request for an unavailable
`matcore-mlir`, or a `matcore-mlir`/bootstrap pairing fail during consumer
CMake configuration. The helper materializes the selected pipeline as an
explicit `mdslc++` argument, so package behavior cannot drift with the calling
environment.

When MLIR is enabled, the package installs `matcore-mlir` as a leaf executable.
Its private MLIR implementation libraries are not exported to consumers. When
MLIR is disabled, validation requires the tool to be absent as well as the
capability value to be `OFF`.

## End-to-end authentication

The configured-default driver test does more than check for a saved semantic
sidecar. For a `matcore-mlir` default it requires:

- deterministic semantic MLIR equal to the explicit semantic route;
- backend producer marker
  `Producer: Matcore MLIR CPU runtime-dispatch lowering v1`; and
- the stable `matcore_runtime_gemm_f32_v0` runtime call.

The `capture-v0` default and explicit capture override require no semantic
sidecar, no Matcore MLIR backend producer marker, and the same stable runtime
dispatch. This prevents an inspection-only semantic file from being mistaken
for executed semantic lowering.

The source-inaccessible gate clones the exact committed HEAD, builds and
installs it, copies only exact input/golden fixtures, relocates the install,
and deletes the producer source and build trees. It then:

- executes installed `matcore-mlir` twice and requires deterministic exact
  golden output when capability is `ON`;
- configures, builds, and runs a real external `.mdsl` consumer through the
  package default;
- builds and runs the opposite explicit pipeline when available;
- rejects invalid or unavailable semantic requests during configuration;
- rejects bootstrap with an inherited `matcore-mlir` default; and
- builds and runs the explicit bootstrap plus `capture-v0` compatibility pair.

## Validation evidence

All compilation used Clang/LLVM 21.1.8 and exact MLIR 21.1.8, with
`nice -n 10` and at most two build jobs.

### MLIR ON, default `matcore-mlir`

```text
cmake -S compiler -B <on-build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_MATCORE_MLIR=ON \
  -DMDSLC_DEFAULT_SEMANTIC_PIPELINE=matcore-mlir \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DMLIR_DIR=<exact-mlir-21-dir> -DBUILD_TESTING=ON
cmake --build <on-build> --target \
  mdslc_driver matcore-mlir matcore-plan matcore-bench -- -j2
```

- focused build: 44/44 actions passed;
- `frontend.native.driver_pipeline`, `driver.native.selection`, and
  `integration.semantic_mlir_cpu_pipeline`: 3/3 passed;
- `consumer.installed`: 1/1 passed after exact clean-HEAD provenance refresh;
- `package.installed_source_inaccessible`: 1/1 passed.

### MLIR OFF, default `capture-v0`

The corresponding Release configuration used
`MDSLC_ENABLE_MATCORE_MLIR=OFF` and
`MDSLC_DEFAULT_SEMANTIC_PIPELINE=capture-v0`.

- focused build: 35/35 actions passed;
- `frontend.native.driver_pipeline`, `driver.native.selection`, and
  `integration.semantic_mlir_unavailable`: 3/3 passed;
- `consumer.installed`: 1/1 passed;
- `package.installed_source_inaccessible`: 1/1 passed.

Additional configuration gates passed:

- an unknown default value was rejected;
- `matcore-mlir` default with MLIR disabled was rejected; and
- configuring `MDSLC_ENABLE_MATCORE_MLIR=TRUE` exported the normalized package
  capability as exact `ON`.

Python syntax validation passed for all five modified test drivers, and
`git diff --check` passed before each implementation commit.

## Resolved validation issue

The first exact-commit source-inaccessible run built and installed 68/68
actions, then its negative test compared a full diagnostic that CMake had
line-wrapped. Commit `c4f459d` changed the test to match stable diagnostic
fragments without weakening the required failure. The exact-commit rerun
passed.

The first parent-tree installed-consumer attempt was intentionally stopped by
the benchmark provenance guard because unrelated workflow edits were still
uncommitted. After workflow commit `6fb8731` made the tree clean,
`matcore-bench` provenance was refreshed and both installed-consumer profiles
passed. No guard was bypassed.

## Review and limitations

An independent static rereview found no unresolved high- or medium-severity
semantic, ABI, package, or installed-default issue in `a97f8bc` after the
review feedback was addressed.

This profile proves installed Linux CPU semantic execution through the
existing native runtime-dispatch lowering. It does not claim Windows MLIR,
Linalg/Vector code generation, recovered-loop execution, a public ABI freeze,
or that every build of the project defaults to Matcore MLIR. Hosted workflow
evidence remains a separate integration gate.
