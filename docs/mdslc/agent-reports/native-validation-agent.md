# Native frontend parity and validation lane

Date: 2026-07-19

Branch: `mdslc/native-validation-v1`

Base: `3e3fa5b2d1990e1c37870f8b2096fbda6128716b`

## Ownership and changes

This lane changed only:

- `compiler/tests/native_validation/**`
- this report

It did not change compiler production code, shared CMake files, package files,
legacy sources, or existing general validation scripts.

The new Python harness has four separately registerable suites:

- `core`: native-default selection, explicit frontend selection, semantic IR
  parity, generated-artifact parity, deterministic SourceManager ranges, and
  native-only adversarial authentication tests;
- `installed`: installed-prefix header discovery, shadow resistance, copied
  header rejection, and checkout-path leakage inspection;
- `unavailable`: native-disabled extractor and driver behavior, including the
  explicit bootstrap compatibility path and the prohibition on silent default
  fallback;
- `driver`: extraction/compile flag forwarding, frontend provenance, and a
  deterministic source-change shim between extraction and later compilation.

Parity normalization removes only the top-level `producer` field. Every other
IR field and every generated host/sites/stubs/backend byte is compared. A
deliberate semantic mutation self-test proves that the comparator detects
non-producer differences.

The adversarial matrix covers all 30 native-frontend cases requested by the
milestone. Trusted annotation and signature branches are authenticated rather
than inferred: exact canonical redeclarations exercise mutated/conflicting
`AnnotateAttr` payloads, while temporary tool prefixes mutate the extractor's
own resolved trusted header to exercise missing annotation and wrong canonical
signature handling.

## Validation evidence

The native extractor came from the native frontend lane:

```text
/home/hamza-usta/MatcoreDSL-wt-native-front-v1/build-native-lane/bin/matcore-extract
```

Core differential/adversarial suite:

```sh
python3 compiler/tests/native_validation/run_native_validation.py \
  --suite core \
  --extractor /home/hamza-usta/MatcoreDSL-wt-native-front-v1/build-native-lane/bin/matcore-extract \
  --clang /usr/bin/clang++-21
```

Result: `native frontend core: 430 checks passed`.

A fresh temporary installation was created with:

```sh
cmake --install \
  /home/hamza-usta/MatcoreDSL-wt-native-front-v1/build-native-lane \
  --prefix /tmp/matcore-native-install-3h7SCe
```

Installed suite:

```sh
python3 compiler/tests/native_validation/run_native_validation.py \
  --suite installed \
  --extractor /tmp/matcore-native-install-3h7SCe/bin/matcore-extract \
  --clang /usr/bin/clang++-21
```

Result: `native frontend installed: 7 checks passed`.

A native-disabled build was configured and compiled with:

```sh
cmake -S /home/hamza-usta/MatcoreDSL-wt-native-front-v1/compiler \
  -B /tmp/matcore-native-off-lqmOwa -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-native-off-lqmOwa \
  --target matcore-extract -- -j2
```

The unavailable suite used that extractor plus the driver lane binary in a
temporary coherent tool layout. Result:

```text
native frontend unavailable: 11 checks passed
```

The driver suite combined the native frontend lane extractor and driver lane
binary in a temporary tool layout. Result:

```text
native frontend driver: 17 checks passed
```

Static lane checks:

```sh
ruff format --check compiler/tests/native_validation/run_native_validation.py
ruff check compiler/tests/native_validation/run_native_validation.py
python3 -m py_compile \
  compiler/tests/native_validation/run_native_validation.py
git diff --check
```

All passed. Every committed `.mdsl` fixture also passed Clang 21 syntax-only
parsing when supplied its deliberate flags and trusted include path. The
missing-flag fixture intentionally fails without
`-DMDSLC_NATIVE_VALIDATION_FLAG=17`.

## Integration handoff

The integration owner must register the four suites in CMake using the correct
artifacts. In particular, `unavailable` requires a distinct native-disabled
build and both `unavailable` and `driver` require `--driver`. Missing required
suite arguments fail rather than silently skipping coverage.

No Release/Debug/sanitizer/full-package claims are made by this lane; those
remain integration-level gates after the frontend and driver commits are
combined.
