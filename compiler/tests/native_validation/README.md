# Native frontend validation

`run_native_validation.py` is the focused differential and adversarial suite for
the Clang LibTooling frontend. It deliberately lives outside the bootstrap
frontend suite so native-only hardening does not weaken or redefine the
bootstrap compatibility contract.

The core suite requires a `matcore-extract` binary with these modes:

```text
--frontend=native             (also the default)
--frontend=ast-json-bootstrap (explicit compatibility mode)
```

Run it from the repository root:

```sh
python3 compiler/tests/native_validation/run_native_validation.py \
  --suite core \
  --extractor /absolute/path/to/build/bin/matcore-extract
```

Additional suites are intentionally separate because they require distinct
artifacts:

```sh
# Installed-prefix trusted-header discovery.
python3 compiler/tests/native_validation/run_native_validation.py \
  --suite installed --extractor /prefix/bin/matcore-extract

# A build configured with MDSLC_ENABLE_NATIVE_FRONTEND=OFF.
python3 compiler/tests/native_validation/run_native_validation.py \
  --suite unavailable --extractor /bootstrap-only/bin/matcore-extract \
  --driver /bootstrap-only/bin/mdslc++

# Driver forwarding and source-snapshot race checks.
python3 compiler/tests/native_validation/run_native_validation.py \
  --suite driver --extractor /build/bin/matcore-extract \
  --driver /build/bin/mdslc++
```

The CMake integration owner must register all suites with the appropriate
build/install fixtures. A missing required argument is an error, not a skip.
