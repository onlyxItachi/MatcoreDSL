# Native driver contract tests

Run the focused driver test after building the standalone project:

```sh
python3 compiler/tests/driver_native/run_driver_tests.py \
  --driver build-mdslc/bin/mdslc++
```

The test copies the driver into a prefix whose path contains spaces, installs a
sibling extractor shim, and verifies native-default selection, explicit native
and bootstrap selection, exact child-status propagation, absence of fallback,
argv-safe compiler-flag forwarding, self-relative trusted-header discovery, and
source-mutation rejection before object publication. It also performs a normal
final link and executes the result from a relocated prefix containing spaces
and a comma, proving that runtime search paths stay single argv values.

The production configuration rejects `--tool-prefix-for-testing`. A build
configured deliberately with `-DMDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE=ON` can
exercise that override by adding `--test-prefix-override=enabled` to the test
command.

Configured relative GNU install directories can be exercised with matching
`--bindir`, `--includedir`, and `--libdir` arguments. The driver derives the
prefix from its executable location; it does not embed an install prefix.
