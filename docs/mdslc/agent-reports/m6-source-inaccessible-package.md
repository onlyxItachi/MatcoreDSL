# Milestone 6 source-inaccessible package regression

Lane ownership:

- `compiler/tests/package/run_source_inaccessible_consumer.py`
- `compiler/tests/package/run_source_inaccessible_consumer_safety_tests.py`
- the `package.installed_source_inaccessible` and safety-test registrations in
  `compiler/CMakeLists.txt`

## Contract

The Linux-focused regression clones the exact local `HEAD` without Git
hardlinks and verifies that the clone is clean. It configures a fresh Release
standalone compiler with the native Clang frontend and OpenBLAS disabled,
builds and installs it, relocates the prefix through a whitespace/Unicode path,
and copies the external consumer fixture outside the producer source.

Before consumer configuration, the test:

1. scans the install and copied consumer for the real checkout, disposable
   source, disposable build, and staging-prefix paths;
2. removes the disposable producer source and build trees;
3. configures, builds, and executes the consumer using only the relocated
   installed prefix and copied consumer source;
4. authenticates the consumer output; and
5. repeats the path-leak scan over the install and consumer build.

The test root must be the exact
`<configured-CMake-build>/tests/installed-source-inaccessible` path. The
configured build root must have a `CMakeCache.txt`; parent components, the
build root, and test path may not escape through `..` or symlinks. An existing
test root is removed only when its versioned sentinel authenticates the exact
canonical build and test-root paths, and only through Python's symlink-safe
recursive-removal implementation. It may not equal or contain the real
checkout. A separate lightweight adversarial CTest preserves mismatched,
symlink-target, unsentinelled, and protected-checkout victim files while
proving each request is rejected.

The nested CTest is Linux-only, serial, and has a ten-minute timeout. Existing
Windows package coverage is unchanged.

## Validation

The first direct run failed during the nested RapidJSON configure probe because
the test resolved `/usr/bin/clang++-21` to the `clang` basename. Clang selects
C++ driver behavior from `argv[0]`. The test now preserves the caller-visible
compiler basename, matching the established installed-consumer rule.

Focused validation after the benchmark measurement window closed:

- clean exact-HEAD clone smoke: passed;
- nested Release build with native Clang 21.1.8 and OpenBLAS disabled:
  51/51 build actions passed;
- relocated install inventory and binary/text path-leak scan: passed;
- disposable producer source and build removal before consumer configure:
  passed;
- copied external `.mdsl` consumer configure/build/execute: passed and printed
  `consumer-before`, `consumer-header=1`, and `consumer-pass`;
- review hardening adversarial cases for a mismatched root, symlinked test
  root, missing sentinel, `..` spelling, symlinked build root, and a test root
  containing the protected checkout: all rejected without mutating their
  victim markers;
- registered CTests:
  `package.installed_source_inaccessible` and
  `package.installed_source_inaccessible_safety` passed 2/2 in 23.92 seconds
  against source checkpoint
  `eeb160a068838890d130511bfdc177257059143e`;
- CTest registration inventory: 48 tests including both package gates;
- Python syntax, `git diff --check`, and repository hygiene: passed.

No production source, public ABI, Windows path, benchmark data, or planner
behavior changed.
