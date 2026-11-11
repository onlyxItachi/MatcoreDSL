# Issue #110: installed-test scratch ownership v1

Date: 2026-11-12. Base: `1c1221e89272eb2ad6ff058b8e26de4653f3a93d`.
Branch: `fix/mdslc-issue110-test-root-ownership-v1`.
Scope: [issue #110](https://github.com/onlyxItachi/MatcoreDSL/issues/110) only.

## Implemented contract

The installed CMake consumer and strict C17 ABI runners no longer resolve and
recursively remove `--test-root`. Both call the shared
`compiler/tests/installed_test_scratch.py` guard before writing test artifacts.
The requested root is now an owned container; each invocation exclusively
creates a fresh `run-*` child. Neither runner nor the helper deletes prior runs.

Before creating any parent, container, marker, or run directory, the guard:

- Checks lexical spellings, rejects `..` and ambiguous Windows drive-relative
  spellings, and inspects every existing build/root component with `lstat`.
- Refuses symlinks (including dangling links), Windows reparse points/junctions,
  and non-directory path components. The build requires a regular, non-link
  `CMakeCache.txt`.
- Refuses filesystem roots and roots equal to or containing the producer
  build, source checkout, caller-supplied consumer source, or home directory.
  It also refuses source descendants, except the registered scratch child
  within a producer build situated inside the checkout.
- Allows only the corresponding `build/tests/installed-consumer` or
  `build/tests/installed-c17-abi` child inside the build. Separately located
  external containers remain allowed, preserving the Windows workflow's
  external consumer path and its whitespace/Unicode spelling.
- Refuses all unowned existing containers, even empty ones. The bounded
  regular non-link marker must exactly identify schema/version, runner,
  producer build, and container path. Wrong, malformed, oversized, linked, or
  directory markers are rejected. Only a newly created container receives a
  marker; existing data is never automatically adopted.

Both runners suppress import bytecode writes before importing the guard, so
loading it does not mutate the source checkout on rejected invocations.
The explicit `--windows-import-report-out` path remains unchanged; the default
report location follows the fresh per-run directory.

This is deliberately not a concurrent-filesystem sandbox or cryptographic
ownership proof. The caller must keep the build, sources and scratch ancestry
under stable control during execution. Reparse rejection is fail-closed; there
is no Windows fallback that follows a link or weakens deletion safety.

## Retention and migration

Every invocation prints its exact retained run directory. Successful and
failed runs remain for inspection; repeated installs consume additional disk
space. There is **no automatic cleanup or new cleanup command**. This avoids
depending on `shutil.rmtree.avoids_symlink_attacks`, which is not available as
a safe reset contract on all supported platforms.

Old scratch roots created before this change have no ownership marker and
are refused unchanged. Use a new build tree or a new external scratch
container. Any relocation/removal of old outputs requires separate explicit
inspection and authorization; do not synthesize a marker to adopt them.

## Focused executed evidence

Host: Linux x86_64, kernel `7.0.0-43-generic`; Python `3.14.3`; CMake `4.3.2`.
Executed from the isolated issue worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 \
  compiler/tests/package/run_installed_test_root_safety_tests.py \
  --cmake /usr/bin/cmake
git diff --check
```

Final safety result: **23 tests, 22 passed, 1 explicit skip** (native Windows
junction execution). `git diff --check` passed. An earlier 22-test version also
passed with the same Windows-only skip; the final suite adds the registered
unowned-root witness and more marker identity controls. No unexpected focused
test failures were observed.

The suite launches both actual runner CLIs. Rejected requests must fail before
a synthetic CMake install boundary and leave complete disposable fixture
snapshots unchanged (contents, links, directory entries, and mutation metadata).
Executed negatives include:

- Root, dangling-root, intermediate `tests`, external-parent, and producer-build
  symlinks; regular-file roots; lexical `..`.
- Unowned empty/nonempty external containers and unowned registered roots.
- Protected build/ancestor paths, unrelated build children, and consumer source
  paths. Repository and home protection use synthetic disposable paths only.
- Wrong schema/version (including boolean/float versions), extra fields, wrong
  runner/build/root identities, malformed/invalid-encoding/oversized markers,
  marker symlinks/directories, and missing/linked CMake caches.

Positive CLI controls reach a deliberate CMake `FATAL_ERROR` install boundary
twice for each runner in both registered and external Unicode paths. They
verify distinct new children and unchanged previous outputs/markers. These
expected nonzero subprocess exits are setup evidence, **not** successful
package installs, compilation, linkage or execution. Additional unit controls
preserve prior-run symlinks and prove repetition does not depend on the
symlink-safe recursive-removal flag. Reparse-bit classification is synthetically
tested on Linux; that does not establish native Windows behavior.

Only suite-created temporary sentinel/script fixtures were automatically
removed by the test fixture teardown. No existing worktree, build output,
source, or user directory was deleted.

## Outstanding gates and non-goals

`package.installed_test_root_safety` is registered beside the existing package
test on Linux and Windows. This agent did not configure or build the product,
run the complete suite, execute a real installed consumer/C17 probe, or execute
Windows. The integration owner must run focused real first/repeated package
checks and the broader Linux suite. Native Windows CI must run the real package
tests, this safety suite (including junctions), and the separately located
external-consumer workflow. Windows symlink tests explicitly report a skip if
the token lacks symlink creation privileges; the junction test is separate and
does not silently skip creation failures.

Independent review and exact-head CI remain outstanding. This report is not
merge/release qualification. No runtime/compiler semantics, public APIs,
workflow, #121 install-layout work, toolchain installation, benchmark, GPU
execution, or `CURRENT_STATE.md` change is included.
