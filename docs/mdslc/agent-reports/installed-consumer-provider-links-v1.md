# Explicit provider links for installed manual consumers

This focused test-only follow-up builds on `e5652ecc7f02e8b8463a6658abfc15ac84fd23d3`.
It complements the integration owner's production installed-driver provider fix;
it does not replace that fix or reinterpret the failed hosted run as passing.

## Change

The install-contract script now constructs one argument list for every manual
archive consumer. It contains the installed runtime, the exact configured
provider file when enabled, separate `-Xlinker -rpath -Xlinker` arguments for the
provider and runtime directories, and the existing link flags. No `-lopenblas`
name lookup is introduced, and provider paths are never embedded inside a
comma-separated `-Wl` payload. Scoped `--push-state --no-as-needed` around the
provider file followed by `--pop-state` retains a direct dependency without
changing the caller's remaining linker state; executable RUNPATH alone would
not locate Runtime's transitive dependency.

The common list is consumed by:

- all three result/candidates/private_value executables;
- both mixed-configuration ownership executables;
- every positive/negative link inside the private-Value ABI child;
- the installed generated-leaf ASan control.

The producer-only `-c` commands remain compile-only and do not acquire irrelevant
link arguments. `PROVIDER` passed to the separate installed-driver contract
retains the configured spelling, because its bound RUNPATH directory can differ
from a symlink target's physical directory. It is cleared for provider-OFF.

Within feature-ON manual-consumer testing, provider-ON requires a nonempty
absolute existing file, rejects a directory, and resolves its canonical path
before deriving arguments. The feature-OFF absence-only package path returns
before this guard and requires no experimental-driver provider property.
Provider-OFF returns
empty provider identity/arguments even when a stale input path was supplied.
This is a test configuration/file guard, not a new provider authenticator;
production artifact authentication remains outside these manual consumers.

The ABI child takes a native-command string instead of an argument list. The
script now serializes the exact common list by quoting every argument and
escaping quotes/backslashes, then lets the existing child
`separate_arguments(NATIVE_COMMAND)` restore the tokens. This preserves spaces
and commas in canonical file/directory paths, without a separate ABI-only
provider lookup or a manually assembled rpath string.

Both the identity regression and the new link-argument regression now execute
as small `cmake -P` subprocesses in the existing feature-ON install-contract
test, before actual consumers. There is no new CTest name or count change.

## Bounded validation on 2026-09-08

Executed with the existing CMake 4.3.2:

```sh
cmake -DSCRATCH=/tmp/mdslc-provider-link-test-XGffM87L \
  -P compiler/tests/cmake/experimental_regions_consumer_link_test.cmake
cmake -P compiler/tests/cmake/experimental_regions_install_identity_test.cmake
```

The link regression passed:

- provider-ON canonicalization through a symlink;
- exact file and separate rpath arguments for a path containing spaces/comma;
- scoped no-as-needed arguments with immediate linker-state restoration;
- provider-ON followed by provider-OFF using the same output variables, proving
  no stale provider identity or flags survive;
- empty, missing, directory, and relative provider rejection;
- native-command serialization locally and through an actual child `cmake -P`,
  preserving runtime/provider paths, sanitizer flags, commas, spaces, embedded
  quotes and backslashes.

The scratch `.so` file is explicitly a text file-identity fixture, not an
OpenBLAS implementation, and no numerical/provider execution claim is derived
from it. An initial negative-test diagnostic matcher was corrected to use a
stable diagnostic token after CMake wrapped the human-readable error across
lines; the provider guard itself had correctly rejected that input.

The existing identity regression again passed the actual NEW-policy branch,
documented OLD-lookup emulation, four positive output formats and six rejection
controls. `git diff --check` passed. Review confirms every manual archive link
and the ABI-child serialized argument set uses the common list; no duplicate
legacy `-Wl,-rpath` path construction remains in the install script.

This lane performed no new full build, reinstall, production edit, root
worktree write, system/toolchain change, or merge. A fresh complete installed
package execution with the integration owner's nonstandard-provider production
fix remains that integration lane's responsibility. These argument-level tests
do not establish candidate identity, authenticated arbitrary manual linking,
performance, GPU support, or cross-platform behavior.
