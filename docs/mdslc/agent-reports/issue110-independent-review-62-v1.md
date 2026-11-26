# Independent review: #110 installed-package test scratch roots

**Accept the no-reset fix and the Linux path-alias correction.** No blocker
found for the stated contract: caller-controlled, stable paths and a fresh
run child, not a hostile-filesystem sandbox. Native Windows validation is
still required for Windows qualification.

Reviewed 2026-11-26 at `883c89dda7c7eb245cafffb8ae926a77205a9b37`;
canonical base `1c1221e89272eb2ad6ff058b8e26de4653f3a93d`; previous
candidate with the alias blocker `4f94d7a1d4c5f5d717e89231dae7e0fc4417494d`.
Read root `AGENTS.md`, [issue #110](https://github.com/onlyxItachi/MatcoreDSL/issues/110),
the helper, both runner deltas, safety tests, implementation report, CTest
registration and the Windows external-consumer workflow.

- Both production runner entry points call the shared helper **before**
  writing their scratch output or invoking CMake. Neither recursively removes
  the user-specified root. Both use the returned fresh child for staged
  output. The Windows consumer's explicitly requested import-report output
  is a separate pre-existing output contract, not confined to that child.
- The old `4f94d7a` compared an `abspath` candidate with resolved protected
  directories. On Linux the first can keep `//` where the second has `/`.
  The candidate inspects lexical build/root components by `lstat` for
  links, reparse flags and non-directories first. Only *after* these checks
  does it resolve the build/root and compare them in the same namespace as
  repository, consumer source and home. Marker identity, accepted build and
  actual scratch creation now use those same canonical paths.
- The guard rejects filesystem roots and equality/ancestry over the protected
  paths; only the exact registered child is allowed inside the producer
  build, including a build nested in the checkout. It rejects other checkout
  descendants and descendants of the explicit consumer source. Home itself
  and its ancestors are protected, not every user-owned path below home.
  A new external container is allowed. Reusing a container requires a bounded
  regular non-link marker matching schema, integer version, runner, build
  and root. An unowned directory is not adopted or cleaned, even if empty;
  a valid marker authorizes only a fresh exclusive child, not deletion.
- CTest's two existing registered paths match the helper's owner names.
  The Windows Release workflow runs CTest and separately passes a Unicode,
  space-containing root under `RUNNER_TEMP` plus its external report path.
  Neither caller needs a fixed external basename or an unowned pre-created
  directory. Ordinary Windows path classification is covered synthetically;
  reading that workflow does not establish that it ran successfully.

Independent checks from the candidate root:

```sh
python3 -B compiler/tests/package/run_installed_test_root_safety_tests.py --cmake /usr/bin/cmake
git diff --check 1c1221e89272eb2ad6ff058b8e26de4653f3a93d...883c89dda7c7eb245cafffb8ae926a77205a9b37
```

**27 test methods: 25 passed, 2 skipped** on Linux (native Windows device
namespace and junction). The diff check passed. Both real CLI entry points
were exercised with disposable fixtures, including the `//` protected-path
cases, missing/unowned/linked roots and markers, and canonical marker reuse
with alternate build/root spellings. Rejection checks compare before/after
snapshots; successful setup reaches a deliberately failing synthetic CMake
install boundary and preserves the prior run. This is not a real package
install or an executed Windows consumer.

This guard does not authenticate cache contents, cryptographically attest a
marker, protect against concurrent path replacement, or establish physical
identity across bind mounts, multiple volume/share aliases, or unusual
filesystem case semantics. Old runs remain on disk. I ran no destructive
historical reproduction, product build, real package integration, native
Windows, full suite or hosted CI in this review.
