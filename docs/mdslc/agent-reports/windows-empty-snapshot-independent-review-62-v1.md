# Independent review: reject empty snapshot paths before filesystem conversion

**Accept the snapshot follow-up; no blocker in this diff.** Reviewed source
`0be2293758418115341805e2eb32c1215e232a94` against its parent
`53399315c13d0da56eba13eabe6f335dad154897` on 2026-11-26.

The earlier [Windows Release job](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45829291153/job/139941496023)
passed 51/52 CTests. The sole failure was the existing compound assertion in
`support.platform.v1` that an empty snapshot request must set an error and
return neither existence nor identity. The log did not print the individual
fields. The [Microsoft 14.44-family STL implementation](https://github.com/microsoft/STL/blob/1f6e5b16ec02216665624c1e762f3732605cf2b4/stl/inc/filesystem#L3011-L3029)
illustrates why relying on `absolute(path, ec)` was insufficient: it clears
the error and returns an empty result for empty input, before calling Win32.
This is source evidence for a plausible route, not field-level telemetry
from the failed process.

The four-line guard rejects only `path.empty()`, returns a default
`FileSnapshotV1` and sets a nonempty diagnostic before any filesystem query.
It does not canonicalize or change the original traversal of nonempty
relative, absolute, nonexistent, directory, hard-link or symlink spellings.
File/argument capture consumers check errors and identities; negative lookup
observation still accepts an ordinary missing nonempty path, and ancestor
traversal already stops at an empty parent. No caller contract requires an
empty path to represent the current directory. No API or ABI changed.

The test retains all three old predicates and adds checks for empty
normalized path and empty traversal chain. A subsequent valid fixture
snapshot using the same diagnostic string must clear the error and recover
the original file identity. The previous valid-file, hard-link and path
normalization controls remain in place.

Independent rerun, from this worktree, using the existing Linux Release
bootstrap build (cache points to this source; native frontend, MLIR, regions
and OpenBLAS disabled):

```sh
ctest --test-dir ../build-windows-support-check --output-on-failure -j1 -R '^support\.platform\.v1$'
git diff --check 53399315c13d0da56eba13eabe6f335dad154897 0be2293758418115341805e2eb32c1215e232a94
```

Focused CTest **1/1 passed, zero skips**; diff check passed. I did not
rebuild or run the full suite. This is not a new Windows result. The earlier
hosted run predates this source and is a failure, not package, Debug, or
ASan qualification for the corrected head.
