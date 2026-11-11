# Issue 119: reject null contraction-map handles

Starting checkpoint: `1c1221e89272eb2ad6ff058b8e26de4653f3a93d`.
Scope: [Issue #119](https://github.com/onlyxItachi/MatcoreDSL/issues/119),
the existing internal topology rejection contract only.

## Change and invariants

The verifier checks every indexing-map handle before the first `getContext()`.
A default/null `AffineMap` is rejected with a diagnostic. A valid affine map
with zero results is not null: rank-zero DOT output remains admitted. Canonical
operation identities, maps, orientations, shapes, numerics and execution
authority are unchanged. Encoding and structured-carrier checking keep using
the same verifier; no parallel validation authority was added.

The existing adversarial test now checks all seven nonempty subsets of null
map slots through verification, encoding and structured-carrier consumption.
It also checks diagnostic replacement/clearing, valid rank-zero DOT encoding,
and non-null maps belonging to different MLIR contexts.

## Local evidence

On Linux x64, Clang/LLVM 21.1.8 from the installed Ubuntu tuple was paired with
the existing private MLIR `21.1.8-6ubuntu1` package. No package was installed.
Release, native frontend, Matcore MLIR and experimental regions were enabled;
OpenBLAS was disabled. Only the contraction test and its dependencies were
initially built with Ninja `-j2`.

- `matcore_mlir_contraction_model_tests`: **301 checks, zero failures**.
- `ctest -R '^mlir\.structured\.contraction-model-v1$' -j1`:
  **1/1 passed**, zero skips.
- Effective negative control: temporarily remove only the new guard in this
  isolated worktree, rebuild the same test, and execute it as a bounded child
  with core dumps disabled. The child returned **SIGSEGV (-11)** on the new
  malformed-handle fixture. Restore the guard, rebuild, and repeat CTest:
  **1/1 passed**. No malformed source CLI or runtime artifact was executed.
- `git diff --check`: passed.

An initial narrower configuration (Matcore MLIR ON, experimental regions OFF)
failed generation because existing strict-CPU-ISA tests refer to the absent
`mdslc-region` target. This unrelated configuration combination was not repaired
or described as passing. Enabling the existing region configuration allowed the
focused target above to build. Generated TableGen code emitted existing
unused-parameter warnings; they were not suppressed or refactored.

Independent review, affected broader regression tests, hosted CI and merge
qualification are separate gates; the focused results alone do not qualify a
release or establish public-source reachability of the old null dereference.
