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

## Resumed qualification, 2026-11-26

Live canonical main and this branch's base remain the starting checkpoint
above. The independent review of `1301ca70e0e09a84e3e75cb9e61619cf8df7f92e`
accepted the bounded guard but requested explicit malformed-count coverage.
The added controls exercise map counts and rank counts of 0, 1, 2 and 4 through
the verifier, encoder and structured-carrier checker; each requires the count
diagnostic before any map access. No production behavior changed in this step.

Rebuilt the existing Release contraction target with Ninja `-j2`:
**325 checks, zero failures**, focused CTest **1/1 passed**, zero skips.
Built and ran the six affected MLIR tests (contraction model, core semantics,
structured GEMM, derived certificate, bufferized GEMM and CPU runtime lowering):
**6/6 passed**, zero skips. `git diff --check` and
`bash tests/check_repository_hygiene.sh` also passed. These results use the same
coherent 21.1.8 tuple and OpenBLAS-disabled configuration described above.
Broader standalone/hosted qualification remains separate; no merge or release
qualification follows from these local results.

## Complete local regression and prerequisite composition

On 2026-11-26, clean source
`0fd5f767dd9ddcf15a19a4223897edd4279eb41e` was reconfigured and built in the
existing `../build-null-maps-release` directory, then tested with
`ctest --output-on-failure -j1`: **346/346 passed, zero failures, zero skips**
in 3232.48 seconds. This is a full run of the registered standalone suite for
that configuration, not a fresh-build claim. It includes frontend/IR/MLIR,
runtime/planner, source-generated execution, ownership/failure negatives,
installed/relocated and source-inaccessible package tests. Configuration stays
Release, native/bootstrap/MLIR/regions ON, OpenBLAS and GPU issuers OFF,
`capture-v0` default and scalar built-in schedule, using the coherent 21.1.8
tuple above. Generated-leaf ASan tests in this suite do not constitute a fully
instrumented compiler build or GPU execution evidence.

The preserved raw CTest log is
`../build-null-maps-release/issue119-ctest-20261126.log`, SHA-256
`d373151fa5ec4e76bd83f5154d0bf92786b7d0f9e3ee9ee9b864538bff7ef77d`.

Normal merge `d011979041f058417bda2452787550ef791c344f` then incorporated
reviewed Windows prerequisite
`be8c59f804f559a66b9c6b2fb8042fde4f6f6023` ([PR #125](https://github.com/onlyxItachi/MatcoreDSL/pull/125)).
There was no overlap with the issue's verifier/test changes. Reconfiguration
and the complete incremental build succeeded with Ninja `-j2` (28 steps).
The composed tree passed **11/11 focused CTests, zero skips**: the six MLIR
contracts listed above, platform snapshots, native frontend focused tests,
frozen-host independence, artifact-symbol ownership and authenticated host
thunks. Direct contraction testing again reported **325 checks, zero failures**;
diff and repository-hygiene checks passed.

The composed-test log is
`../build-null-maps-release/issue119-composed-ctest-20261126.log`, SHA-256
`82b19276ee509afe6506d8449e1674f34d81d1171686110b451eb6eab3c529fc`.
The 346-test result belongs to the pre-composition SHA, not the later merge or
this report-only update. Fresh exact-head hosted qualification remains required
before integration. No default, numerical contract, execution authority or
performance claim changed.
