# CPU-beta documentation truth update

Date: 2026-08-11

Reviewed branch base: `0bafa513ee3c`

Scope: Milestone H product/status documentation only. No compiler, runtime,
package, workflow, public header, or test implementation was changed by this
lane.

Postscript: this focused documentation checkpoint predates later product
hardening. The final local matrix subsequently passed at immutable candidate
`6796fd8`; see
[cpu-beta-final-candidate-validation.md](cpu-beta-final-candidate-validation.md).
That evidence supersedes the intermediate `69d099e` matrix. The historical
scope statements below describe this lane, not the current candidate.

## Purpose

The semantic-foundation implementation and focused independent reviews had
advanced beyond several roadmap/status statements. This update aligns the
product narrative with accepted bounded evidence while keeping the final
CPU-beta acceptance boundary honest.

## Corrected state

- Milestone C map/domain composition is implemented and independently accepted
  for optimizer inspection, but has no CPU lowering or public map API.
- Milestone D recovered GEMM is implemented and independently accepted for
  authenticated analysis/equivalence inspection, but performs no source rewrite
  and cannot enter executable CPU lowering.
- Milestone E explicit F32 GEMM is implemented and independently accepted for
  the focused Linux native-v1 to Matcore MLIR to stable runtime-dispatch object
  and executable path.
- Linux x86-64 source-evaluation and execution-thread FP admission, native
  worker admission, and supported backend conformance are implemented and
  independently accepted for the focused scope. Additive status 26 does not
  change an existing C layout or function signature.
- Milestone F accepts the existing reviewed bounded technical limit; Milestone
  7 itself remains partial/open and native-BLAS parity is not claimed.
- Milestone G accepts bounded current-version packed-B ownership/invalidation,
  borrowed-string, and additive-evolution contracts; the public API/ABI/backend
  freeze remains deferred.
- Milestone H is active, not complete. The final full local configuration
  matrix passed at code candidate `6796fd8`; exact-head hosted results, hosted
  Windows validation, normal merge/tag, and final independent review remain
  pending.

## Product-profile clarification

The source compatibility default is Matcore MLIR `OFF` with semantic default
`capture-v0`. The Linux CPU-beta product profile enables exact MLIR 21.1.8 and
defaults to `matcore-mlir`. Windows Release, Debug, and supported sanitizer
profiles remain MLIR `OFF`/default `capture-v0` and must prove an explicit
semantic request unavailable without creating an artifact.

The installed package capability/default variables and the optional
`SEMANTIC_PIPELINE` consumer argument are documented. Invalid or unavailable
combinations fail closed. MLIR-enabled installation exposes `matcore-mlir` only
as a leaf tool; private MLIR libraries, targets, headers, and development paths
remain unexported.

The executable semantic backend's one-shot
`matcore_runtime_gemm_f32_v0` boundary is also explicit: it requires zero caller
workspace and performs no MDSLC-owned packing/workspace allocation. A linked
opaque OpenBLAS provider may manage internal memory under its own contract
while being probed or executed. Workspace/context native packed and parallel
variants remain available through their existing additive APIs; they are not
silently made available through the one-shot semantic route.

## Claims intentionally withheld

This update does not claim:

- a final clean-head Release/Debug/sanitizer/package matrix as of this report's
  original `0bafa51` checkpoint (an intermediate matrix later passed at
  `69d099e`, and final-candidate evidence passed at `6796fd8`);
- hosted pull-request or Windows results for the semantic-foundation candidate;
- a merged, tagged, or published CPU beta;
- executable map/domain or recovered-loop replacement;
- Windows semantic MLIR execution or generated Linalg/Vector compute;
- native/OpenBLAS parity or Milestone 7 completion;
- a public API/ABI/backend-contract freeze; or
- GPU/NPU work.

`CPU_BETA_V1.md` now records the local Milestone H gates as passed at
`6796fd8` and retains the independent-review, hosted Windows/Linux, merge, and
tag gates as pending.

## Documentation validation

Before the focused documentation commit:

- `git diff --check` passed;
- `tests/check_repository_hygiene.sh` passed;
- stale implementation-state phrases were searched across the four updated
  existing documents; and
- the changed-file and staged-file sets were checked to contain only the six
  files owned by this lane.

No build or runtime test was run by this documentation lane. Its original
checkpoint was not a substitute for the fixed-head Milestone H matrix; that
matrix was subsequently run and recorded separately. The final superseding run
is recorded at `6796fd8`.
