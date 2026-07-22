# Milestone 5 public context ABI follow-up

Date: 2026-07-22

Final reviewed integration checkpoint: `1c40346`

Ownership: public CPU execution-context validation integration, OpenBLAS
provider-state interaction, planner evidence regressions, persistent-executor
lifetime hardening, and focused ABI/runtime validation. No public C ABI layout
or exported symbol was changed.

## Findings resolved

The first independent review found one high-severity placement mismatch and one
medium-severity diagnostic risk.

- The public C runtime previously retained a private validator whose serial,
  provider, and packed probes ran on the creator thread. It now uses the shared
  `validate_cpu_runtime_variants_v1` helper. That helper executes every serial
  probe on persistent worker zero and both parallel probes on two persistent
  workers. The duplicate validator was removed.
- A direct regression creates two strictly bound workers and proves that exact
  validation advances the persistent executor by seven submissions when
  OpenBLAS is absent, or eight when it is linked. This distinguishes worker
  dispatch from creator-thread execution.
- OpenBLAS 0.3.32 can derive `openblas_get_num_procs()` from the affinity of the
  thread that first initializes compute. Provider identity and its maximum
  process-level thread ceiling are therefore frozen on the first provider-info
  query, before bound-worker SGEMM validation. Planner topology and context
  ceilings remain the stricter execution limits. This prevents validation on a
  single bound worker from silently collapsing later provider planning to one
  thread.
- Exact `runtime_validated` evidence is populated for all eight stable variants
  before per-problem legality checks. Regressions prove that tiny parallel
  rejection and invalid AVX-512 workspace rejection retain the exact evidence
  while still reporting their contextual rejection reasons.

The benchmark lane then found an intermittent high-severity
stack-use-after-return under repeated tiny automatic plans. Every persistent
worker copied a pointer to the submitter's stack-resident `SubmissionV1`, but
`run_tasks` waited only for active workers. An inactive worker could therefore
resume after the active-worker barrier and dereference the expired submission.

- Worker participation and `active_threads` are now read while holding the
  executor state mutex.
- Only active workers retain the borrowed pointer after unlocking; inactive
  workers retain no submission pointer and are outside the lifetime barrier.
- A regression alternates 4,096 single-active-worker stack submissions with 64
  all-worker barriers across an eight-worker persistent context. It verifies
  every submission and executor-generation count.
- The benchmark lane replayed 100/100 isolated tiny Release processes and
  100/100 sanitizer processes after the fix without reproducing the failure.

Integrated focused commits:

- `922ed68` — `fix(runtime): authenticate public contexts on bound workers`
- `32efbb1` — `test(planner): preserve exact evidence on rejected plans`
- `55943dc` — `fix(runtime): bound borrowed submissions to active workers`
- `1c40346` — `test(runtime): stress borrowed submission lifetime`

## Validation

Fresh Clang 21.1.8 builds used Ninja `-j2` and OpenBLAS 0.3.32 pthread unless
stated otherwise.

- Full Release suite on the equivalent final source state: 41/41 passed.
- Full Debug suite on the equivalent final source state: 41/41 passed.
- ASan+UBSan focused execution-context, planner, public-context, and benchmark
  matrix: 4/4 passed with leak detection and halt-on-error enabled.
- Integration-lead ASan+UBSan matrix: 6/6 passed. The independent benchmark
  lane's focused ASan+UBSan matrix passed 3/3 before its 100/100 sanitizer
  process stress.
- The executor lifetime regression itself passed 50/50 repeated ASan+UBSan
  process runs after the focused matrix.
- TSan focused execution-context, planner, and public-context matrix with
  OpenBLAS disabled: 3/3 passed.
- Public-context tests force serial packed and parallel AVX2 execution, verify
  query/execute stable-ID agreement, and verify repeated persistent-context
  submission without worker recreation.
- Planner tests verify exact evidence remains independent from per-problem
  legality.
- Release and Debug install/consumer tests passed as part of both 41-test
  suites.
- `git diff --check` passed.

## ABI and ownership result

The existing 15-symbol C export surface remains unchanged. Public descriptor
layouts and size/offset pins remain covered by the ABI compatibility and
installed C17 probes. Execution contexts still own their persistent workers;
callers still own tensors and explicit workspace. No hidden tensor copy,
workspace allocation, fallback, or C++ type was introduced across the C ABI.

## Review status

The creator-thread authentication mismatch, rejected-plan evidence loss, and
borrowed-submission lifetime race are resolved. The final fresh independent
review is recorded separately and is required to have no unresolved high or
medium finding before Milestone 5 completion.
