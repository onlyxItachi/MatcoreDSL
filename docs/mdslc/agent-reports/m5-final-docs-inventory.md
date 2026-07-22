# Milestone 5 final documentation and GitHub evidence inventory

Audit time: 2026-07-22 21:00:23 +03:00

Audit checkpoint: `11d0116cc3fffb6ce64c49ff9aa739324c93a188`

Ownership: read-only inspection of current milestone documentation and live
GitHub metadata, plus this report. No production source, shared status document,
GitHub issue, milestone, pull request, branch, tag, or workflow was changed.

## Executive boundary

The local Milestone 5 integration history contains the advanced Linux CPU
implementation, but it is not yet a published or accepted milestone. At the
audit checkpoint the local integration tip was 71 commits ahead of the remote
Milestone 5 branch. Exact-tip performance calibration, complete acceptance
reruns, a final whole-diff review, hosted checks, normal merge, and the immutable
tag remained gates. Windows remained modeled or deferred; it had no compiler,
COFF/PE, DLL, package, consumer, workflow, or runtime evidence.

Documentation may describe implemented mechanisms and completed focused tests,
but it must not yet say that Milestone 5, hosted CI, Windows, optimized BF16 or
INT8, AMX, or physical multi-node NUMA passed.

## Live GitHub state

| Item | Audited state |
| --- | --- |
| Default branch | `main` |
| `origin/main` | `e4dc0affff6c540a65435ba25c5cefa4d69cb562` |
| Milestone 4 PR | [#10](https://github.com/onlyxItachi/MatcoreDSL/pull/10), normally merged into the commit above |
| Milestone 4 issue/milestone | [issue #8](https://github.com/onlyxItachi/MatcoreDSL/issues/8) closed; [milestone #2](https://github.com/onlyxItachi/MatcoreDSL/milestone/2) closed |
| Milestone 4 tag | `mdslc-cpu-performance-foundation-v1`, peeled commit `e4dc0affff6c540a65435ba25c5cefa4d69cb562` |
| Milestone 5 issue/milestone | [issue #9](https://github.com/onlyxItachi/MatcoreDSL/issues/9) open with every checklist item unchecked; [milestone #3](https://github.com/onlyxItachi/MatcoreDSL/milestone/3) open with one open issue |
| Milestone 5 remote branch | `mdslc/cpu-isa-parallel-v1` at `e4dc0affff6c540a65435ba25c5cefa4d69cb562` |
| Milestone 5 local audit tip | `11d0116cc3fffb6ce64c49ff9aa739324c93a188`, 71 commits ahead of the remote branch |
| Milestone 5 PR | none |
| Milestone 5 tag | `mdslc-cpu-backend-v2` does not exist |
| PR #3 | closed, unmerged, and unchanged |

PR #10's four reported checks passed, and `main` passed its post-merge Linux CI
and repository-hygiene runs. Those runs authenticate Milestone 4 only. The only
GitHub Actions run on `mdslc/cpu-isa-parallel-v1` used the remote base commit
`e4dc0af`; it does not validate any local Milestone 5 commit. There is no
Windows workflow under `.github/workflows/`; all current jobs use Ubuntu.

## Statements supported at the local checkpoint

The focused reports and integrated source support stating that the branch has:

- capability record v2 with distinct hardware, OS state, compiler,
  implementation, and physical runtime-validation domains;
- deterministic topology v1, allowed-processor restriction, compact/scatter
  worker affinity, and synthetic multi-node NUMA planning without page
  placement or migration;
- eight stable F32 planner-v3 candidates: the five Milestone 4 candidates plus
  packed AVX-512, parallel AVX2, and parallel AVX-512;
- a persistent execution context, caller-owned shared/per-worker workspace,
  exact-context validation, deterministic row-band work, and no nested native
  and OpenBLAS pools;
- physically executed packed and parallel AVX2/FMA and AVX-512F/FMA F32 paths
  on the declared Ryzen host, with an isolated AVX-512 microkernel artifact
  containing ZMM packed-FMA instructions;
- typed BF16-to-F32 and I8-to-I32 reference semantics and their additive C ABI;
- the existing 15-function exported C ABI surface, including the public
  execution-context path; and
- a balanced forward/reverse complete-call planner-regret measurement contract.

These are implementation or focused-validation statements. The complete suite
and package evidence in the newest follow-up report is from `77afa12`, before
the later planner calibration and balanced-regret commits. The fairness report
proves its focused Release tests only. A fresh exact-tip acceptance run is still
required before promoting these statements to the milestone verdict.

## Claims that remain unsupported

- **Milestone completion:** no final exact-tip Linux gate, whole-diff final
  review, hosted M5 checks, merge commit, closed issue, or tag exists.
- **Optimized low-precision ISA paths:** AVX-512 BF16, AVX-512 VNNI, AMX-BF16,
  and AMX-INT8 variants are not implemented or runtime-validated. BF16/I8
  support is reference semantics only. The host has no AMX feature.
- **NUMA:** one-node Linux discovery is physical; multi-node behavior is
  synthetic-only. There is no NUMA allocation, binding, interleaving, page
  migration, or physical multi-node performance result.
- **Windows:** frontend, runtime DLL/import library, planner, native variants,
  package/consumer, and ZIP distribution are all unvalidated or not produced.
  `_WIN32` modeling is not Windows support.
- **Hosted CI:** the locally edited OpenBLAS ON/OFF workflow and all Milestone 5
  code have not executed on GitHub at this audit point.
- **Final performance/regret:** the committed tree has no sanitized Milestone 5
  performance report, and the balanced-regret change postdates the earlier
  calibration evidence. Do not reuse preliminary lane diagnostics or the
  Milestone 4 single-thread report as final Milestone 5 evidence.

## Exact documentation updates required after final evidence exists

Line references below refer to audit checkpoint `11d0116`.

### `docs/mdslc/STATUS.md`

- Lines 5-22 still identify the pre-Milestone-4 `main` baseline and say
  Milestone 4 awaits publication. Replace that header with the verified M4
  merge/tag/PR state and the current M5 branch, issue, milestone, and verdict.
- Lines 18-70 should remain a historical Milestone 4 summary, but the
  publication sentence must record PR #10, merge `e4dc0af`, closed issue #8,
  closed milestone #2, and the immutable M4 tag.
- Lines 72-96 incorrectly say the Milestone 3 and Milestone 2 hosted merge gates
  remain. Mark both historical gates complete instead of presenting them as
  current work.
- Insert a new Milestone 5 section before the historical milestones. It should
  enumerate the eight F32 variants, capability-v2/topology-v1 records,
  persistent context and 15-symbol ABI, exact validated ISA status, typed
  reference status, single-node/synthetic-NUMA boundary, balanced-regret and
  scaling results, exact-tip test results, review status, and Windows status.
- Lines 153-175 end the pipeline at planner v2 and five variants. Extend it
  through exact-context capability validation, planner v3, explicit execution
  context/workspace, packed AVX-512, and parallel AVX2/AVX-512.
- Lines 204-312 are Milestone 1 evidence with old temporary paths. Label them
  explicitly historical or move them behind a milestone-specific heading so
  they are not mistaken for current acceptance results.
- Lines 314-340 contain stale current-state language. In particular, lines
  334-336 incorrectly say AVX-512, native parallel execution, and BF16/INT8 are
  not implemented. Replace this with the precise reference-versus-accelerated,
  physical-versus-synthetic, Linux-versus-Windows status.

### `docs/mdslc/ROADMAP.md`

- Lines 55-70 list capability v2, AVX-512 F32, persistent parallel execution,
  topology, and BF16/INT8 reference semantics as future tasks although they are
  implemented locally. After Linux acceptance, add an explicit Milestone 5
  completed gate and replace these items with the remaining publication and
  focused Windows validation work. Do not mark Windows complete until its
  hosted job executes.
- Retain the later-operation/GPU section as future scope; no GPU milestone is
  authorized by this work.

### `AGENTS.md`

- Lines 37-50 describe the five-candidate planner-v2 registry as if it were the
  complete current runtime. Preserve the serial requirements, but distinguish
  the Milestone 4 baseline from planner v3's eight-candidate, explicit-context
  advanced registry.
- Add the exact current boundary: BF16/I8 are typed reference paths, accelerated
  BF16/VNNI/AMX are unavailable, and Windows is unsupported until native
  validation passes.

### ADR and preflight documents

- `docs/adr/0008-mdslc-advanced-cpu-backend-v2.md` line 3 says only “Accepted
  for implementation.” After acceptance, record the implemented Linux outcome
  and add a short evidence/status table without converting unavailable hardware
  into support. The public handle remains single-destroy; only internal
  `shutdown()` is repeat-safe.
- `docs/mdslc/ADVANCED_CPU_PREFLIGHT.md` should gain a historical-snapshot banner
  like the other preflights. Its lines 65-79 state eligibility and planned
  validation, not final results.
- `docs/mdslc/PERFORMANCE_PREFLIGHT.md` is already correctly labeled historical.
  Lines 156-159 should say specifically that Milestone 5 added AVX-512 F32,
  typed BF16/I8 references, persistent parallelism, and topology planning; it
  did not add optimized BF16/VNNI/AMX or Windows validation.
- `docs/mdslc/PREFLIGHT.md` is already clearly historical and should not have old
  rewritten SHAs presented as current elsewhere.

### Reports and performance evidence

- Preserve lane reports as chronological evidence. Do not rewrite statements
  that were true in their isolated lane, such as AVX-512 being unavailable in
  `m5-capability-topology.md`, the public ABI being absent in
  `m5-avx512-types.md`, worker affinity being absent in `m5-benchmark-v2.md`, or
  full hosted CI being pending in `m5-package-ci-hardening.md`. The final status
  should explicitly point to their later follow-up reports.
- Add one sanitized reviewed Milestone 5 report under `docs/performance/cpu/`.
  It must use the balanced forward/reverse regret contract, identify the exact
  source checkpoint, and separate complete-call native measurements from
  OpenBLAS provider scheduling. It should report noise exclusions rather than
  converting them into passing regret samples.
- Add a final exact-tip integration report and
  `docs/mdslc/agent-reports/m5-final-adversarial-review.md`. The existing public
  context follow-up closes executor findings through `77afa12`; it is not a
  whole-diff review of the later planner and benchmark changes.
- Add a separate Windows validation report after the hosted run. Until then the
  exact status is: frontend unvalidated; DLL/import library not built; planner
  unvalidated; native variants unavailable on Windows; package/consumer
  unvalidated; ZIP not produced.

## Publication sequence required for truthful final documentation

1. Complete balanced-regret and cross-ISA/thread evidence at one frozen source
   checkpoint and commit only the sanitized summary.
2. Run fresh exact-tip Release, Debug, ASan/UBSan, TSan, OpenBLAS ON/OFF,
   install/consumer/C17 ABI, artifact, native `.mdsl`, hygiene, and legacy gates.
3. Run a fresh whole-diff adversarial review and resolve every high/medium
   finding.
4. Update `STATUS.md`, `ROADMAP.md`, ADR-0008, `AGENTS.md`, and preflight
   cross-references from those exact results.
5. Push the M5 branch, open a normal non-draft PR, and wait for hosted Linux
   checks. Only hosted runs at the final branch SHA count.
6. Perform and monitor the focused Windows workflow. Report each Windows
   frontend/runtime/planner/package/native-variant result independently.
7. After normal merge, close issue #9/milestone #3 and create
   `mdslc-cpu-backend-v2` at the verified merge commit. Do not pre-create the
   tag or mark optional hardware paths as runtime-supported.

## Commands used

```text
git status --short --branch
git branch --show-current
git rev-parse HEAD
git worktree list --porcelain
git log --oneline --reverse origin/mdslc/cpu-isa-parallel-v1..HEAD
git ls-remote --heads --tags origin
gh auth status
gh repo view onlyxItachi/MatcoreDSL --json ...
gh pr list --repo onlyxItachi/MatcoreDSL --state all --json ...
gh pr checks 10 --repo onlyxItachi/MatcoreDSL
gh issue list --repo onlyxItachi/MatcoreDSL --state all --json ...
gh issue view 9 --repo onlyxItachi/MatcoreDSL --json ...
gh api --paginate repos/onlyxItachi/MatcoreDSL/milestones?state=all
gh api --paginate repos/onlyxItachi/MatcoreDSL/branches?per_page=100
gh run list --repo onlyxItachi/MatcoreDSL --limit 15 --json ...
```

All GitHub operations were read-only.
