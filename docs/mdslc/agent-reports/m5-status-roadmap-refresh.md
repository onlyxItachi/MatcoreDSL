# Milestone 5 status and roadmap refresh

Date: 2026-07-22

Base integration checkpoint:
`37d8ea96c88c1274388b190acdb2e851e9573afd`

Ownership was limited to `docs/mdslc/STATUS.md`, `docs/mdslc/ROADMAP.md`,
ADR-0008, `docs/mdslc/PERFORMANCE_PREFLIGHT.md`, and this report. No compiler,
runtime, build, test, workflow, performance-summary, or GitHub state changed.

## Outcome

The current documents now distinguish three different facts that the prior
status blurred:

1. Milestone 4 is published: PR #10 merged normally at `e4dc0af`, issue #8 and
   milestone #2 are closed, and `mdslc-cpu-performance-foundation-v1` anchors
   the merge.
2. The Milestone 5 Linux implementation and focused validation exist on
   `mdslc/cpu-isa-parallel-v1`.
3. Milestone 5 is not accepted or published until final-tip gates, a whole-diff
   review, hosted checks, normal merge, issue/milestone closure, and the
   immutable tag complete.

The refreshed technical inventory records:

- eight stable F32 planner-v3 variants;
- capability record v2 and exact-context legality;
- physically exercised packed/parallel AVX2/FMA and AVX-512F/FMA on the
  declared Linux host;
- persistent workers, explicit per-worker workspace, deterministic placement,
  and a 15-function public C export surface;
- one-node physical topology evidence and synthetic-only multi-node planning;
- BF16-to-F32 and I8-to-I32 reference semantics only;
- no optimized BF16/VNNI/AMX implementation or claim; and
- no Windows compiler, runtime, artifact, package, consumer, workflow, or ZIP
  validation claim.

ADR-0008 remains an accepted architecture decision, now labeled as an
implementation-complete candidate with acceptance pending. It explicitly
preserves the distinction between repeat-safe internal `shutdown()` and the
public opaque handle's consume-once destroy contract.

## Evidence reviewed

- `docs/mdslc/agent-reports/m5-final-docs-inventory.md`
- `docs/mdslc/agent-reports/m5-capability-topology.md`
- `docs/mdslc/agent-reports/m5-avx512-types.md`
- `docs/mdslc/agent-reports/m5-parallel-planner.md`
- `docs/mdslc/agent-reports/m5-public-context-abi.md`
- `docs/mdslc/agent-reports/m5-public-context-abi-followup.md`
- `docs/mdslc/agent-reports/m5-balanced-regret-final-closure.md`
- the integrated public C header and runtime/planner tests at the base
  checkpoint.

The host-specific 20-shape regret summary is labeled with its exact guarded
schema-v4 source checkpoint, `5f634ae`, and is not presented as a universal
claim or a substitute for the pending full acceptance run.

## Validation

```text
git diff --check
bash tests/check_repository_hygiene.sh
gh pr view 10 --repo onlyxItachi/MatcoreDSL --json ...
gh issue view 8 --repo onlyxItachi/MatcoreDSL --json ...
gh issue view 9 --repo onlyxItachi/MatcoreDSL --json ...
test -z "$(git diff --name-only -- . ':!docs/mdslc/STATUS.md' \
  ':!docs/mdslc/ROADMAP.md' \
  ':!docs/adr/0008-mdslc-advanced-cpu-backend-v2.md' \
  ':!docs/mdslc/PERFORMANCE_PREFLIGHT.md' \
  ':!docs/mdslc/agent-reports/m5-status-roadmap-refresh.md')"
```

All checks passed. The live read-only GitHub queries confirmed PR #10 merged at
`e4dc0af`, issue #8 is closed under milestone #2, and issue #9 remains open
under milestone #3. No build was required because this lane changed Markdown
only; final integration owns the exact-tip build and test gates.
