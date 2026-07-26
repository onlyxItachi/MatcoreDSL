# Local MatcoreDSL worktree cleanup

Date: 2026-07-26

This report records the local-only cleanup performed before opening the
Milestone 6 CPU performance audit. It is operational evidence, not a
repository portability contract.

## Safety method

Every registered worktree was checked with:

```text
git -C <worktree> status --short --branch
git -C <worktree> rev-parse HEAD
git -C <worktree> log --oneline --decorate -5
git branch --contains <HEAD>
git tag --contains <HEAD>
git cherry origin/main <branch>
```

Process working directories and open file descriptors were also checked for
MatcoreDSL paths. Registered worktrees were removed only through
`git worktree remove`; no forced removal, raw deletion of a registered
worktree, reset, branch deletion, or history rewrite occurred.

## Removed completed worktrees

The following 38 worktrees were clean, idle, and had no commits absent from
`origin/main` according to `git cherry`. Their local branches were retained.

- `MatcoreDSL-wt-cpu-isa-parallel-v1`
- `MatcoreDSL-wt-cpu-performance-v1`
- `MatcoreDSL-wt-m4-build-validation`
- `MatcoreDSL-wt-m4-calibration`
- `MatcoreDSL-wt-m4-compute-only-fix`
- `MatcoreDSL-wt-m4-final-review`
- `MatcoreDSL-wt-m4-kernel`
- `MatcoreDSL-wt-m4-perf`
- `MatcoreDSL-wt-m4-plan`
- `MatcoreDSL-wt-m4-platform`
- `MatcoreDSL-wt-m5-affinity-runtime`
- `MatcoreDSL-wt-m5-avx512-types`
- `MatcoreDSL-wt-m5-balanced-regret-final-closure`
- `MatcoreDSL-wt-m5-balanced-regret-review`
- `MatcoreDSL-wt-m5-balanced-regret-review-closure`
- `MatcoreDSL-wt-m5-benchmark-evidence-final2`
- `MatcoreDSL-wt-m5-benchmark-integration`
- `MatcoreDSL-wt-m5-benchmark-v2`
- `MatcoreDSL-wt-m5-capability-topology`
- `MatcoreDSL-wt-m5-ci-planner-portability`
- `MatcoreDSL-wt-m5-docs-status`
- `MatcoreDSL-wt-m5-executor-lifetime-fix`
- `MatcoreDSL-wt-m5-final-adversarial`
- `MatcoreDSL-wt-m5-final-docs-inventory`
- `MatcoreDSL-wt-m5-linux-matrix-report`
- `MatcoreDSL-wt-m5-package-ci-hardening`
- `MatcoreDSL-wt-m5-parallel-planner`
- `MatcoreDSL-wt-m5-perf-report`
- `MatcoreDSL-wt-m5-public-context-abi`
- `MatcoreDSL-wt-m5-public-context-final-review`
- `MatcoreDSL-wt-m5-regret-fairness`
- `MatcoreDSL-wt-m5-topology-planner-hardening`
- `MatcoreDSL-wt-m5-windows-ci-seed`
- `MatcoreDSL-wt-win-platform-support`
- `MatcoreDSL-wt-windows-host-topology-callers`
- `MatcoreDSL-wt-windows-runtime-topology`
- `MatcoreDSL-wt-windows-x64-portability-audit`
- `MatcoreDSL-wt-windows-x64-v1`

Four unregistered, generated-only review directories had no `.git` metadata
and no process references. They were removed after inspection:

- `mdslc-audit-cross-repo`
- `mdslc-audit-header-race`
- `mdslc-audit-symlink-quote`
- `mdslc-review-tmp`

`git worktree prune --expire now` completed afterward.

## Preserved dirty or unmerged work

These registered worktrees were deliberately retained because their tips
contain commits not reachable from a preserved remote ref or because they
contain uncommitted work:

- `MatcoreDSL-wt-m5-benchmark-evidence`: clean, one tip commit absent from
  `origin/main`.
- `MatcoreDSL-wt-m5-benchmark-evidence-delivery`: clean, two tip commits absent
  from `origin/main`.
- `MatcoreDSL-wt-m5-benchmark-evidence-final`: clean, three tip commits absent
  from `origin/main`.
- `MatcoreDSL-wt-m5-public-context-abi-hardening`: three modified files and
  three tip commits absent from `origin/main`.
- `MatcoreDSL-wt-windows-driver-coff`: clean, two tip commits absent from
  `origin/main`.

No attempt was made to infer whether these lines should eventually be
published, archived, or discarded.

## Preserved recovery and ambiguous material

The history-sanitation mirror, validation clone, pre-rewrite checkout and its
linked worktrees, dirty quarantine, pre-purge mirror, verified Git bundles,
and purge evidence remain preserved below the local MatcoreDSL archive and
recovery locations. They were not moved because the sanitation documentation
records their existing recovery paths.

The local OpenBLAS source archive and the Milestone 5 raw performance archive
were classified as intentional but ambiguous local evidence and retained.
Neither is tracked by this repository.

## Space and integrity result

- Registered worktrees reclaimed: `1,434,284,032` bytes.
- Generated-only review directories reclaimed: `1,327,104` bytes.
- Total reclaimed: `1,435,611,136` bytes (approximately 1.34 GiB).
- Registered worktrees after cleanup: six, including the canonical checkout.
- Canonical `main` remained clean at
  `951239f1bee5541a4cf5ad72fab2192de07cf89d`.
- `git fsck --full --strict`, `git diff --check`, and repository hygiene
  completed without an integrity failure.

