# ADR 0005: Git history artifact sanitization

- Status: Accepted
- Date: 2026-07-22

## Context

Repository-hygiene checks removed generated output from the current `main`
tree, but older commits and checkpoint refs still made build directories,
compiler outputs, cache metadata, raw benchmark logs, profiler databases, and
an agent-local context file reachable. Keeping those objects in user-controlled
branches and tags made clones larger and allowed obsolete machine-local output
to reappear during merges.

The contaminated history also contains the complete legacy Python/JIT/MLIR
implementation and the standalone MDSLC compiler lineage. A cleanup that
discarded or flattened that source history would be worse than retaining the
artifacts.

## Decision

Rewrite every user-controlled branch and checkpoint tag with `git filter-repo`
using an independently reviewed manifest of exact paths and blob IDs.

The migration:

- removes only 56 proven generated paths representing 62 unique blobs;
- retains all 292 source-history commits, including commits that become empty;
- retains merge topology and commit messages, authors, committers, and dates;
- preserves every non-purge tree entry byte-for-byte at every mapped commit;
- recreates both MDSLC checkpoint tags on their mapped commits;
- adds `matcoredsl-legacy-final-v1` at the mapped final legacy-only commit;
- preserves all 10 existing remote branches, atomically updating the nine
  rewritten branches and two rewritten tags with exact old-object leases; and
- preserves 21 local-only development heads in a sanitized external bundle
  without publishing them as new remote branches.

The manually written benchmark follow-up
`benchmark_reports/complete_20260423_182039/notes.md` is preserved
conservatively. It is engineering analysis, is not emitted by the benchmark
harness, and has no sanitized equivalent. Intentional fixtures, goldens, ADRs,
MDSLC reports, legacy implementation source, profiling scripts, and normalized
performance records are likewise preserved.

## Hard invariants

The pre-rewrite `main` tree and rewritten source checkpoint tree are both:

```text
90e04d986d1b9c2621d3a77d69fa85624663c0b3
```

All approved purge paths and blobs are unreachable from every rewritten head
and tag. No additional blob is removed. The working source tree changes only in
the later documentation commit containing this ADR and the migration report.

## Consequences

Commit and annotated-tag object IDs change wherever a purged object was in
ancestry. Existing clones must not push their pre-rewrite refs. A fresh clone is
the supported migration path.

Seven rewritten commits had signed commit objects. Their old `gpgsig` headers
could not remain valid after the signed payload changed and were removed by
`git filter-repo`; their source, metadata, messages, and mapped topology are
otherwise unchanged. Existing annotated tag messages and tagger metadata are
preserved.

GitHub may retain unreachable objects temporarily in internal pull-request
refs, caches, or backend storage. This decision guarantees only that no
user-controlled branch or tag retains the old objects; it does not claim that
GitHub has physically garbage-collected them.

Recovery mirrors and bundles remain offline and must never be pushed into the
active repository.

## Validation

Validation included exhaustive comparison of all 292 old/new commit pairs,
Git object integrity, path/blob reachability, ancestry, tag, backup and secret
checks, fresh Release and Debug standalone builds, sanitizer checks, native
GEMM artifact execution, install/package/consumer tests, a fresh network clone,
the legacy frontend contract, hosted CI, and an independent adversarial review.

The detailed evidence and migration map are in
`docs/repository/history-sanitization-report.md`.
