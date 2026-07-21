# GitHub mainline integration checkpoint

Date: 2026-07-21

## Published objects

- Branch: `mdslc/mainline-integration-v2`
- First published branch tip:
  `135cf4440975f8c11453e7fc9ff703cea7fa8798`
- Draft pull request: [#6](https://github.com/onlyxItachi/MatcoreDSL/pull/6)
- Pull request head/base: `mdslc/mainline-integration-v2` -> `main`
- Pull request title: `MDSLC: consolidate compiler lineage on main`
- GitHub milestone:
  [#1, MDSLC Milestone 3: Mainline consolidation and compatibility hardening](https://github.com/onlyxItachi/MatcoreDSL/milestone/1)

The pull request was created as a draft and was mergeable when opened. The
initial hosted checks were queued:

- `ci / build-and-test`
- `mdslc-native / Clang 21.1.8 standalone Release`

The milestone was attached through the documented Issues REST endpoint after
the installed `gh pr edit` GraphQL path failed on the deprecated Projects
Classic field. No undocumented API or force operation was used.

## Safety checks

Before publication:

```sh
git fetch origin --prune --tags
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote --heads origin mdslc/mainline-integration-v2
gh pr list --state all --head mdslc/mainline-integration-v2 ...
gh api 'repos/onlyxItachi/MatcoreDSL/milestones?state=all&per_page=100'
```

The remote integration branch, PR, and milestone did not previously exist.
`origin/main` remained at
`7d2f8475ce3148658743ebc6a58be808a7b36423`. The new branch was pushed as a
normal new ref; no force push occurred.

No tag or GitHub Release was created at this checkpoint. No existing pull
request was edited, closed, retargeted, or merged. Historical feature and
MDSLC branches remain available.

This report commit is intentionally the next branch tip after the first
published SHA. Hosted checks must run against the updated tip before the draft
may become ready for review or merge.

