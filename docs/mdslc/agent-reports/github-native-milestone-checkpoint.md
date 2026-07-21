# GitHub checkpoint: native LibTooling CPU milestone

Date: 2026-07-21

Repository: `onlyxItachi/MatcoreDSL`

## Verified immutable milestone

- Completed branch: `mdslc/native-libtooling-v1`
- Reviewed commit: `c025df534d11d1bc08285a174f2cd357aecadb0e`
- Bootstrap branch: `mdslc/bootstrap-v0`
- Bootstrap commit: `3e3fa5b2d1990e1c37870f8b2096fbda6128716b`
- Closest remote ancestral base: `feature/device-resident-tensors`
- Base commit: `351075e4d8af1880330b7c0474d701ca76776dfa`

`origin/feature/device-resident-tensors` is an ancestor of the native branch.
`origin/main` is not an ancestor, so the draft PR deliberately targets the
feature branch rather than `main`.

The completed integration worktree was clean. No merge, rebase, cherry-pick,
revert, or sequencer operation was active. The final evidence and independent
review commits were present, and all 28 native milestone commits were reachable
from the bootstrap head.

## Published refs

Remote branch SHAs:

```text
3e3fa5b2d1990e1c37870f8b2096fbda6128716b  refs/heads/mdslc/bootstrap-v0
c025df534d11d1bc08285a174f2cd357aecadb0e  refs/heads/mdslc/native-libtooling-v1
c025df534d11d1bc08285a174f2cd357aecadb0e  refs/heads/mdslc/matcore-ir-v1-cpu-planner (initial checkpoint)
```

Annotated tag:

```text
4bb7be7be251e2f85cf662b1651351a5a5a7787c  refs/tags/mdslc-native-cpu-proof-v1
c025df534d11d1bc08285a174f2cd357aecadb0e  refs/tags/mdslc-native-cpu-proof-v1^{}
```

The GitHub release-by-tag endpoint returned HTTP 404, confirming that the tag
has no GitHub Release.

## Draft pull request and tracker

- Draft PR: https://github.com/onlyxItachi/MatcoreDSL/pull/3
- PR head: `mdslc/native-libtooling-v1`
- PR base: `feature/device-resident-tensors`
- PR state: open, draft, unmerged, mergeable
- Tracker issue: https://github.com/onlyxItachi/MatcoreDSL/issues/4
- Issue state: open

Duplicate searches found no earlier PR from the native head and no equivalent
open milestone tracker. No existing PR was modified or retargeted. The issue
was created without labels rather than inventing a new taxonomy.

## Next development branch

- Branch: `mdslc/matcore-ir-v1-cpu-planner`
- Dedicated worktree:
  `/home/hamza-usta/MatcoreDSL-wt-ir-v1-cpu-planner`
- Starting commit: `c025df534d11d1bc08285a174f2cd357aecadb0e`
- Commits between the milestone tag and initial branch checkpoint: zero

This report is the first branch-only documentation commit. Its parent is the
immutable tagged milestone; it does not change the completed native branch or
tag.

## Commands and operations

```sh
git status --short --branch
git branch --show-current
git rev-parse HEAD
git log --oneline --decorate -12
git diff --check
git remote -v
git worktree list

git fetch origin --prune --tags
git remote show origin
git branch -r
git log --oneline --decorate --all --graph -40
git merge-base --is-ancestor \
  origin/feature/device-resident-tensors mdslc/native-libtooling-v1
git merge-base --is-ancestor origin/main mdslc/native-libtooling-v1

git ls-remote --heads origin \
  mdslc/bootstrap-v0 mdslc/native-libtooling-v1
git push -u origin mdslc/bootstrap-v0
git push -u origin mdslc/native-libtooling-v1

git show --no-patch --decorate \
  c025df534d11d1bc08285a174f2cd357aecadb0e
git tag -a mdslc-native-cpu-proof-v1 \
  c025df534d11d1bc08285a174f2cd357aecadb0e \
  -m "MDSLC native Clang LibTooling CPU architecture proof"
git push origin refs/tags/mdslc-native-cpu-proof-v1

git worktree add ../MatcoreDSL-wt-ir-v1-cpu-planner \
  -b mdslc/matcore-ir-v1-cpu-planner \
  c025df534d11d1bc08285a174f2cd357aecadb0e
git push -u origin mdslc/matcore-ir-v1-cpu-planner
```

The connected GitHub app was used to search for duplicates, create and verify
draft PR 3, and create and verify issue 4.

## Safety and deviations

- No force push, force-with-lease, reset, rebase, merge, tag replacement,
  auto-merge, PR retarget, or GitHub Release occurred.
- Git SSH authentication worked for fetch and pushes.
- The standalone `gh` CLI token was stale, so connector-backed GitHub actions
  were used for PR/issue reads and writes. The connector authenticated as
  `onlyxItachi` with repository admin/push access.
- Two zero-byte untracked review-capture files named `stdout` and `stderr` were
  removed from the old native final-review worktree. No tracked or user-owned
  content was removed.
- The original `feature/device-resident-tensors` checkout retains its
  pre-existing user-owned dirty state and was not modified.

## GitHub permissions and CI state

The connected GitHub app authenticated as repository owner `onlyxItachi` with
admin/push permission. Git SSH fetch and push also succeeded. The stale local
`gh` token did not block any requested checkpoint operation.

Draft PR 3 triggered workflow run `29839161544`. Its single `build-and-test`
job failed during the legacy root build, before tests, at
`src/gpu_runtime_symbols.cpp`: the Ubuntu 24.04 runner reported unknown CUDA
driver types `CUstream`, `CUmodule`, and `CUfunction`. The workflow configures
the root Python/nanobind/MLIR 18 target; it does not build the standalone
Clang-21 `compiler/` project validated by this milestone.

This is a visible pre-existing legacy CI blocker. It was not hidden or
represented as a native MDSLC failure, and no CUDA/legacy production change was
made during this checkpoint task. The PR remains draft, unmerged, and
mergeable while the CI conclusion is failure.
