# Source peeling: complete hosted Debug qualification budget

This second, independently reviewed correction changes only the
`clang-21-debug-mlir` job timeout/comment, from 60 to 90 minutes. Every configure,
build, registration, test, assertion and per-test timeout remains unchanged,
including two-job builds and serial CTest. It is not a product-code correction,
performance claim or permission to relabel an incomplete run as passing.

At `d3460226d7f9358ac815e6f49c73a7a9cff8bbea`, both native Debug jobs reached
the one-hour job deadline. Their annotations explicitly identify that deadline;
neither log reports a failed individual test before cancellation:

- [PR job129616340141](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626689/job/129616340141)
  completed tests through #346 of 350, then was cancelled 75.452 seconds into
  #347, `driver.source_pattern_peel.phase_mutations`. That same test passed at
  frozen compiler `1ad7fe56` in 76.46 seconds (PR) and 80.68 seconds (push).
  The current job had already passed peeled connected-source execution in
  185.13 seconds. Cancellation before an earlier passing duration is not
  evidence of a hang, nor is the completed prefix a whole-suite pass.
- [Push job129616322713](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751621276/job/129616322713)
  completed tests through #343: 324 executed passes and 19 explicit AVX512
  skips. It was cancelled about 24.7 seconds into #344, peeled connected-source
  qualification. Earlier passing durations were 108.18/112.53 seconds, and the
  current PR job passed it in 185.13 seconds. It had not exceeded those known
  passing durations, much less its unchanged per-test timeout.

The source/compiler subtree at `d346022` is exactly
`07c0b72c91d837777f933484f3ea0a905ea563ad`, also the subtree at the frozen local
producer `1ad7fe56`. The intervening change was only the Release CI budget/report.
This establishes source-tree identity, not identical hosted binaries or hosts.
Distributed increases in completed test durations, rather than one stationary
test, support a job-budget correction. The 90-minute budget retains substantial
margin without changing test acceptance, parallelism or production behavior.

Retained raw logs under `build-source-peel-release` and the independent
`ci-debug-diagnosis-QAwoGf` / `ci-debug-review.EAEHLq` receipt directories.
The latter's eleven-file checksum manifest hashes to
`585fe5d9f4ba57f301205a6defc214b9c80fe1402af66ba02a4b5b7e01e99982`:

| Receipt | SHA256 |
| --- | --- |
| PR Debug timeout log | `c607178a04da11590acbe0d7f174ab31063e7305c4fbd31fc8ab3f899a4ec9b0` |
| Push Debug timeout log | `e9f7d6b0cf4f03dbf1baca7632740570e86e1ee496c41dba42fc0af7e4a69f8b` |

The original cancellations remain in history. No retry-until-green, omitted
test or local result substitutes for the fresh exact-head hosted gate. The
producer/SDK and P5 source/plan stay frozen at their recorded identities; timing
collection and merge remain blocked until the required qualification finishes.
