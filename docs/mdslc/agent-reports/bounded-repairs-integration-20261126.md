# Bounded repair integration — 2026-11-26

## Scope and provenance

Resumed saved repairs #110, #115 and #119, plus a demonstrated Windows
qualification prerequisite. No #108 timing experiment, performance policy,
compiler architecture, public API/ABI or execution-authority change.
Starting canonical main: `1c1221e89272eb2ad6ff058b8e26de4653f3a93d`.
At most two concurrent subagents; the explicitly selected reviewers were
`gpt-6.2-astra`, `ultra`. This is not a controlled model comparison.

| PR / boundary | Qualified pre-merge head | Canonical normal merge | Tree shared by head and merge |
| --- | --- | --- | --- |
| [#125](https://github.com/onlyxItachi/MatcoreDSL/pull/125), Windows prerequisite | `be8c59f804f559a66b9c6b2fb8042fde4f6f6023` | `26f4b43397c8358d182b2e62df97361d73f48280` | `bd805d5fad73bb360693b5d36d182eadadad8425` |
| [#124](https://github.com/onlyxItachi/MatcoreDSL/pull/124), exact GPU versions (#115) | `fa84e8895f15e38a611c3f0febb6e9e2bda3188a` | `b90540fdd84ebd1f2ac871642d56308787b5112a` | `db97a36d383bc5a0c23146e8f4153cbff7cd1a67` |
| [#126](https://github.com/onlyxItachi/MatcoreDSL/pull/126), nondestructive scratch roots (#110) | `1203dc45e97b59e78e0043797841e74e26eebe9e` | `e6338e72a2d6d3d70bd2ad2f5abca9b069edbf3b` | `49023a9247bc27d41808d02dabbd340fd2218bfd` |
| [#123](https://github.com/onlyxItachi/MatcoreDSL/pull/123), null contraction maps (#119) | `7cfb30ca7238bf80799fa387aa13417b11ad816d` | `30112dcbe9d7d3df69c563b5fd56a06cb6c0a829` | `8c25e04eff43d4b6db501d72058600554a3be1f2` |

Each merged head had 26/26 completed successful retained hosted checks, not
cancelled or pending checks. Normal merge parents preserve the original heads.
PR #125 parent main was the starting checkpoint; subsequent merges follow the
table order. Superseded/duplicate cancellations are separate, never passes.
Tree identity does not turn old performance artifacts into measurements of a
new revision. Windows CI built and uploaded distribution-candidate ZIPs;
no product release was promoted or published.

## Qualified changes and negative results

- **Windows:** select the already installed MSVC 14.44 family and verify/log its
  identity; retain pinned Clang 21.1.8. Explicitly refuse empty snapshot paths
  before platform-dependent filesystem conversion. Reports:
  [toolset](windows-toolset-qualification-v1.md),
  [toolset review](windows-toolset-independent-review-62-v1.md),
  [empty-path review](windows-empty-snapshot-independent-review-62-v1.md).
  Old main/PR configure failures and the first repair's 51-pass/1-fail Release
  result remain failures; no gate was waived.
- **#115:** exact leading llc/LLD/ptxas header parsing, duplicate/ambiguous record
  refusal and NVIDIA release consistency. The first patch's 81 passing controls
  missed four independent false-admission witnesses. Corrected source passed
  90 GPU controls, retained seven Clang controls, 24 independent additional
  decisions and three real installed banners. Reports:
  [corrective implementation](issue115-anchored-gpu-banners-v1.md),
  [independent review](issue115-final-independent-review-62-v1.md).
  Version parsing is qualification, not authentication against a lying executable;
  no new GPU execution or performance claim follows.
- **#110:** validate lexical path components, then compare canonical identities,
  require bounded ownership markers for existing containers, and create a new
  exclusive retained run child. Safe equivalent registered-root spellings remain
  supported; protected-path aliases cannot bypass the guard. No recursive reset
  of prior runs; existing unmarked roots are refused unchanged. The original
  double-leading-slash bypass and its correction remain in the
  [implementation](issue110-test-root-ownership-v1.md) and
  [independent review](issue110-independent-review-62-v1.md).
  This is not a hostile concurrent-filesystem or mount-alias sandbox.
- **#119:** after checking vector lengths, reject null affine-map handles before
  any context access. Preserve canonical topology, mixed-context rejection and
  valid zero-result DOT maps. Tests cover all seven nonempty null-slot subsets
  and malformed map/rank lengths through verification, encoding and structured
  checks. The historical guard-removal SIGSEGV witness and restored passing
  result remain in the [implementation report](issue119-null-contraction-maps-v1.md).
  Independent source review accepted the bounded internal verifier fix, not a
  demonstrated public-source exploit or new operation surface.

Current-head hosted evidence for #125:
[native PR](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45830648162),
[native push](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45830637467),
[ARM64](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45830648476),
[Windows PR](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45830648233),
[Windows push](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45830637473).
Windows PR: Release 51 pass/1 AVX-512 skip, Debug 32 pass/1 AVX-512 skip,
focused ASan 1 pass. Same-head push: Release 52 pass/0 skips, Debug 33 pass/0
skips, focused ASan 1 pass. Installed ABI, relocated consumer and distribution
checks passed. A skip on one worker is not physical qualification by that worker.

Current-head hosted evidence for #124:
[native PR](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45832479367),
[ARM64](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45832479578),
[Windows PR](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45832479639).
The complete PR check rollup was re-read as 26 successful checks before merge.

## Composed repair qualification

Both #110 PR #126 and #119 PR #123 merged only after their 26 retained jobs passed.
The normal composition commits are `1203dc45e97b59e78e0043797841e74e26eebe9e`
and `7cfb30ca7238bf80799fa387aa13417b11ad816d`, respectively. They preserve prior
heads `f0d0ac18f1b2a06c79b6b7b7e8fba0cb6dccd782` and
`bc8327b56934bc8848dab8280b06b9e7ca272697`. Independent read-only review found
no changed-file overlap and byte-identical accepted issue patches across composition.

At clean `7cfb30ca`, the existing coherent Clang/LLVM/MLIR 21.1.8 Release build
(OpenBLAS/GPU issuers off, regions on) reconfigured and rebuilt in five incremental
steps. Thirteen focused CTests passed, zero CTest skips, in 20.25 seconds;
the direct contraction executable reported 325 checks, zero failures.
The scratch suite's platform-specific inner skips are distinct from CTest status.
Raw local log: `mdslc-work/issue-hunt-20261112/build-null-maps-release/linear-stack-ctest-20261126.log`
under the owner's home; SHA-256
`3d84351bf458f117b143f46ba69925d2e116d3600663f2dc26982b47a98905d9`.
Three root invocation mistakes failed explicitly: a wrong direct test basename,
a missing required CMake script argument, and a nonexistent `/usr/local/bin/cmake`
passed to the scratch suite (six failing subcases, two platform skips). None was
counted as a pass. The registered suite uses `/usr/bin/cmake` and `/usr/bin/python3`;
no test/source expectation was weakened to accommodate the invocation mistakes.
The two registered CTests were then repeated with their configured executables:
both passed (90 GPU plus seven retained Clang controls; scratch 25 passed and
two native-Windows-only skips on Linux).

The same clean composed producer also passed a real installed C17 compile/link/run
probe authenticating all 15 public exports, and a real relocated consumer
configure/build/run/rebuild/no-op/semantic-capability probe. Existing owned scratch
containers gained new retained children `c17/run-nd9fmaxe` and
`consumer/run-z6dmsw3m` under `mdslc-work/issue-hunt-20261112/package-probes-20261126`.
Earlier children were not reset or removed. This is same-tree package evidence,
unlike the earlier cross-tree harness qualification.

During push/base reconciliation, duplicate hosted runs appeared. Only the older duplicates
and superseded-head native runs were cancelled, never treated as passes. Retained
current PR native runs are #126 `45846397089` and #123 `45846399430`; their push
runs remain `45846387015` and `45846391251`. Independent read-only audit of the
retained hygiene checkout logs confirmed synthetic merge `b5ec0b6a94034303bc3cd8cf6a65e3b3f975a61f`
for #126 and `8bb4822bc49ae9d172d182badd7e34f6eb493df1` for #123, each with the same
file tree as its respective PR head. This does not claim a checkout identity for
jobs whose logs were not yet available.

The superseded PR #125 canonical-merge push native run `45846224672` was also
cancelled after main advanced to PR #124. Its Windows run `45846224364` had
already succeeded. This does not alter the 26 successful pre-merge checks of
PR #125; neither cancelled post-merge work nor a later tree qualifies an earlier
revision retroactively. The latest canonical push remains separate evidence.

Fresh Windows push/PR runs for composed #123 (`45846391018`, `45846399525`)
both passed: Release 53/53, Debug 33/33, focused ASan 1/1, zero CTest skips.
The #126 push/PR runs (`45846387061`, `45846396830`) passed the same counts.
These workflows also passed
installed ABI, relocated-consumer, artifact and distribution checks. Inner Python
platform skips are not represented by those CTest skip totals. Windows uses the
existing MLIR-disabled platform lane: these passes do not execute the null-map
MLIR verifier or establish generated-region Windows support.

Coverage routing is explicit: the new null-handle/malformed-count unit cases run
in local Release and in the full Linux MLIR-enabled Release/Debug CTest suites.
The current focused ASan+UBSan regex and ARM64 closed-source qualification regex
do **not** select `mlir.structured.contraction-model-v1`. Those lanes check their
existing downstream/runtime scopes, not direct sanitizer or ARM execution of the
new null-handle cases. No such direct qualification, or MLIR 22 qualification,
is claimed. The issue's bounded verifier acceptance does not require a new
sanitizer policy or changed ARM qualification scope.

Both fresh ARM64 runs ([#126](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846396899),
[#123](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846399622)) passed
all three schedules. Each scalar/output-tiled lane executed 130 closed-source
tests; each row-contiguous lane executed 132, with no skips in those scopes.
Every lane also passed the separate 44 private-peel and four source-peel gates.
This is native AArch64 correctness qualification, not a performance measurement.

The current-head Linux row-contiguous Release PR jobs also completed successfully:
[#126 job 140001074718](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846397089/job/140001074718)
and [#123 job 140001081328](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846399430/job/140001081328).
Each ran 352/352 CTests with no CTest-level failures/skips in the logs, including the
direct `mlir.structured.contraction-model-v1` unit. Their remaining job steps
also succeeded.

The current-head Debug jobs also passed 352/352 CTests without CTest-level skips:
[#126 job 140001074590](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846397089/job/140001074590)
and [#123 job 140001081435](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846399430/job/140001081435).
The #123 log explicitly includes the updated contraction-model test. Focused
ASan+UBSan jobs [140001074149](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846397089/job/140001074149)
and [140001081560](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846399430/job/140001081560)
each passed their separate 236/2/44/4 gates with no CTest-level skip markers.
The direct unit/ASan selection distinction above still applies.

The last #123 MLIR-enabled, OpenBLAS-disabled PR Release job
[140001081802](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846399430/job/140001081802)
passed 355/355 CTests with no CTest-level skips. The same-head push Debug job
[140001057095](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45846391251/job/140001057095)
had **333 passed, 19 AVX-512 skips, zero failures**: one packed-runtime, 13 strict-ISA
and five output-pattern cases were skipped on that worker. Its updated contraction
unit passed. The separate PR Debug worker executed all 352; no skipped test is
counted as physical qualification by the worker that skipped it.

Final independent gate audit reauthenticated #123's unchanged head, all eight
retained workflows / 26 jobs, zero pending/failed/cancelled retained jobs and
zero unresolved review threads. The cached test merge still used old base
`1203dc45`; retargeted main `e6338e72` had exactly the same base tree. Both the
tested tree and final canonical merge equal the accepted head tree in the table.
Source/composition and final gate reviews were independent agent reviews;
there were no submitted GitHub approval reviews. Root separately repeated the
live gate checks before each normal merge. No forced merge or history rewrite.

The prior full 346/346 suite belongs to `0fd5f767dd9ddcf15a19a4223897edd4279eb41e`;
the prior post-Windows 11/11 suite belongs to `d011979041f058417bda2452787550ef791c344f`.
Neither qualifies the new combined heads. See the original
[null-map review](issue119-independent-review-62-v1.md) and
[scratch-package qualification](issue110-local-package-qualification-20261126.md).

## Preserved project frontier

Exactly one next product boundary remains **P6 / #108 timing-method
qualification**, not a performance optimization or policy. P1/P5 remain
inconclusive; #15 remains partial/open and #20 design-only/open. PR #109's
separate historical P5 documentation and PR #84's SPIR-V work were not modified
or merged. No cleanup, system toolchain change or release claim was made.
