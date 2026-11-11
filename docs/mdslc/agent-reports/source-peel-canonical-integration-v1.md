# Source-peel canonical integration and evidence reconciliation

Recorded 2026-11-11 after the source/product merge; this is not a new build,
benchmark collection, target qualification or performance claim.

## Normal merges and preserved preheads

| PR | Reviewed head | Actual canonical merge | Merge tree |
| --- | --- | --- | --- |
| [95](https://github.com/onlyxItachi/MatcoreDSL/pull/95) | `633664d3debdcb86c778038d338ad251f1e74ee9` | `c903351f7009848a00c83498e50183135e9c6404` | `da722e0fff00de2891257ec8115cd2948ee5b682` |
| [105](https://github.com/onlyxItachi/MatcoreDSL/pull/105) | `cf154deacea2bddb3240660c1af27ff4ffbc4042` | `6d30a204aa6f7224b736c2035de15ce20b0e6b3d` | `2aab2289ccfa246c5f921b2e741263fc96217e48` |

PR95's first parent is `1f7c491f230251d04d42b5d4a0af5eaf29bcb685` (PR103).
PR105's first parent is the PR95 merge above; their second parents are the
reviewed heads. Both PRs had exactly 26 successful hosted checks and independent
review before normal merge. No squash, history rewrite or forced update occurred.
The clean canonical checkout was fast-forwarded; frozen producer/SDK `1ad7fe56...`
and unrelated worktrees were not changed.

An independent isolated-object `git merge-tree` audit predicted both exact trees
before mutation. Its record is `/var/tmp/mdslc-pr95-pr105-merge-audit-wCicsR/audit.md`,
SHA256 `16e6f2d557905f5f97a603c6fe65e1215b5041525995cb8523ab3fbb7e7b7d3d`.
Actual PR105 production libraries/tools/headers/tests/build logic/workflows equal
qualified `cf154de`; the sole compiler difference is P1 README navigation.
Thus its complete compiler tree is **not** identical to the frozen producer tree.
Tree equivalence does not establish binary reproducibility or rerun qualification.

## PR95 infrastructure recovery, not a test waiver

Three original jobs stalled acquiring Ubuntu APT indices, before configuration,
build or tests. Their cancelled results remain cancelled. One targeted same-head
retry wave changed no package tuple, workflow command, test or budget.

| Actual retry job | Executed results | Additional proof |
| --- | --- | --- |
| Row Release `129673685087` | 346 passed, zero skipped/failed | Production `BUILD_TESTING=OFF` build/install passed |
| Scalar Release `129673685531` | Separate 1/1 readiness test, then 349/349; zero skipped/failed | Production `BUILD_TESTING=OFF` build/install passed |
| Debug `129673750356` | 327 passed +19 AVX512 skips /346, zero failed | No production-proof scope |

These are 1,042 result entries, not unique tests. The other 23 successes were
not rerun: their original run/name/start/end timestamps match. GitHub regenerated
15 check IDs during retry bookkeeping; those IDs are not new executions.
Push jobs checked out exact `633664d3`; Debug checked out synthetic merge
`290f0e756e2c5216fbb90ed31df6bfc64ec12380`, with the same tested tree.

Independent receipts remain under
`build-source-peel-release/hosted95-633-install-infra-C7cdRa/recovery-attempt2/`
in the local campaign workspace. SHA256 identities:

- `FINAL_CI_AUDIT.md`: `0f9218b721288727649eee215c45591be49d0ec78129195b88208ad336582e54`.
- `pr95-final26.json`: `8e352f086f8a69734806dfa0c6c701f5a9b84ce19451dee6b60fda5d899d9cd9`.
- `final-ci-recount.json`: `e15796f37138c3487c604c3fe24bc9d4102e59b60fb325fd71b66fd790e9e1e9`.
- `final-ci-receipt-identities.json`: `4ad0a92d9d9b8d50f2cbc9a3d9c995c28218d1885535358831d02e28b4a97443`.

## P5 integration boundaries

[PR107](https://github.com/onlyxItachi/MatcoreDSL/pull/107) is retargeted to main
and includes the actual canonical merges through local reconciliation commit
`c5188f21641bd4be5dd8e5a71032a4f3a70b91d6`. Its [P5 result](../../performance/cpu/source-pattern-p5-2026-11-10.md)
retains the executed producer `1ad7fe56...`, qualified-but-unmerged source head
`cf154de...`, harness head `ea7accfe...`, and frozen helper/source bytes `b63ed320...`.
Both source and harness had 26 successful checks before that one collection.
No later merge relabels the artifacts. All 18 timing ratios remain null.

The reconciled branch changes no production code, workflow, P1/P5 protocol or
helper bytes relative to the qualified harness head, except the P1 README's
documentation-only navigation. Focused existing synthetic CTests passed 2/2;
the repository hygiene check and diff check passed. These are synthetic/audit
checks, not a rerun of native execution or the performance experiment. Final
candidate hosted status is recorded on PR107, separately from its earlier head.
The retained final focused invocation is
`/var/tmp/mdslc-final-integration-6FjXBH5y/focused-synthetic.log`, SHA256
`9032b81af6198665ae54e4199c6d6ff6b5e2cf23ef201c9aa70020e9029c8875`;
its `focused-synthetic.xml` is
`e2e01c9790fa6baf358eb1d457408e4fa434445b3f3b3d19cdbc145bb14f05cc`.
It reports both tests executed, zero skips/failures, 5.64 seconds.

[CURRENT_STATE.md](../CURRENT_STATE.md) records the source/product merge and
one next boundary: [Issue108](https://github.com/onlyxItachi/MatcoreDSL/issues/108),
prospective measurement-method qualification. The rejected sleep-only proposal
could accept an observer returning `20 ms + sleep phase` while ignoring CPU work;
the revised gate requires verified CPU-work sensitivity and identical-binary
null/order controls. No method implementation or new collection is claimed.
