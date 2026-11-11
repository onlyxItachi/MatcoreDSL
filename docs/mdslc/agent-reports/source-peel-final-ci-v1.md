# PR107 final hosted-CI and merge receipt

Recorded 2026-11-11. This report documents the completed
[PR107](https://github.com/onlyxItachi/MatcoreDSL/pull/107) boundary, not the CI or
merge of its own documentation follow-up.

## Exact identities and final gate

| Role | Commit |
| --- | --- |
| Reviewed final PR107 head | `9476e8b90836cb31c665f56f8d2fcb0c8da38b0f` |
| Actual synthetic PR checkout in the selected CI logs | `85fce0b9a38ff837c0f0b94fe823ac10ded3310e` |
| Actual normal canonical PR107 merge | `1c1221e89272eb2ad6ff058b8e26de4653f3a93d` |

All three bind tree `7c4c5fea752188b1c110ba53a4e1361439abf524`.
The synthetic CI checkout and actual canonical merge are distinct commits with
the same parents: base `6d30a204aa6f7224b736c2035de15ce20b0e6b3d`, then the reviewed
final head. Actual checkout-log output and retained Git commit API records bind
the CI identity; local Git independently confirms the canonical merge identity.

The final 2026-11-11 20:08:21 UTC snapshot contains exactly **26 completed SUCCESS
checks**, independently confirmed by the PR rollup and commit check-run API list.
All eight workflow runs completed successfully on attempt 1. No job retry,
cancellation, failed job or dependency-install stall over ten minutes occurred
during this fresh-head watch. Earlier `ea7accfe94fbb05a79c5135f074e25390b0e147d`
qualification remains historical and was not reused as the final-head gate.

## Actual selected PR-side scopes

Each row names retained completed-job logs, not inferred coverage from job names.
Addends denote separately executed CTest invocations. Registration-only output is
excluded; these are result entries, not a sum of unique tests or an all-26 test
count. Every listed job checked out the synthetic merge and exact tree above.

| PR-side job / workflow run | Executed CTest passes | CTest skipped / failed |
| --- | --- | --- |
| Required-OpenBLAS MLIR Release `130741434401` / `43083383300` | 353 | 0 / 0 |
| MLIR Debug `130741433909` / `43083383300` | 332 of 351 entries | 19 / 0 |
| ASan+UBSan `130741433707` / `43083383300` | 236 + 2 + 44 + 4 | 0 / 0 per invocation |
| ARM scalar `130741434023` / `43083383249` | 130 + 44 + 4 | 0 / 0 per invocation |
| ARM row-contiguous `130741433595` / `43083383249` | 132 + 44 + 4 | 0 / 0 per invocation |
| ARM output-tiled `130741434065` / `43083383249` | 130 + 44 + 4 | 0 / 0 per invocation |
| Windows x64 `130741432970` / `43083383253` | Release 52; focused Debug 33; focused AddressSanitizer runtime 1 | 0 / 0 per invocation |
| Focused runtime TSan `130741433211` / `43083383300` | 4 | 0 / 0 |

Debug's 19 AVX-512-labelled skips are individually listed in the recount; they
are not executed passes or proof of that ISA's runtime support. Required Release's
`Prove opt-in production build without test targets` step was conditionally
**skipped**, not passed: that proof requires MLIR ON and OpenBLAS OFF. Debug, ASan,
ARM, Windows and TSan have no such named production-proof step in these scopes.

ARM evidence is normal hosted native qualification, not ARM source-ASan.
Windows uses coherent clang-cl x64 21.1.8 with MLIR/OpenBLAS disabled; its bounded
runtime AddressSanitizer/native/package and relocated-consumer evidence does not
establish Windows source-peel, region-DSO support or whole-compiler sanitization.
TSan executes four focused runtime tests with native frontend, MLIR and OpenBLAS
disabled; it establishes neither source-peel sanitizer coverage nor universal
race freedom. No broader platform or performance qualification follows.

## Exact local receipt identities

The six Release/Debug/ASan/ARM logs, completed metadata and per-job scopes remain
under `/var/tmp/mdslc-pr107-final-ci.foY56J/`. Its final manifest's 194 payload
hashes were independently checked. SHA256 values below are relative to that root:

| Receipt | SHA256 |
| --- | --- |
| `pr107-final26.json` | `ecfd2c52bb6fabcc1b2502164bc151c311019285cfc0618d6e23254d539802c2` |
| `head-check-runs.final26.json` | `8230a4b5ac294a14add9a5044529b2a168a2a8608afee831eb00849af97a5fe7` |
| `final-ci-recount.json` | `c6da8cc12be605eebaec58a9b16cd706ea00d7de088bec6db2590c775c0b6f29` |
| `final-ci-receipt-identities.json` | `130b2b1a11337c7dead4426b95c9614e2f02b14459fb2574ba9fe7ff4db64337` |

Windows/TSan receipts remain under `/var/tmp/mdslc-pr107-platform-ci-26Fjn5/`:

| Receipt | SHA256 |
| --- | --- |
| `PLATFORM_CI_AUDIT.md` | `8c11cac245b791bf9a850595a5dd7bc3e614869584606beae5760168bca6d4da` |
| `platform-recount.json` | `630fdcacb8a48c3a423be9afa67d797f47a5919695719f6435a937106a437044` |
| `SHA256SUMS` | `8e53c78e0217e433816464841de3623a5494073b68c456f0d6e22741f8be7a76` |

The platform report hash was independently recomputed and matched its manifest;
all 14 platform manifest entries verified. That audit completed at 19:32 UTC and
cleared only its two selected job scopes, not the later all-26 final gate. Both receipt roots
are preserved local evidence, not a published backup or a locally re-executed CI
build. No GitHub mutation, package/toolchain change, local native/build/test or
benchmark execution was performed for this report.

## P5 remains unchanged

The [P5 result](../../performance/cpu/source-pattern-p5-2026-11-10.md) remains
`INCONCLUSIVE_TIMING`: all 18 comparison ratios are null. Its executed producer
and installed SDK remain `1ad7fe56b5fcb10efe37e523206228e9ef303f6a`; the collection's
source/harness/artifact identities are not relabelled by PR107's later merge.
This report changes no frozen capsule, sample, protocol, estimator or performance
claim and records no new collection.
