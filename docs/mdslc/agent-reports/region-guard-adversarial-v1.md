# Independent adversarial review: region guard ledger v1

Reviewed immutable integration `11f99037a1a5e431e526fda1bb17bc78016898ac`,
diff from docs-only canonical `c01e3851158fe87f3ace4fd5031d432ebcc483a4`;
engineering base `63d642d9e4eb4183acf58a4391e1c668e734b9f0`. MLIR implementation
source head `b7171fafa8b385d0199abf4bb2fdf4409e68b8ed`. The integration worktree
was clean during this read-only review.

## Disposition and coverage

**ACCEPT** the code boundary; no remaining implementation blocker identified.
This is independent source/diff review, not independent execution evidence.
No tests/builds/GitHub writes were performed by this reviewer for this
checkpoint. Owner-reported focused results are not relabeled as independent
runs; final integrated regressions belong in the main checkpoint record.

Reviewed the complete changed compiler code, MLIR and runtime oracle tests,
CLI changes, CMake/ASan test registration, and architectural documentation.
Cross-checked unchanged native admission/source pairing, adapter generation,
`validate_gemm_v0`, plan v1, one-shot execution, FP decoding, OpenBLAS post-call
checks, and current buffer/vector/CPU consumers. No runtime/provider/frontend
implementation or public ABI changed.

## Review findings resolved

1. The existing region binding `argument` used `IntegerAttr::getInt()` without
   constraining bit width, before source pairing. An upstream-valid i128
   high-bit dictionary value could reach an unsafe narrowing/assertion path.
   `MatcoreTwoGemmRegion.cpp` now requires exact signless i64 before conversion.
   Tests mutate argument and snapshot with wide, wrong-width, negative and
   noninteger values; ledger/source alignment receives analogous controls.
   This report does not claim an independently reproduced pre-fix crash.
2. Isolated site, stage and descriptor mutations avoid allowing one copied
   ledger failure to stand in for all binding dimensions. Both output/lhs and
   output/rhs overlap obligations are independently removed in tests; equal
   input descriptors retain distinct operand-role requirements.
3. Report snapshots use `memcpy` to avoid aggregate-padding-copy assumptions.
   The capacity canary assertion claims unchanged elements, not that canaries
   alone prove absence of reads. This is test precision, not a runtime defect.

## Mechanical falsification ledger

| Requirements | Reviewed controls and limits |
| --- | --- |
| L1–L2 | Closed predicate/evidence/frontier/subject vocabulary and exact dictionary fields; unknown, extra, missing, duplicated, promoted and invented executed/discharged records reject. Exact expected-catalog comparison prevents known-but-wrong classes from passing. |
| L3 | Source-required natural alignment, dimension equations, both output/input overlap roles and legal equal/partially overlapping inputs are retained. No A/B noalias or SIMD alignment inference is introduced. |
| L4 | Independent site/binding/stage mutations and copied ledgers reject; existing order-chain adversaries retain guard1 after commit0. |
| L5 | A coordinated source-alignment change with a regenerated ledger explicitly passes standalone verification but fails sealed-source pairing. This distinguishes internal consistency from authentication. |
| L6 | Provider completion and possible post-write failure remain execution/return obligations. Promotion to pre-compute predicates rejects; the no-rollback commit contract remains enforced. No provider fault-injection or general retry proof is claimed. |
| L7 | Existing named-to-generic, canonicalization/CSE, observable-write retention and fresh-context parsed/source-paired positive controls remain active. |
| L8–L9 | Real plan-v1 queries exercise the descriptor validator. Source inspection establishes that the API accesses descriptor records, not tensor elements; canaries check unchanged arrays. Three simultaneously live 64-aligned one-float arrays safely establish distinct 64-byte numeric claimed intervals while actual typed capacity remains one element. They are never sent to execution. |
| L10–L11 | Scoped actual FTZ yields incompatible FP inspection while planning can succeed; a null output descriptor retains failure priority over FP on the old entry point. RAII restores MXCSR exactly. Synthetic controls reject incompatible rounding/masks/subnormal handling without inventing sticky-status or x87-precision requirements. |
| L12 | CLI publishes two bound inspection ledgers while refusing rewrite/executable options. CPU lowering rejects forged region authority/candidate labels. New old-route null-input/output-overlap failures preserve the first successful C write in both existing pipelines; they do not execute region IR. |

Ledger-only MLIR mutations leave the sealed function source contract unchanged.
Their paired-verifier rejection therefore cannot be explained merely by a
changed source fingerprint: region ledger checks themselves are necessary.
Tests requiring valid IR explicitly run upstream verification first. C++ and
Python checks survive Release/NDEBUG or optimized Python; they are not disabled
assertions.

## Ownership and remaining authority

The codec decodes only a finite vocabulary, derives obligations without runtime
or provider queries, and compares with the complete regenerated catalog. Native
evidence remains source authority. Representation tags do not establish
physical accessibility, capacity, writable allocation, lifetime, alignment or
FP state. Frontiers are obligation scopes, not a reordered schedule: one-shot
discovery/provider work can precede its final FP check, and OpenBLAS can fail
postconditions after output mutation.

No generated region execution, bufferization legality, candidate artifact
authentication, automatic fallback/retry, performance, zero-copy, new operation
authority or toolchain portability result follows from this patch. Issue #15
remains incomplete; Issue #20 remains separate.

## Exactly one next justified boundary

Extend this same inspection route to mirrored RHS-only dependence,
`C=A*B; E=D*C`, in a separate checkpoint. Existing bindings already carry both
roles; lhs-only forwarding is deliberate boundedness, not a correctness bug.
Prove asymmetric/non-square and noncommuting examples, late reads when the
other operand aliases C bytes, independently bound guard roles, and
first-success/second-failure behavior.

Preserve the existing `E=C*C` subset with lhs-priority forwarding and a guarded
late rhs read; do not introduce XOR admission. Keep new dual carried SSA edges,
general DAGs, fusion, region bufferization and generated execution excluded.
Deeper consumers accept exact single-site specimens and are not automatically
unlocked by this ledger.
