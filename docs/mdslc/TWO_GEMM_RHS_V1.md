# Mirrored two-GEMM inspection dependency v1

Engineering base: `8a825350ae8e3f24e55bc8d050375a0b2c0da3dd`
([PR #31](https://github.com/onlyxItachi/MatcoreDSL/pull/31)).
Local integration code/test checkpoint:
`33423177da947bc40d43680a33a6501a31d14736`.
The documentation-only main checkpoint
`e1045988577eeaee4c420d637a3bd6b38dcb3660` (PR #32) is reconciled before
publication; it changes no compiler or test code.

This implements the single next boundary recorded by the
[guard-ledger checkpoint](REGION_GUARD_LEDGER_V1.md). It changes which second-call
input can consume the first committed value, not the storage or execution model.

## Source evidence and accepted design

Observed at the base: native sealed bindings already preserve both input roles.
The established execution routes compute `C=A*B; E=D*C` correctly, while region
admission rejects it because only the second lhs is recognized as carried.
The existing `E=C*C` control is already admitted; an exclusive-or admission
rule would regress it. Baseline experiments are retained locally at
`/tmp/mdslc-rhs-baseline.OrZrRY4Z`; they use the PR #31 compiler code and only
the existing source execution routes, not generated-region execution.

Architectural consequence: derive the carried role from authenticated
descriptor identity, preserving mathematical operand order. No new frontend
DTO role, certificate family, operation, public contract or guard predicate
is required.

```text
C = A * B; E = C * D  -> E = C_post * read_after_guard1(D)
C = A * B; E = D * C  -> E = read_after_guard1(D) * C_post
C = A * B; E = C * C  -> E = C_post * read_after_guard1(C)
```

The last line retains lhs precedence and the separate late rhs read. Dual SSA
forwarding is not admitted by this representation; this is not a universal
claim that every equivalent alternative is semantically illegal.

### Files and invariants

- [Native admission](../../compiler/lib/frontend/native_frontend.cpp) accepts
  first-output identity in either second input. Direct declaration
  authentication, adjacency, references, host/control barriers, distinct
  outputs and descriptor self-alias rejection stay unchanged.
- [Region derivation and paired verifier](../../compiler/lib/mlir/MatcoreTwoGemmRegion.cpp)
  derive that role from bound descriptors. Exact signless i64 argument and
  snapshot-stage fields are checked before role selection. The committed
  tensor must retain the selected input type.
- The other input is read only after guard1 and commit0. Descriptor inequality
  remains no proof of physical disjointness. Rows come from current lhs axis 0,
  columns from current rhs axis 1; dynamic types do not make operands commutative.
- The [existing guard ledger](REGION_GUARD_LEDGER_V1.md) retains all 36 rows per
  call in output/lhs/rhs order. No predicate is discharged, no failure can be
  hoisted, and no rollback or retry is introduced.
- Standalone verification checks self-consistency. Source-paired verification
  additionally requires sealed native evidence. Serialized labels never
  authorize CPU lowering.

The production diff is confined to native admission and the existing region
builder/verifier. ODS, runtime/provider implementation, code generation, public
headers, scheduling and the allocator-registration sanitizer shim are unchanged.

## Acceptance and falsification

| Control | Mechanical evidence |
| --- | --- |
| Preserve noncommutativity | Swapped second operands reject even with identical dynamic types; ordinary execution proves D*C=[105,122;43,50], not C*D. |
| Preserve rectangular geometry | C[2x4], D[5x2], E[5x4] executes against an independent double oracle; wrong dimension operand/axis mutations reject. Captured shapes remain dynamic. |
| Preserve storage/value distinction | Stale precommit tensors, wrong carried type, erased/unused noncarried reads and forged role/binding/snapshot metadata reject. |
| Preserve late alias reads | C[2x3] and distinct D[3x2] share six float objects; D is read after C changes. Hoisting the read or borrowing guard0 rejects. |
| Preserve ordered failure | Invalid second-call shape leaves completed C intact and all E sentinels unchanged on both original execution pipelines. No universal pre-write failure promise is inferred. |
| Preserve host observation | Intervening observer rejects region admission; original execution must print exactly `rhs-observed:62`. |
| Preserve existing admission | Lhs-only, A=A, transparent references and C*C remain covered; independent calls, copied descriptors, barriers and self-aliasing remain rejected. |
| Tolerate legal upstream rewriting | All four source regions survive actual named-to-generic Linalg conversion, canonicalization/CSE/symbol DCE, fresh-context parsing and source pairing. |
| Reject authority forgery | CPU lowerer still rejects regions, including forged producer/capability/target/retry labels; no derived region is executed. |

The CLI's 16 cases run through both existing `capture-v0` and
`matcore-mlir` per-call routes: 32 executable witnesses, not region execution.
The six new cases use in-bounds live arrays and small integer arithmetic exactly
representable in binary32. This justifies exact comparisons for these witnesses,
not general bitwise floating-point equivalence. Unavailable-region mode exits
before these witnesses; its success is not RHS numerical execution evidence.

## Validation

All local integration builds started from clean `33423177`, with coherent
Clang/LLVM/MLIR 21.1.8 and Ninja parallelism 2. No system package or toolchain
changed. Focused tests preceded full affected regressions.

| Scope | Outcome |
| --- | --- |
| Frontend lane, native ON / MLIR OFF | 44 extractions, focused 1/1 passed. |
| MLIR lane, Release / OpenBLAS OFF | 426/426 checks, focused 1/1 passed. |
| Fresh integration Release / OpenBLAS OFF / vector readiness ON | Focused scope passed; full 74/74 passed, no skips. Complete log was independently re-authenticated after session interruption, not represented as a second rerun. |
| Fresh integration Debug / OpenBLAS required 0.3.32 / vector readiness ON | Focused 4/4 (80.20 s); full 74/74 (355.15 s), no skips. |
| Fresh integration Debug ASan+UBSan / OpenBLAS OFF | Exact 24-test scope: 24/24 (5.35 s), no skips. |
| Region MLIR / native / nonexecuting ledger oracle | 426 checks / 44 extractions / 219 checks, zero failures. |
| Region CLI in both complete builds | 105 grouped steps each, including 32 original-route executables. |
| Repository hygiene and whitespace | PASS. |

The full suites include frontend contracts, semantic/structured/buffer/vector
proofs, runtime/planners, ABI compatibility, installed/relocated consumers and
a source/build-inaccessible consumer bound to exact clean `33423177`.
The sanitizer scope matches the existing hosted 24-test in-process selection,
including allocator-protocol positive/negative controls. It is not a claim of
fully instrumented CLI/package execution. Local symbolization is disabled to
avoid symbolizer network stalls; memory, leak, initialization-order and
undefined-behavior checks remain enabled.

Reproduction uses the exact tuple in [AGENTS.md](../../AGENTS.md), with local
MLIR CMake package at
`/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir`.
Builds: `build-rhs-release`, `build-rhs-debug-openblas`, `build-rhs-asan`.
Run `cmake --build <build> --parallel 2`, focused CTest first, then
`ctest --test-dir <build> --output-on-failure -j1` for Release/Debug.
ASan selection and flags are the unchanged
[hosted workflow](../../.github/workflows/mdslc-native.yml).

Local complete `Testing/Temporary/LastTest.log` SHA-256 identities:

- Release: `01ac3b531f76f27ea225fcf723ea17d8878381ee08c2baca1dfba52c6888416d`.
- Debug: `581b53f4ccc8b614a33e26a074c1183caff36d0f8e75c44cea5b510ee1ffe493`.
- ASan: `bb8fb3ee7fee23c75f2a26c6aff4155fbf99eb7f287d1caeb292c2a3ae722d55`.

Independent evidence: [frontend lane](agent-reports/two-gemm-rhs-frontend-v1.md),
[MLIR lane](agent-reports/two-gemm-rhs-mlir-v1.md),
[architecture review](agent-reports/two-gemm-rhs-architecture-review-v1.md),
[oracle and Release evidence review](agent-reports/two-gemm-rhs-oracle-review-v1.md).
Final exact-head hosted checks and merge identity are recorded in the PR and
the post-merge operator checkpoint; no pending hosted result is counted here.

## Deliberate limits and next boundary

No generated region execution, fusion, general DAGs, dual SSA forwarding,
tensor/view frontend, guard discharge, storage ownership, region bufferization,
tiling, target-specific policy, runtime replacement or API/ABI freeze.
No zero-copy, target-support, performance or BLAS-parity claim. Issues #15 and
#20 remain open; this work adds no missing benchmark envelope evidence.

Exactly one next justified boundary is **the connected region storage/commit
handoff design decision**, before implementing its consumer.

The existing [buffer handoff](../../compiler/lib/mlir/MatcoreBufferizedGemmHandoff.h)
accepts exact isolated structured GEMM with original-output identity and no
allocations/copies; it rejects unknown operations. The existing
[vector specimen](../../compiler/lib/mlir/MatcoreStructuredGemmVectorReadiness.h)
is also isolated and fully static. Neither consumes the authenticated ordered
region. The upstream materialization control exposes allocation plus copying
with unresolved ownership, not a selected region storage model.

Decide how borrowed versus snapshot values, cross-call physical aliases, late
reads, destination identity, allocation/deallocation ownership and allocation
or post-write failures compose. Boundary bufferization interfaces versus
tensor islands with explicit materialization are consequential architectural
alternatives. Under the owner's stop rule, obtain explicit architectural
approval before production implementation. Do not treat unknown-op
bufferization, outlining arithmetic or a successful pass as this decision's
proof, and do not infer generated execution authority from inspection success.
