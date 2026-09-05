# Independent adversarial review: ordered two-GEMM region

Date: 2026-09-05. Canonical base:
`5f455bacde0959983b2b888f15fd5dabd4b1ceaa`.
Reviewed production checkpoint:
`de204b5bc288fe90865284b30661aeffadd132d4`.
Final reviewed test checkpoint:
`f8f15eb3fad50af81e07e2deecb25b024975c572`.
The latter adds only the seven-line fresh-context round-trip test.

The reviewer worked independently of both implementation lanes, without
production edits or duplicate builds. No unresolved production correctness
blocker was found **within the explicitly inspection-only admission envelope**.
This is not approval for generated execution, region bufferization, generic
dispatch, or merging without the integration workflow's remaining gates.

## Coverage

The review covered:

- Native admission, declaration/reference identities, reserved-intrinsic
  authentication, physical dependency sealing and immutable frontend evidence.
- Registered descriptor/order types, every region boundary operation's effects
  and verifier, the source-to-region builder, and complete standalone/paired
  verification in `MatcoreTwoGemmRegion.cpp`.
- CLI mode/output isolation, build and CI wiring, direct frontend and MLIR
  tests, upstream storage controls, and unchanged-route executable test source.
- The complete F1-F14 ledger and claim boundaries in `TWO_GEMM_REGION_V1.md`.

Runtime/provider/codegen implementation is unchanged. This was not a new
repository-wide runtime audit or an executable-candidate implementation review.

## Concrete corrections and prudential tightening

Concrete counterexamples caught during design/implementation and covered by
the final rejection/protection tests include:

1. A persistent `static auto &alias = C` can refer to a previous invocation's
   parameter rather than the current one. Only automatic transparent reference
   initializers are chased.
2. Repeated macro-expanded declarations can share a spelling location while
   denoting different bindings. Ambiguous macro identities are rejected.
3. Memory-effect freedom alone admits an unused integer division by zero.
   The final tests require these mutants to pass upstream IR verification and
   then reject them through the Matcore verifier, both outside and inside the
   scalar contraction body.
4. An inspection output can alias a parsed header/dependency, not just the
   main source. Publication now protects the complete captured dependency set.

These are concrete code-level counterexamples; the review does not claim that
every pre-fix implementation was separately built and executed. The source
observer/mutation barriers, pure-tensor DCE loss, and first-success/second-failure
`C=6, E=-9` were motivating architectural falsifiers, not a demonstrated
wrong-result bug in the unchanged per-call runtime.

Prudential hardening is distinct: individual attribute-origin checks protect
against declaration attribute merging; the modeled incidental-operation
allowlist is a conservative certification scope, not proof that upstream
speculatability is unsound. Fresh-context serialization/parse/source pairing
exercises the new type parser without introducing another format or an
authorization route.

## Direct invariants and unopened authority boundaries

F1-F10 directly challenge the admitted region. The first committed postvalue
feeds the second contraction; each read is paired with its stage guard, and D
remains late under MAYalias storage. Distinct declarations are not disjoint
allocation proofs, and A/B aliasing remains legal. Ordered opaque failure
frontiers cannot cross commits. Effects preserve unused output writes through
actual upstream canonicalization, CSE and SymbolDCE. Generalized Linalg is
accepted by maps, iterators and scalar meaning; upstream-valid semantic changes
are rejected. Native source pairing is mandatory: coordinated editable contract
changes can be self-consistent while failing the sealed-source check.

F11-F14 deliberately test unopened authority boundaries. The actual upstream
materialization control exposes allocation/copy and unresolved ownership; it
does not establish source-paired region storage or zero-copy. The actual CPU
lowerer rejects that control and region artifacts even after producer, target,
capability and retry labels are forged.

Those negative tests do **not** implement or prove generic candidate admission,
safe retry after partial writes, loaded-binary authentication, target-artifact
reuse, or hardware/numerical candidate compatibility. The failure obligation is
not a captured C++ handler/noexcept/diagnostic contract. Unchanged host C++ still
owns those behaviors. A retained requirement is not a discharged runtime fact.

## Validation provenance

Independently executed at `de204b5`:

- `frontend.native.two_gemm_region`: PASS, 1/1 CTest, 0.14 seconds.
- Direct frontend test binary: PASS, 28 real native extractions.
- Direct MLIR region test binary: PASS, 83/83 checks.

Independently executed at `f8f15eb`:

- Direct MLIR region test binary: PASS, 85/85 checks, including fresh-context
  text parsing and paired source verification.
- The worktree was clean before this report; `git diff --check` passed at the
  reviewed production checkpoint.

The integration owner's full Release run at `de204b5` was independently
inspected, **not independently rerun**: 73/73 tests passed in 201.33 seconds,
with 73 individual PASS records and no skipped, failed or not-run entries.
Local log: `/tmp/mdslc-region-release-ctest.log`, SHA-256
`dd6d0b7818d9c77ffe95d18ed5ddb4433815d8e02e4574a9e6816fce0fdc485c`.
The owner's focused round-trip log records 85/85 checks, one CTest, 0.04 seconds;
local log `/tmp/mdslc-region-roundtrip-tests.log`, SHA-256
`16d0ab669b468d35b519df2c96132a9186d6305a31a533c3dc2651fa2bcbe414`.
Temporary log paths are local evidence pointers, not portable retained assets.

Wider Debug, OpenBLAS, sanitizer, compatibility and hosted outcomes belong to
the integration record. They must not be inferred from these focused runs or
from the Release configuration, which has OpenBLAS disabled.

## Exactly one next engineering boundary

Build an inspection-only per-call guard/discharge ledger for this same admitted
region. Map retained requirements to exact source proofs or still-required
predicates, using existing runtime validation as the oracle and preserving both
failure frontiers. Raw-pointer shape/range validation must not be mislabeled
proof of backing allocation capacity or lifetime. Do not bundle fusion,
generated execution or generic candidate selection into that next boundary.
