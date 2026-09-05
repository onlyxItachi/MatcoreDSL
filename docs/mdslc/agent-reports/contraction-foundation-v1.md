# Contraction foundation v1 implementation record

Date: 2026-09-05

## Checkpoints

- Canonical base: `327530d287e41c4115365598e76b17e149a1c45a`
- Implementation commit:
  `6bd3fcc3b3d50af584f73c4897b392d79b83f987`
- Initial durable record:
  `c36b6fc91da16412c3e007c3c420566365df4ecc`
- Independent-review hardening commit:
  `472e695f061fa9577279b5c159aef1b4b7086419`
- Exact-head provenance-binding hardening commit:
  `e0c48b35a468d36d6ade96d5c2e0d2a3717618a6`
- Reviewed pre-merge head:
  `1b7bc9fa212894e330777c18c918c7c533d05c4b`
- Canonical PR #24 merge:
  `5983c2c1bf067bed9e69d0172b0944b7a4c14c00`
- Branch: `mdslc/contraction-foundation-v1`
- Corpus and re-entry identity: inherited unchanged from
  `docs/mdslc/CORPUS_REENTRY_RECONCILIATION_V1.md`
- Proven structured-GEMM boundary: inherited from
  `docs/mdslc/STRUCTURED_GEMM_HANDOFF_V1.md`

This checkpoint is an internal compiler foundation. It does not add a public
operation, source language surface, runtime route, or serialized interchange
contract.

## Surviving implementation

The implementation commit adds two deliberately separate reusable layers:

1. `MatcoreContractionModel` is the canonical internal description of operand
   ranks, logical loop roles, affine indexing maps, orientation, and topology
   class for GEMM, GEMV, DOT, GER, and aligned-batch GEMM.
2. `MatcoreStructuredHandoffCertificate` is the operation-neutral portion of
   the authenticated semantic-to-structured pairing contract: source/site
   identity, ordered opaque semantic contract, inspection-only authority, and
   a versioned deterministic semantic fingerprint.

The existing `mdsl.gemm` handoff consumes both layers without changing the
emitted structured-GEMM v1 schema or golden. GEMM-specific admission,
13-field contract interpretation, overwrite proof, positive-zero fill,
scalar region, numerical restrictions, and execution boundary remain in the
GEMM implementation.

The detailed design, exact topology table, ownership split, vocabulary audit,
and non-results are recorded in
`docs/mdslc/CONTRACTION_FOUNDATION_V1.md`.

## Mechanical invariants and falsification

- Decoding a topology attribute regenerates the selected standard operation's
  canonical model and requires exact equality; encoding also verifies that
  model and its MLIR context before producing an attribute. Artifact-supplied
  arbitrary affine maps cannot enter through either direction.
- GER is an outer/rank-1 update with two parallel loops and no reduction. Its
  representative Linalg body is checked as exact no-fast-math
  `mul(x,y)` -> `add(old C,mul)` -> `yield(add)` dataflow with no fill;
  neither a reduction classification nor GEMM-overwrite semantics validates.
- A rank-two GEMM with a unit M, N, or K extent remains `mdsl.gemm`; geometry
  does not silently change operation identity to GEMV or DOT. Independent unit
  and zero tests cover M, N, and K. Authoritative Matcore IR v1 continues to
  reject zero static extents and specifies dynamic symbols as positive runtime
  values; dynamic MLIR types do not grant concrete-zero authority.
- MLIR 21.1.8 named-op fixtures mechanically match GEMV-N, explicitly
  operand-reordered GEMV-T, DOT, GEMM TN/NT, and aligned-batch NN/TN/NT
  topology. This proves maps/ranks/iterators only. TT remains model-only because
  the inspected version has no one named TT carrier.
- GEMV, DOT, GER, and batched-GEMM fixtures prove only that upstream Linalg can
  carry their maps/ranks/iterator structure. Every such model-only fixture is
  rejected by the authenticated structured-GEMM source API and by the CPU
  runtime lowering API.
- The fingerprint domain is
  `matcore-structured-semantic-fingerprint-v1`; length-delimited canonical
  fields make it stable across MLIR contexts. Every ordered entry-block
  argument location is included, so either one-sided location drift or a
  semantic contract mutation changes the digest. The generic paired verifier
  also compares source/structured argument locations directly. The digest is
  substitution detection, not a source-byte signature or execution authority.
- Generic certificate verification runs upstream MLIR verification before
  interpreting source/structured envelopes. Raw diagnostic fingerprint calls
  reject core-invalid source and structured carriers, non-member function
  handles, same-context mixed modules, semantic/structured hybrids, and
  non-exact source contracts.
- The generic certificate does not invent an argument-location policy. The
  exact current semantic bridge and structured GEMM verifier separately require
  all three entry arguments to retain the authenticated function/source
  location. They reject source-only, structured-only, and matching two-sided
  location forgeries even when the latter has equal non-authoritative raw
  fingerprints.
- Standard visible vocabulary is GEMM, GEMV, DOT, GER, and batched GEMM. The
  legacy private `GEVM` spelling remains evidence-only documentation debt;
  future user-facing orientation should use GEMV plus an explicit transpose or
  orientation. No generic public contraction operation or `GEVV` spelling was
  introduced.

Adversarial tests reject rank, map, iterator, orientation, topology-class,
attribute-version/context, malformed MLIR, module/function identity, contract,
provenance, destination, numerical, alias, authority, and cross-operation
substitutions.

## Validation

A fresh build directory was configured from the clean implementation commit
with:

- Release configuration;
- Clang and Clang++ 21;
- native frontend enabled;
- bootstrap frontend enabled;
- Matcore MLIR enabled;
- exact MLIR 21.1.8 package;
- OpenBLAS disabled.

Results at implementation commit `6bd3fcc3b3d50af584f73c4897b392d79b83f987`:

| Validation | Exact result |
| --- | --- |
| Fresh Release build | `136/136` targets completed |
| `ctest --test-dir build-contraction-foundation-clean --output-on-failure -j 2` | `65/65` passed, `0` failed, 125.66 s |
| Focused CTest regex for contraction model and structured GEMM handoff | `2/2` passed |
| `matcore_mlir_contraction_model_tests` | `164` checks, `0` failures |
| `matcore_mlir_structured_gemm_handoff_tests` | `237` checks, `0` failures |
| `git diff --check` | passed |

Focused hardening results at
`472e695f061fa9577279b5c159aef1b4b7086419`:

| Validation | Exact result |
| --- | --- |
| Incremental Release build | completed; only pre-existing generated-ODS unused-parameter warnings |
| Focused CTest regex | `2/2` passed, `0` failed |
| `matcore_mlir_contraction_model_tests` | `277` checks, `0` failures |
| `matcore_mlir_structured_gemm_handoff_tests` | `274` checks, `0` failures |
| Structured GEMM golden SHA-256 | unchanged: `632d2fec0eb972a54f0cc0714e2e1811f92d39e8fce8417efe6110927965143d` |

The first full CTest attempt after that commit reported `60/65` passing. All
five failures were stale configure/build provenance rather than test behavior:
the package fixture still expected `c36b6fc`, and `matcore-bench` recorded that
it had been built while the pre-commit tree was dirty. The final clean-head
reconfigure/rebuild result is recorded below.

At prior documentation checkpoint
`20980deb30acd400ac1c3d1d9e16f3688283386f`, CMake was re-run to refresh the
exact checkout identity, the Release tree was rebuilt, and the complete CTest
surface passed `65/65`, `0` failed. The direct focused binaries reported
`277/0` contraction checks and `274/0` structured-GEMM checks; `git diff
--check` passed.

Exact-head re-review hardening at
`e0c48b35a468d36d6ade96d5c2e0d2a3717618a6` added ordered entry-argument
locations to the generic fingerprint, direct argument-location pairing, the
current GEMM-specific source/structured location invariants, and direct
core-invalid structured-fingerprint rejection. Validation after a clean-head
CMake refresh produced:

| Validation | Exact result |
| --- | --- |
| Complete Release build | completed; 11 affected targets rebuilt |
| Full CTest | `65/65` passed, `0` failed, 93.94 s |
| Focused CTest regex | `2/2` passed, `0` failed |
| `matcore_mlir_contraction_model_tests` | `277` checks, `0` failures |
| `matcore_mlir_structured_gemm_handoff_tests` | `359` checks, `0` failures |
| Structured GEMM golden SHA-256 | unchanged: `632d2fec0eb972a54f0cc0714e2e1811f92d39e8fce8417efe6110927965143d` |
| `git diff --check` | passed |

The full CTest surface includes native/bootstrap frontend, semantic IR/MLIR,
structured handoff, CPU dispatch/runtime/planner, C ABI, installed consumer,
source-inaccessible package, benchmark-contract, and CLI tests. It does not
constitute Debug, sanitizer, Windows, OpenBLAS-enabled, performance, BLAS
parity, or physical target validation.

## Rejected or deferred alternatives

- No generic public `mdsl.contraction` was created: shared indexing topology
  does not justify erasing standard operation identity or destination
  semantics.
- GER was not canonicalized as singleton-K GEMM: GER has update semantics and
  no reduction iterator.
- Unit dimensions were not used as an operation classifier, and zero extents
  were not enabled by the extent-neutral topology layer.
- No model-only standard-family fixture received authenticated source or
  execution authority.
- The semantic fingerprint was not persisted as a new schema field, ABI, or
  interchange commitment.
- No physical-transpose operation, batching broadcast, alpha/beta contract,
  bufferization, vector lowering, scheduling policy, or provider crossover was
  inferred from topology alone.
- Legacy research documents were not broadly renamed; the exact vocabulary
  debt is inventoried in the foundation document.

## Next justified boundary

The foundation can now be used by independent bufferization and
transform/vector-readiness experiments. Each experiment must consume facts
explicitly, retain operation-specific postconditions, and remain
inspection-only. New authenticated GEMV, DOT, GER, or batched-GEMM source
operations require a separate frontend and semantic decision; this checkpoint
does not pre-empt Issue #20 or authorize generated execution.
