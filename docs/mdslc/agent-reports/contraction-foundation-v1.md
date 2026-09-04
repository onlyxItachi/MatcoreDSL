# Contraction foundation v1 implementation record

Date: 2026-09-05

## Checkpoints

- Canonical base: `327530d287e41c4115365598e76b17e149a1c45a`
- Implementation commit:
  `6bd3fcc3b3d50af584f73c4897b392d79b83f987`
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
  canonical model and requires exact equality; artifact-supplied arbitrary
  affine maps are not accepted as canonical.
- GER is an outer/rank-1 update with two parallel loops and no reduction. Its
  representative Linalg body adds the product to the prior destination value;
  neither a reduction classification nor GEMM-overwrite semantics validates.
- A rank-two GEMM with a unit M, N, or K extent remains `mdsl.gemm`; geometry
  does not silently change operation identity to GEMV or DOT. Authoritative
  Matcore IR v1 continues to reject zero static extents.
- GEMV, DOT, GER, and batched-GEMM fixtures prove only that upstream Linalg can
  carry their maps/ranks/iterator structure. Every such model-only fixture is
  rejected by the authenticated structured-GEMM source API and by the CPU
  runtime lowering API.
- The fingerprint domain is
  `matcore-structured-semantic-fingerprint-v1`; length-delimited canonical
  fields make it stable across MLIR contexts. A semantic contract mutation
  changes the digest. The digest is substitution detection, not a source-byte
  signature or execution authority.
- Standard visible vocabulary is GEMM, GEMV, DOT, GER, and batched GEMM. The
  legacy private `GEVM` spelling remains evidence-only documentation debt;
  future user-facing orientation should use GEMV plus an explicit transpose or
  orientation. No generic public contraction operation or `GEVV` spelling was
  introduced.

Adversarial tests reject rank, map, iterator, orientation, topology-class,
attribute-version, contract, provenance, destination, numerical, alias,
authority, and cross-operation substitutions.

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
