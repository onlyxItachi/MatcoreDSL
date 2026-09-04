# MDSLC reusable contraction and structured-certificate foundation v1

Date: 2026-09-05

Status: internal implementation checkpoint on
`mdslc/contraction-foundation-v1`. This is not a public operation/API design,
an execution route, or a claim that non-GEMM source operations are
authenticated.

## Identity and evidence boundary

The exact canonical base is
`327530d287e41c4115365598e76b17e149a1c45a`, the canonical merge checkpoint
containing the compiler-archaeology reconciliation and structured GEMM proof.
This work inherits the corpus identities and limitations recorded in
`CORPUS_REENTRY_RECONCILIATION_V1.md` and the semantic/provenance and
inspection-only boundaries in `STRUCTURED_GEMM_HANDOFF_V1.md`.

The corpus and implemented GEMM seam establish that Matcore must retain
operation identity, destination behavior, numerical permissions, effects,
alias requirements, and provenance while upstream MLIR owns structured index
maps and iterator machinery. They do not authenticate GEMV, DOT, GER, batched
GEMM, transposed source operations, or an executable generated path.

## What is implemented

### Canonical topology model

`MatcoreContractionModel.{h,cpp}` is an internal structured-boundary model for
the exact logical loops, affine indexing maps, operand ranks, orientations,
and parallel/reduction roles of standard dense linear-algebra operations. It
has a strict versioned attribute encoding and decoder. Both directions fail
closed: encoding first verifies the complete canonical model and requires the
builder and maps to share one MLIR context; decoding regenerates the canonical
model and compares the entire attribute. Neither path accepts an arbitrary
artifact-supplied affine map as canonical.

The model deliberately contains no extents, layouts, strides, memory spaces,
alias facts, numerical flags, destination update rule, scheduling constants,
or execution policy. Those are different semantic or lowering contracts.

| Standard identity | Canonical ranks `(lhs,rhs,out)` | Loops | Classification | Current authority |
| --- | --- | --- | --- | --- |
| GEMM | `(2,2,2)` | parallel M, parallel N, reduction K | reduction contraction | Authenticated `mdsl.gemm` supports NN only; its existing structured handoff now uses this model mechanically |
| GEMV | `(2,1,1)` | parallel M, reduction K | reduction contraction | Model plus MLIR 21 `linalg.matvec`/`linalg.vecmat` topology proof only |
| DOT | `(1,1,0)` | reduction K | reduction contraction | Model plus MLIR 21 `linalg.dot` topology proof only |
| GER | `(1,1,2)` | parallel M, parallel N | outer-product update, no reduction | Model and an accumulating upstream `linalg.generic` carrier proof only |
| batched GEMM | `(3,3,3)` | parallel B/M/N, reduction K | reduction contraction | Aligned-batch model plus MLIR 21 named NN/TN/NT topology proof only; no broadcast form |

GEMM and batched GEMM cover NN/NT/TN/TT logical maps. GEMV covers normal and
transposed matrix orientation. DOT and GER reject invented matrix-orientation
axes. An orientation changes logical indexing; it does not imply a physical
transpose, copy, layout, or scheduling decision.

The inspected MLIR 21.1.8 install provides named carriers for
`linalg.matvec`, `linalg.vecmat`, `linalg.dot`,
`linalg.matmul_transpose_a`, `linalg.matmul_transpose_b`,
`linalg.batch_matmul`, `linalg.batch_matmul_transpose_a`, and
`linalg.batch_matmul_transpose_b`; their maps/ranks/iterators mechanically
match the corresponding model after the explicit normalization described
below. It has no single named TT matmul/batched-matmul carrier, so TT remains
a canonical topology result, not an upstream named-op proof. For transposed
GEMV, upstream
`linalg.vecmat` orders DPS inputs as `(vector, matrix, output)`, whereas the
Matcore model retains the standard GEMV semantic order
`(matrix, vector, output)`. The test adapter performs that explicit operand
permutation before comparing maps; it does not silently redefine GEMV-T as a
different source operation.

All named and generic carrier results in this section prove topology only.
They do not prove source admission, alpha/beta or destination behavior,
numerical legality, provenance, aliasing, execution, or performance.

### Vocabulary reconciliation

The repository still contains legacy private research documents using `GEVM`
for row-vector-times-matrix, principally
`docs/performance/audits/gemv-gevm-kernel-design.md`,
`docs/performance/kernels/gevm.md`, the performance handbook/checklist, and
historical roadmap/status/ADR non-goal lists. One future-state validation entry
also retains `FUTURE_GEMV_GEVM`. Those records are evidence/design history,
not an accepted public name or operation. This branch does not broadly rewrite
historical documents. Any future user-facing operation should use standard
GEMV terminology with an explicit transpose/orientation variant. No `GEVV`
spelling was found in the audited `docs/` or `compiler/` trees.

This answers the canonicalization question in two parts:

1. The standard family can share one strict affine-map/iterator/rank carrier.
2. It must not share one erased semantic identity. Destination behavior and
   source authority remain operation-specific. GER is not a reduction
   contraction or “GEMM with K equal to one”; standard GER is an outer/rank-1
   update and its model fixture adds the product to the prior output.

Likewise, a rank-two GEMM with M, N, or K equal to one remains GEMM. Independent
authenticated fixtures exercise unit M, unit N, and unit K with the other
dimensions non-unit and prove each remains rank two and retains `mdsl.gemm`
identity. Independent zero-M, zero-N, and zero-K fixtures are rejected by the
authoritative Matcore IR v1 positive-extent rule. A dynamic symbol in IR v1 is
also specified as a positive runtime value: the MLIR `?` records an unknown
extent, not permission for a concrete zero. This structured/topology layer
does not observe runtime values or consume that precondition. It therefore
neither enables dynamic zero nor changes operation identity from extents.

### Reusable proof-carrying certificate

`MatcoreStructuredHandoffCertificate.{h,cpp}` extracts the operation-neutral
part of the structured GEMM certificate. It now owns:

- the exact semantic-source and structured-module envelopes;
- inspection-only profile admission;
- source producer/bridge/capture and translation-unit binding;
- ordered site, source symbol, function type, location, and opaque exact
  semantic-contract binding;
- exact source/structured paired comparison across MLIR contexts;
- a deterministic versioned SHA-256 semantic fingerprint.

Generic source-envelope, structured-envelope/site, and raw-fingerprint
verification entry points run upstream MLIR verification before interpreting
certificate fields. Any function handle accepted alongside a module must be a
direct member of that exact module;
same-context, identical-looking handles from another module and semantic/
structured hybrid handles fail closed. The raw fingerprint functions remain
internal diagnostic primitives, are explicitly documented as
non-authoritative, verify their carrier, and require operation-specific paired
verification before any authority-bearing use.

The fingerprint domain is
`matcore-structured-semantic-fingerprint-v1`. Length-delimited canonical
fields cover source module identity, capture ordinal, source operation,
semantic symbol/site, function type, source location, and the complete opaque
semantic contract. Cross-context print/parse tests produce the same digest;
changing an operation contract changes it. This is substitution detection,
not a source-byte signature, imported-artifact authentication, or execution
capability. The digest is computed during paired verification and is not a new
serialized/public contract field.

The existing structured GEMM implementation now uses this reusable
certificate for construction, standalone envelope checking, exact contract
selection, source pairing, and fingerprints. Its emitted v1 module and textual
golden are unchanged.

The following remain intentionally GEMM-specific:

- admission through `verifyMatcoreV1BridgeModule`;
- interpretation of the 13-field `mdsl.gemm` semantic contract;
- reconstruction and verification of an `mdsl.gemm` witness;
- positive-zero fill and overwrite dataflow;
- exact `linalg.matmul` scalar region and no-fast-math checks;
- the current explicit-GEMM source and runtime authority.

This split is the safety property: a future operation can reuse provenance and
pairing without inheriting GEMM's overwrite behavior or execution permission.

## Mechanical falsification coverage

The focused tests reject:

- swapped maps, lost reduction iterators, wrong ranks, forged topology class,
  malformed versions, and noncanonical encoded attributes;
- treating GER as a reduction or as singleton-K GEMM;
- treating independent unit-M, unit-N, or unit-K rank-two GEMM as GEMV or DOT;
- admitting static zero M, N, or K, or treating a dynamic `?` as concrete-zero
  authority;
- DOT/GER vector transpose axes and a transpose on GEMV's vector operand;
- encoding a noncanonical topology or encoding maps through a different MLIR
  context;
- core-invalid semantic-source MLIR, cross-module function handles, and
  semantic/structured hybrid fingerprint handles;
- feeding any model-only `linalg.generic` GEMV, DOT, GER, or batched GEMM to
  the authenticated structured-source API or CPU runtime lowering;
- altered module/site provenance, contract fields, destination behavior,
  numerical permission, alias/alignment preconditions, maps, scalar dataflow,
  or inspection authority in the inherited GEMM adversarial suite.

Positive carrier tests parse and upstream-verify representative generic and
MLIR 21 named operations, then compare their maps/ranks/iterators with the
canonical model. The GER fixture separately requires the exact no-fast-math
body `mul(x,y)` -> `add(old C,mul)` -> `yield(add)` and forbids a GEMM-style
fill. These tests prove that upstream Linalg can carry the stated topology and,
for that one GER fixture, the stated scalar update. They do not prove a
Matcore source operation, full BLAS alpha/beta semantics, bufferization,
vectorization, execution, or performance.

## Architectural result

```text
authoritative operation semantics (currently only explicit mdsl.gemm here)
                    |
                    +--> operation-neutral certificate
                    |      provenance / site / contract / fingerprint
                    |
                    +--> operation-specific semantic verifier
                    |      overwrite / effects / numerical / authority
                    |
                    +--> canonical structured topology
                           standard identity / ranks / maps / iterators
                                      |
                                      v
                              upstream Linalg carrier
```

Matcore should continue to own operation identity and every legality fact that
would otherwise be lost. MLIR should own canonical structured operations,
affine indexing, tiling, bufferization, and vector transformation mechanics.
LLVM/backends should own instruction selection, registers, and machine
scheduling. The authenticated runtime/providers remain the only executable
route.

## Explicit non-results

- No public GEMV, DOT, GER, batched-GEMM, tensor, or view type/API was added.
- No `mdsl.contraction` operation or serialized interchange format was added.
- No existing structured schema, golden, public ABI, or package export changed.
- No transpose, batch, zero-extent, recovered-C++, buffer, vector, target, or
  generated execution authority was created.
- No scheduling constants, provider policy, performance result, BLAS parity,
  zero-copy, GPU, or NPU claim follows.

The smallest justified use of this foundation is for downstream inspection
branches to consume the common certificate and topology checks while retaining
their own bufferization/vector postconditions. New authenticated source
operation identities require a separate semantic and frontend decision; this
checkpoint does not pre-empt Issue #20.
