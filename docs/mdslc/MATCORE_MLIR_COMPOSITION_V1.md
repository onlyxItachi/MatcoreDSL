# Matcore MLIR composition contract v1

Status: internal semantic-optimizer contract for Milestone C

Schema identifier: `matcore-mlir-composition-v1`

This contract defines the first versioned multi-operation envelope for the
internal `mdsl` MLIR dialect. It is deliberately separate from the exact
Matcore IR v1 capture bridge documented in
`docs/mdslc/MATCORE_MLIR_DIALECT_V1.md`.

The strict capture bridge remains GEMM-only and lossless. A module may claim
this composition contract only after adding the exact composition marker and
passing `verifyCompositionV1Module`. Setting the marker does not bypass the
operation verifiers or weaken the capture bridge.

## 1. Versioned module envelope

A composition-v1 module requires:

```text
mdsl.composition_schema  = "matcore-mlir-composition-v1"
mdsl.composition_version = 1 : i32
mdsl.semantic_version    = 1 : i32
mdsl.execution_intent    = "generic"
mdsl.source_file         = <nonempty .mdsl path>
```

At module scope, only `func.func` is allowed. Each semantic-root function is
public, defined, and single-block, with three unencoded ranked-tensor
arguments and one unencoded ranked-tensor result. Its `FileLineColLoc` names
the module source. Semantic roots remain in capture order: each carries a
unique canonical `mdsl.site_id`, has the exact symbol
`__matcore_semantic_<site-id>`, and has a signless-i64
`mdsl.capture_ordinal` equal to its zero-based module position. Duplicate site
IDs, missing/duplicate/noncontiguous ordinals, reordered roots, and symbols
that do not match their site reject.

The version-1 function body contains exactly one `mdsl.gemm`, at least one
`mdsl.map`, and exactly one `func.return`. Direct function calls, declarations,
helper functions with another signature, unknown operations, and operations
with unmodelled effects are rejected. The return exposes the final GEMM/map
semantic value; every functional map result must remain live.

The GEMM site ID and location equal the function site ID and location. Every
map extends a GEMM/map SSA producer chain. Derived map and scalar locations
equal their producing semantic location. All non-synthetic map/sine
provenance names the module source and semantic-root site.

The envelope is an allowlist, not a general MLIR program container. Future
control flow, calls, reductions, contractions, or different root signatures
require a new versioned contract.

## 2. Tensor and value contract

Composition v1 uses ordinary MLIR SSA use-def chains. Operation identity is
not encoded through fixed operand names: a map consumes its producer's SSA
result and returns another SSA value.

The implemented surface is deliberately narrow:

- ranked, rank-two F32 tensors;
- host memory;
- row-major contiguous layout;
- static or operation-local symbolic dynamic dimensions;
- no non-null `RankedTensorType` encoding;
- exact propagation of shape, strides, layout, memory space, and alignment;
- functional map results with no in-place alias.

An unknown tensor encoding is not treated as equivalent to the explicit host
row-major contract. GEMM values, map input/result values, and mask values all
reject non-null encodings.

## 3. Functional map

`mdsl.map` applies one pure scalar region to active input elements and returns
a distinct functional tensor value. It never mutates its input. For every
partial domain, an inactive output element preserves the corresponding input
element byte-for-byte/value-for-value.

The body has exactly one block and one scalar F32 argument. Version 1 admits a
nonempty linear chain of pure `mdsl.sin` operations followed by exactly one
`mdsl.yield`. Parent-region captures, dead scalar branches, calls, unknown
operations, side effects, missing/wrong/multiple terminators, and extra blocks
or block arguments reject.

The map's exact semantic dictionaries state:

- input and optional mask reads;
- no writes or read-write effects;
- `inplace = false`;
- `result_identity = "new_functional_value"`;
- `domain_application = "active_elements_only"`.

The pure map does not erase the producing GEMM's explicit destination write.
`mdsl.gemm` remains effectful and non-trivially dead even when its SSA result
feeds a functional map.

## 4. Closed domain model

Every domain dictionary contains `version = 1 : i32` and one exact closed
kind. Coordinates are zero-based.

### 4.1 `all`

```text
{kind = "all", version = 1 : i32}
```

Every tensor coordinate is active. There is no mask operand and
`outside_domain = "not_applicable"`. This is the unique canonical whole-domain
encoding.

### 4.2 `slice`

```text
{begin = [...], end = [...], kind = "slice", step = [...], version = 1 : i32}
```

Slice intervals are zero-based and half-open. Coordinate `i[d]` is active for
dimension `d` exactly when:

```text
begin[d] <= i[d] < end[d]
and
(i[d] - begin[d]) % step[d] == 0
```

All dimensions must satisfy the predicate. Begin is in bounds, end may equal
the static extent, every interval is nonempty, and every step is a positive
signless i64. Slice v1 requires a static tensor shape. A full static slice with
zero begins, extent ends, and unit steps is rejected in favor of `all`.
Inactive elements use `outside_domain = "preserve_input"`.

### 4.3 `indices`

```text
{coordinates = [[...], ...], kind = "indices", version = 1 : i32}
```

Each coordinate is a zero-based, full-rank list of in-bounds signless i64
values. The set is nonempty and coordinates are unique. Its serialized
representation must use strict ascending lexicographic coordinate order after
rank, type, bounds, and uniqueness validation. A reversed or otherwise
permuted spelling is rejected rather than silently normalized. This ordering
does not change the mathematical set; it provides one deterministic byte
representation. Indices v1 requires a static tensor shape. A statically
complete coordinate set is rejected in favor of `all`. Inactive elements use
`outside_domain = "preserve_input"`.

### 4.4 `mask`

```text
{kind = "mask", shape_equality = "required_precondition", version = 1 : i32}
```

The sole optional map operand is a rank-compatible, unencoded i1 tensor. A
`true` mask element is active. A `false` element preserves the corresponding
input element. Statically unequal dimensions reject.

Two dynamic `?` dimensions are not proof of equality. Their equality is an
explicit runtime precondition which a later lowering must prove statically or
guard before execution/output publication. Composition v1 contains no guard
insertion pass and therefore makes no execution claim for an unguarded dynamic
mask.

## 5. Sine numerical contract

`mdsl.sin` consumes and returns scalar F32. Its exact version-1 contract is:

```text
profile               = "sin-f32-ieee-v1"
accuracy              = "correctly_rounded_f32"
approximate_math      = false
rounding              = "nearest_ties_even"
signed_zero           = "preserve"
nan                    = "quiet_nan_payload_unspecified"
infinity               = "quiet_nan"
subnormals             = "ieee_gradual_ftz_daz_forbidden"
exception_status       = "postcall_unspecified"
trapping_exceptions    = "unsupported"
```

These fields are legality obligations, not evidence that a CPU lowering has
met them. No sine lowering or execution claim is part of Milestone C. A future
lowering must implement this contract exactly or reject before consuming the
semantic operation.

## 6. Provenance authenticity

Map and sine provenance is exact, versioned, closed, and paired with an exact
`FileLineColLoc`. `authenticity` distinguishes three cases:

- `derived_from_producer`: the operation was introduced by a reviewed semantic
  transformation. It uses a canonical producer site ID, carries no source
  expression claim, and in the composition envelope has the same location as
  its producer.
- `source_authenticated`: the operation corresponds to actual source text. It
  requires an exact nonempty half-open byte range and a `sha256:<64 lowercase
  hex>` source snapshot identity in addition to file/line/column and anchor.
- `synthetic_test_fixture`: the operation exists only in a focused verifier
  fixture. It may pass operation verification but is rejected from the
  production composition envelope.

Map kinds are respectively `derived_semantic_composition`,
`source_authenticated_map`, and `synthetic_semantic_composition`. Sine kinds
are `derived_elementwise_expression`, `source_authenticated_expression`, and
`synthetic_elementwise_expression`. Crossed authenticity/kind combinations or
extra fields reject.

The operation verifier validates the closed syntax of source-authenticated
provenance, but syntax is not authentication. The context-free production
`verifyCompositionV1Module` entry point therefore fails closed whenever a map
or sine claims `source_authenticated`.

A caller may instead provide `AuthenticatedSourceSnapshotV1` to the trusted
composition-verification overload. That context binds:

- the exact `mdsl.source_file` identity;
- caller-trusted source bytes;
- their separately supplied canonical SHA-256 identity; and
- their exact byte length.

The verifier checks the declared length, recomputes the digest, requires every
source-authenticated operation to carry that file and digest, bounds every
half-open range by the trusted byte length, and derives its one-based
file/line/column from the range begin. Columns are byte-oriented; LF, CRLF, and
lone-CR line endings are handled as source line breaks. Every nested
source-authenticated scalar operation is checked independently. The generic
verifier performs no filesystem I/O: loading and trusting source bytes is the
caller's responsibility.

The canonical GEMM-to-sine goldens use derived provenance. They do not claim
that the captured GEMM source contains a sine expression.

## 7. Determinism and validation

Canonical `GEMM -> SIN(all)` and static partial-slice modules are complete
parse/print goldens. Serialization must be byte-stable after reparsing and both
operation verification and the composition-v1 envelope must pass before and
after that round trip.

Focused adversarial tests cover all domain kinds, dynamic-mask guard
obligations, exact inactive-element preservation, malformed region structure,
captures/effects, tensor encodings, full-domain canonicality, provenance
authenticity/ranges/snapshots, strict lexicographic index ordering, trusted
source identity/digest/length/range/line binding, context-free fail-closed
behavior, source and site binding, calls, unknown
operations, unsupported functions, dead semantic results, and malformed
composition versions.

## 8. Deliberate limitations

- No public C++ `map` or `sin` API exists.
- No map/sine lowering, fusion, vectorization, or runtime execution exists.
- Slice and indices domains are static-only.
- Dynamic masks require a later dominating runtime equality guard.
- The scalar allowlist contains only sine.
- No reductions, contractions, views, control flow, calls, or arbitrary MLIR
  operations are accepted.
- The Matcore IR v1 JSON capture and strict GEMM bridge remain unchanged.
- This is an internal optimizer contract, not a public native ABI.
