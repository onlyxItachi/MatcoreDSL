# Matcore MLIR semantic dialect contract v1

- Contract version: 1
- Textual dialect namespace: `mdsl`
- Bridge schema: `matcore-mlir-semantic-v1`
- Capture schema: `matcore-ir-v1`
- Implementation status: accepted internal Milestone B contract
- Public API/ABI status: unfrozen; this document does not make MLIR an
  installed C++ API

This document is the precise contract implemented by the first Matcore MLIR
semantic layer. ADR-0009 owns the architectural rationale. This document owns
the version-1 encoding, verifier boundary, and current limitations.

Matcore IR v1 remains the deterministic JSON capture/provenance DTO. The
`mdsl` dialect is the compositional optimizer representation. The bridge is a
loss-checked conversion from the former to the latter, not a replacement
capture schema and not an alternate source language.

## 1. Toolchain and package gate

The component is opt-in through:

```text
MDSLC_ENABLE_MATCORE_MLIR=ON
```

The default remains `OFF`. Product/default use requires an explicit `MLIR_DIR`
and exact LLVM, Clang, and MLIR version `21.1.8`. The advanced internal
compatibility selector admits only the separately reviewed coherent exact
22.1.8 Linux x64 tuple; it is not product support or an installed contract.
Configuration fails closed if the selected exact tuple differs or any of these
imported targets is absent:

```text
MLIRIR
MLIRFuncDialect
MLIRParser
MLIRTransforms
MLIRDestinationStyleOpInterface
MLIRSideEffectInterfaces
```

The standalone build remains Ninja/C++20/Clang-first. The coherent audited
product configuration uses LLVM and Clang 21.1.8 from the system package and
an exact MLIR 21.1.8 development prefix. Mixed distro-22 components and the
legacy root project's MLIR 18 requirement are not valid substitutes. Exact
22.1.8 compatibility details and limitations are recorded in
`agent-reports/mlir-upstream-portability-v1.md`.

`matcore_mlir_semantics` is an internal static build target. It is not
installed or exported. Only the leaf inspection executable `matcore-mlir` is
installed. Existing `find_package(MatcoreDSL)` consumers therefore acquire no
MLIR headers, libraries, CMake targets, or development-prefix paths.

## 2. Responsibility and version boundary

Version 1 represents semantic WHAT:

- authenticated GEMM identity and origin;
- tensor shape, layout, stride, dtype, memory-space, and mutability contracts;
- destination identity, effects, alias requirements, and synchronization;
- numerical permissions and source provenance;
- source target/fallback policy and execution intent.

It does not represent a selected implementation, schedule, library, ISA,
microkernel, or machine target. Those are HOW or MACHINE decisions. In
particular, no version-1 operation may encode `openblas`, `avx2`, `avx512`, a
tile size, or a future accelerator instruction family.

The current version identifiers are:

| Meaning | Encoding | Required value |
| --- | --- | --- |
| semantic module version | `mdsl.semantic_version` (`i32`) | `1` |
| bridge schema | `mdsl.bridge_schema` | `matcore-mlir-semantic-v1` |
| capture schema | `mdsl.capture_schema` | `matcore-ir-v1` |
| capture version | `mdsl.capture_version` (`i32`) | Matcore IR version `1` |
| origin version | `origin.version` (`i32`) | `1` |
| provenance version | `provenance.version` (`i32`) | `1` |

Unknown or unrepresentable capture information is a conversion error. A
future incompatible encoding requires a new version and conversion boundary;
it must not loosen this verifier in place.

## 3. Module and semantic-site structure

The deterministic Matcore IR v1 bridge emits one `module`. Its required module
attributes are:

| Attribute | Type and closed value |
| --- | --- |
| `mdsl.bridge_schema` | string `matcore-mlir-semantic-v1` |
| `mdsl.capture_schema` | string `matcore-ir-v1` |
| `mdsl.capture_version` | signless `i32` equal to `1` |
| `mdsl.semantic_version` | signless `i32` equal to `1` |
| `mdsl.numerical_profile` | string `explicit-gemm-f32-v1` |
| `mdsl.execution_intent` | string `generic` |
| `mdsl.producer` | `clang-libtooling-v1` or compatibility producer `clang-ast-json-bootstrap-v0` |
| `mdsl.source_file` | nonempty string ending in `.mdsl` |
| `mdsl.translation_unit` | nonempty string |

The explicit bridge envelope permits only one `func.func` per captured site,
in capture order. Each function has:

- public MLIR symbol visibility;
- name `__matcore_semantic_<site-id>`;
- `mdsl.capture_ordinal` as a signless `i64` equal to its zero-based module
  order;
- `mdsl.site_id` equal to the contained GEMM's site ID;
- three arguments ordered `lhs`, `rhs`, `output`;
- one result with exactly the output tensor type;
- one block containing exactly one `mdsl.gemm` and one `func.return`;
- a return operand equal to the GEMM result.

The function and operation use the captured `FileLineColLoc`. The bridge
emits a separate function for each site so dynamic dimension symbols remain
operation-local even when two sites reuse the same spelling.

Public symbol visibility is an internal semantic-liveness root. Standard MLIR
SymbolDCE otherwise removes unreferenced private capture functions. These
names and their visibility are **not** a native symbol ABI, an export promise,
or a public C++ API. Machine emission must first choose translation-unit-safe
names and deliberate visibility.

The explicit envelope verifier currently requires the attributes and
structure above but does not reject every additional module or function
attribute accepted by generic MLIR. Unknown attributes therefore convey no
Matcore v1 permission. Named nested dictionaries below are exact and closed.

## 4. `mdsl.gemm`

The sole Matcore IR v1 **capture-bridge** operation is:

```text
%result = mdsl.gemm %lhs, %rhs outs(%output : tensor<...>) {...}
          : tensor<...>, tensor<...> -> tensor<...>
```

### 4.1 Operands, result, and destination identity

All operands and the result are ranked tensors. The supported surface is
rank-two, row-major contiguous, host-resident F32 GEMM:

```text
lhs    : tensor<M x K x f32>
rhs    : tensor<K x N x f32>
output : tensor<M x N x f32>
result : tensor<M x N x f32>
```

Static or dynamic dimensions are allowed. The M/K/N equalities are exact
operation-local equality of the encoded scalar-expression attributes. Equal
symbol spellings in different site functions do not establish cross-site
identity.

`output` and `result` must have identical types. The operation implements
`DestinationStyleOpInterface`: `lhs` and `rhs` are DPS inputs, `output` is the
single DPS init, and the sole result is tied to that destination. Semantically,
the result is the post-overwrite value of the explicit destination, not a new
allocation. The destination is write-only for GEMM and is not an accumulator
input.

This destination identity is a semantic obligation. Version 1 does not yet
implement bufferization or prove physical storage aliasing; a later lowering
must preserve the result/output relationship structurally.

### 4.2 Top-level attributes

`mdsl.gemm` requires these ODS attributes:

```text
site_id                 : string
origin                  : dictionary
accumulation_type       : type
lhs_semantics           : dictionary
rhs_semantics           : dictionary
output_semantics        : dictionary
semantic_requirements   : array
aliasing                : array of dictionaries
effects                 : dictionary
synchronization         : string
policy                  : dictionary
numerical               : dictionary
provenance              : dictionary
```

`site_id` is exactly `mc_` followed by 32 lowercase hexadecimal digits.
`accumulation_type` is exactly `f32`. The current op verifier validates every
required field but generic MLIR may retain unrelated extra top-level operation
attributes; those extras grant no semantic permission.

## 5. Tensor-semantics dictionaries

Each of `lhs_semantics`, `rhs_semantics`, and `output_semantics` contains
exactly these nine fields:

```text
alignment_bytes
alignment_contract
layout
memory_space
mutability
role
shape
source_expression
strides
```

The closed values are:

| Field | Contract |
| --- | --- |
| `role` | `lhs`, `rhs`, or `output`, matching the dictionary position |
| `mutability` | `read` for lhs/rhs; `write` for output |
| `memory_space` | `host` |
| `layout` | `row_major_contiguous` |
| `alignment_contract` | `required_precondition` |
| `alignment_bytes` | signless `i64`, power of two, at least 4 |
| `source_expression` | nonempty string retained from capture |
| `shape` | exactly two scalar-expression dictionaries |
| `strides` | exactly two scalar-expression dictionaries |

A scalar-expression dictionary is one of exactly:

```text
{kind = "static", value = <positive signless i64>}
{kind = "dynamic", symbol = <canonical symbol>}
```

A static shape value must equal its corresponding positive ranked-tensor
dimension. A dynamic shape expression requires a dynamic tensor dimension.
Dynamic symbols use the canonical identifier form represented by the verifier
(`[_A-Za-z][_A-Za-z0-9]*` in the normal C locale). Static strides are positive;
dynamic strides use the same canonical-symbol rule.

For row-major contiguous tensors, `strides[0]` must be attribute-identical to
`shape[1]`, and `strides[1]` must be `{kind = "static", value = 1 : i64}`.

## 6. Requirements, aliases, effects, and synchronization

`semantic_requirements` is the exact ordered string array:

```text
["rank2_gemm", "f32_arithmetic", "host_addressable",
 "synchronous_execution"]
```

`aliasing` contains exactly two ordered relation dictionaries. Each relation
contains exactly `contract`, `first`, `relation`, and `second`:

```text
{contract = "required_precondition", first = "output",
 relation = "no_alias", second = "lhs"}
{contract = "required_precondition", first = "output",
 relation = "no_alias", second = "rhs"}
```

`effects` contains exactly:

```text
{
  reads = ["lhs", "rhs"],
  writes = ["output"],
  read_write = []
}
```

The ordered textual effect dictionary is not the only effect signal. The op
also implements `MemoryEffectOpInterface` and reports three operand effects:
read on lhs, read on rhs, and write on output. It is not memory-effect-free or
trivially dead. The observable write prevents an unused SSA result from making
the GEMM pure, while public site functions prevent symbol-level deletion.

`synchronization` is exactly `synchronous`.

## 7. Required preconditions are not facts

`alignment_contract = "required_precondition"` and the two no-alias
relations express obligations for a legal execution. They do not prove that a
concrete pointer is aligned or that runtime storage does not overlap.

A later optimization may consume either requirement only after:

1. static analysis proves it for the actual SSA/runtime values; or
2. a dominating runtime guard checks it on every path before packing,
   speculative access, destination mutation, or other observable effects.

A candidate may be conditional on a precondition. It may not execute and check
afterward. Guard hoisting or combination must preserve effect and failure
ordering. The same discipline applies whenever a captured shape, stride, or
source-derived condition is not statically established for the concrete
execution.

## 8. Origin, numerical profile, policy, and provenance cross-products

Core `mdsl.gemm` verification recognizes exactly three correlated states.
Mixing a field from one row with another row is invalid.

| State | Origin permission | Numerical profile | Policy | Execution status |
| --- | --- | --- | --- | --- |
| authenticated explicit eDSL | no permission field | `explicit-gemm-f32-v1` | `cpu` / `error` | only state accepted by the Matcore IR v1 executable bridge |
| relaxed recovered C++ | `source_proven_guard_required` | `recovered-cpp-gemm-f32-source-proven-v1` | `generic` / `preserve_original_cpp` | core-representable; requires trusted producer proof plus a dominating guard; not accepted by the v1 bridge envelope |
| strict recovered C++ | `recognized_rewrite_rejected` | `recovered-cpp-gemm-f32-strict-v1` | `generic` / `preserve_original_cpp` | analysis-only; ordinary C++ must remain unchanged; never execution permission |

### 8.1 Origin dictionaries

The explicit dictionary contains exactly:

```text
{kind = "explicit_call", version = 1 : i32,
 canonical_callee = "matcore::mdsl::gemm"}
```

The recovered dictionary contains exactly:

```text
{kind = "recovered_cpp_loop", version = 1 : i32,
 pattern = "canonical-row-major-f32-gemm-v1",
 permission = "source_proven_guard_required" |
              "recognized_rewrite_rejected"}
```

Recovered operations never carry or forge `canonical_callee`.

### 8.2 Numerical dictionary

The numerical dictionary contains exactly these 15 fields:

```text
accumulation_dtype, approximate_math, contraction, derivation,
exception_status, infinity, inplace, nan, profile, reassociation,
reduction_order, rounding, signed_zero, subnormals,
trapping_exceptions
```

Fields common to all three current profiles are:

| Field | Exact value |
| --- | --- |
| `accumulation_dtype` | `f32` |
| `approximate_math` | `false` |
| `inplace` | `false` |
| `infinity` | `ieee_no_no_infs_assumption` |
| `rounding` | `nearest_ties_even` |
| `exception_status` | `incoming_not_preserved_postcall_unspecified` |
| `subnormals` | `ieee_gradual_ftz_daz_forbidden` |
| `trapping_exceptions` | `unsupported` |

The correlated differences are:

| Field | Explicit eDSL | Relaxed recovered | Strict recovered |
| --- | --- | --- | --- |
| `profile` | `explicit-gemm-f32-v1` | `recovered-cpp-gemm-f32-source-proven-v1` | `recovered-cpp-gemm-f32-strict-v1` |
| `derivation` | `explicit_edsl_contract` | `effective_cpp_semantics` | `effective_cpp_semantics` |
| `reassociation` | `within_k_reduction` | `within_k_reduction` | `forbidden` |
| `contraction` | `allowed` | `allowed` | `within_statement` |
| `reduction_order` | `implementation_defined_within_k` | `implementation_defined_within_k` | `increasing_k` |
| `nan` | `preserve_classification_payload_order_unspecified` | same as explicit | `strict` |
| `signed_zero` | `relaxed` | `relaxed` | `preserve` |

The relaxed recovered tuple is semantically equal to the explicit numerical
tuple only after ignoring the deliberately different `profile` and
`derivation`; its permission still comes from effective C++ options and a
runtime/static guard, not from similarity to the eDSL. The strict tuple records
why a normally evaluated increasing-K C++ loop cannot be replaced by the
relaxed implementation-defined reduction.

`nan = "strict"` is the current fail-closed source-semantics label. Version 1
does not use it to authorize a lowering. A future executable strict lowering
must define and validate its complete source-language conformance before this
analysis-only state can change permission.

### 8.3 Policy dictionary

`policy` contains exactly `target` and `fallback`:

- explicit: `{target = "cpu", fallback = "error"}`;
- either recovered state:
  `{target = "generic", fallback = "preserve_original_cpp"}`.

This is source/compiler policy, not detected target capability. The dialect
does not currently contain a CPU capability record or a selected backend.

### 8.4 Explicit provenance dictionary

Explicit provenance contains exactly:

```text
argument_ranges, call_range, column, file, kind, line, offset, version
```

The closed fields are `kind = "explicit_call"` and `version = 1 : i32`.
`file` is nonempty. Line and column are positive signless `i64` values that fit
MLIR's unsigned location fields; offset is a nonnegative signless `i64`. These
fields must equal the operation's `FileLineColLoc`.

`call_range` is exactly `{begin, end}` with signless `i64`, `begin >= 0`, and
`end > begin`; its begin equals offset. `argument_ranges` contains exactly
three or four `{begin, end}` dictionaries. They are ordered, non-overlapping
under the verifier's half-open ordering rule, and contained in the call range.

### 8.5 Recovered provenance dictionary

Recovered provenance contains exactly:

```text
column, compilation_identity, file, kind, line, offset, proof_ranges,
source_range, source_snapshot, version
```

The closed fields are `kind = "recovered_cpp_loop"` and `version = 1 : i32`.
Source location fields follow the explicit rules. `compilation_identity` is a
nonempty string. `source_snapshot` is exactly `sha256:` followed by 64
lowercase hexadecimal digits.

`source_range` is a nonempty `{begin, end}` range beginning at offset.
`proof_ranges` has at least three exact `{begin, end, kind}` dictionaries.
Each kind is a unique canonical symbol, each range is nonempty and contained
within `source_range`, and the following kinds are mandatory:

```text
outer_loop
accumulator_update
output_store
```

The `outer_loop` range must equal `source_range`. Additional unique canonical
proof kinds are structurally allowed. Binding the digest, compilation
identity, ranges, and effective floating-point options to the actual Clang
source snapshot is the trusted recovered-producer obligation; the dialect
cannot reconstruct that proof from strings.

## 9. Execution intent

The bridge C++ boundary enumerates `Invalid`, `Generic`, `Inference`, and
`Training`, but version 1 accepts only an explicit `Generic` context and emits:

```text
mdsl.execution_intent = "generic"
```

Missing, inference, or training intent fails conversion. No intent grants
immutability, prepacking, caching, mutation, or numerical permission. The core
GEMM verifier is intentionally reusable and does not itself authenticate a
module-level intent; the strict explicit envelope does. Future intents require
their own versioned, verified module contract.

## 10. Deterministic v1 bridge and envelope verifier

The executable bridge follows this sequence:

1. require the complete `explicit-gemm-f32-v1` `BridgeContext` including
   generic execution intent;
2. run the existing Matcore IR v1 verifier;
3. reject any non-F32 operand, output, or accumulation type;
4. register only the `mdsl` and `func` dialects required here;
5. emit sites in capture order without merging operation-local symbols;
6. run normal MLIR/dialect verification;
7. run `verifyMatcoreV1BridgeModule` for the strict explicit envelope;
8. print with debug locations, remove all trailing line feeds, then add exactly
   one final line feed.

The envelope verifier additionally authenticates:

- exact bridge/capture/profile/intent fields;
- producer, source, translation-unit, and integer-width metadata;
- unique ordered site identity and public semantic roots;
- the one-GEMM/one-return site shape;
- operand order and destination-tied return;
- explicit origin and explicit provenance only;
- operation provenance file equal to module `mdsl.source_file`.

`matcore-mlir` requires all of:

```text
--input <capture.v1.json>
--numerical-profile explicit-gemm-f32-v1
--execution-intent generic
```

It accepts optional `--output`; otherwise it writes to stdout. Unsupported or
missing profiles/intents, malformed or non-v1 JSON, same input/output identity,
and bridge errors return nonzero with an actionable diagnostic. The CLI does
not expose recovered construction because recovered-source authentication is a
future frontend milestone.

Normal `mlir::verify` proves core operation well-formedness. It is not a
substitute for `verifyMatcoreV1BridgeModule`, and core verification of a
recovered operation is never executable permission.

## 11. Information-consumption rules

Version 1 consumes no planning choice. The following information must remain
until the named future decision is structurally encoded:

| Information | May be consumed only when |
| --- | --- |
| origin and source provenance | diagnostics/recovery no longer require it and a verified lower representation retains the required source mapping |
| numerical profile | all reassociation, contraction, reduction, approximation, non-finite, rounding, exception, and subnormal decisions are fixed and encoded downstream |
| alignment/no-alias requirement | statically proven or guarded before effects, then encoded in the selected lower operation/pointer contract |
| destination/result identity | bufferization or library/generated-call lowering structurally preserves the overwrite and storage relationship |
| effects and synchronization | use-def plus the lower memory/effect model fully represents ordering and observability |
| symbolic shape and strides | legality, guard formation, specialization, and layout selection have completed or the lower representation retains them |
| target/fallback policy | a legal planner decision is fixed and failure behavior remains represented |
| execution intent | every intent-sensitive legality/lifetime/reuse decision is fixed and represented |

No pass may replace unknown with permissive, infer a numerical permission from
target capability, or treat a requirement as proof. If the next
representation cannot express a required fact, conversion fails.

## 12. Tests and accepted evidence

The reviewed test surface covers:

- exact TableGen op registration and textual parse/print;
- byte-identical golden output and parse/verify/print stability;
- two independent bridges of identical input;
- static and dynamic shapes and operation-local symbols;
- two-site ordering and identity;
- DPS input/init/result tying;
- exact `MemoryEffectOpInterface` effects and non-purity;
- preservation under standard MLIR SymbolDCE;
- every explicit numerical-context field failing closed when missing or
  changed;
- relaxed recovered and strict recovered core verification;
- semantic inequality of strict recovery and explicit eDSL;
- normalized numerical equality of relaxed recovered and explicit eDSL;
- crossed origin, permission, policy, profile, derivation, provenance, and
  source-file rejection;
- malformed ranges, dimensions, alignments, integer widths, types, effects,
  mutability, return wiring, and module metadata;
- CLI determinism, output handling, required options, malformed capture, and
  version mismatch;
- installed leaf-tool golden equality and package path-leakage inspection.

Accepted focused evidence at implementation commits `339ff7b` and `a66ade8`
is recorded in the lane and independent-review reports:

- semantic core: 204 checks, 0 failures;
- CLI contract: 9 checks, 0 failures;
- independent `mlir.semantic.core` and `mlir.semantic.cli`: 2/2 passed;
- installed inspection output matched the reviewed golden;
- no unresolved high- or medium-severity Milestone B finding.

The provenance-sensitive whole-repository suite is an integration-owner gate
after documentation commits. This document does not replace that final clean
tree run.

## 13. Known version-1 limitations

1. Only rank-two, row-major contiguous, host F32 GEMM is supported.
2. Only explicit Matcore IR v1 capture reaches the executable bridge. Recovered
   forms are core representation tests, not a producer or lowering.
3. Only generic execution intent is accepted. Inference and training are
   enumerated but deliberately unsupported.
4. `mdsl.map`, domains, and `mdsl.sin` are not part of the strict Milestone B
   capture bridge. Their separate multi-operation envelope is documented in
   `docs/mdslc/MATCORE_MLIR_COMPOSITION_V1.md`; general reductions, non-GEMM
   contraction semantics, and views remain unsupported.
5. Only explicit GEMM semantic MLIR-to-runtime-dispatch lowering is executable,
   and it delegates to the established CPU runtime. Certified Linalg,
   bufferized, and Vector derivations remain inspection-only; no generated
   execution goes through them.
6. The runtime enforces the `explicit-gemm-f32-v1` control-state guard for the
   authenticated CPU path, including its execution thread and active native
   workers. This does not consume the recorded numerical permissions or
   authorize generated structured/vector execution, and preservation of
   incoming floating-point exception-status flags remains unspecified.
7. The certified buffer seam proves SSA/function-boundary argument-2 identity
   only for its exact admitted program and options. It does not prove physical
   storage identity, a callable memref ABI, or general zero-copy behavior, and
   the public semantic function names are not machine-emission-safe ABI names.
8. Required preconditions are represented but no guard-producing pass exists
   yet.
9. Tensor layout, strides, memory space, mutability, and operation-local
   symbolic relationships are retained in verified attributes in addition to
   builtin ranked tensor types; no custom Matcore tensor type exists.
10. The strict explicit bridge envelope is intentionally not a general
    compositional-module verifier. The separate composition-v1 verifier does
    not loosen this bridge; future recovered or broader multi-op envelopes
    require their own explicit contracts.
11. Unknown extra generic MLIR module/function/op attributes are not globally
    prohibited. They carry no Matcore permission; all version-1 semantic
    dictionaries and closed values documented above remain exact.
12. The compatibility bootstrap producer is accepted only after the same
    verified Matcore IR v1 boundary; this does not elevate it above the native
    Clang frontend.
13. No detected capability, selected implementation, cost, schedule, or
    performance claim is encoded by this dialect version.
14. The dialect C++ library is internal, opt-in, and unfrozen. Only the textual
    versioned inspection contract and verified conversion behavior described
    here are accepted for Milestone B review.

## 14. Authoritative implementation and evidence

- architecture: `docs/adr/0009-mdslc-semantic-compiler-foundation.md`;
- multi-operation contract: `docs/mdslc/MATCORE_MLIR_COMPOSITION_V1.md`;
- TableGen schema: `compiler/lib/mlir/MatcoreOps.td`;
- core verifier/interfaces: `compiler/lib/mlir/MatcoreOps.cpp`;
- bridge/envelope: `compiler/lib/mlir/MatcoreV1Bridge.{h,cpp}`;
- reviewed golden: `compiler/tests/mlir/gemm_capture.semantic.golden.mlir`;
- core/CLI tests: `compiler/tests/mlir/`;
- implementation evidence:
  `docs/mdslc/agent-reports/matcore-mlir-core.md`;
- independent acceptance:
  `docs/mdslc/agent-reports/matcore-mlir-core-independent-review.md`.
