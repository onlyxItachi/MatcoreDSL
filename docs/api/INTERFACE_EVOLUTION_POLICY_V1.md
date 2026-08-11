# MatcoreDSL pre-freeze interface evolution policy v1

Date: 2026-08-11

Status: active compatibility policy for CPU-beta hardening. This policy does
not freeze the public API, ABI, backend contract, operation set, or release
cadence. A final freeze requires a separately approved milestone and review.

## Purpose

This policy prevents an unfrozen interface from becoming an unversioned one.
Existing validated consumers must keep working while semantic operations,
capture formats, optimizer representations, execution resources, and backend
implementations evolve. New requirements are introduced through explicit,
additive boundaries rather than by changing the meaning or layout of an
existing version in place.

## Compatibility domains

The project versions these domains independently:

1. source operations and their canonical annotation contracts;
2. serialized Matcore capture schemas;
3. internal Matcore MLIR semantics and bridges;
4. public C ABI symbols, records, enums, and stable variant identities; and
5. human-readable diagnostics and inspection output.

A matching numeral in two domains does not imply that they share a lifecycle.
Every converter or bridge validates both its input version and its destination
contract.

## Source-operation evolution

- The existing canonical `matcore::mdsl::gemm` declaration, annotation, and
  F32 rank-2 overwrite semantics are retained for existing source.
- A new operation, overload, or materially different semantic profile needs an
  explicit canonical declaration and annotation/version contract. It must not
  reinterpret an existing call silently.
- Source recognition and permission to replace ordinary C++ remain separate.
  Recognition failure or failed legality preserves ordinary C++ behavior where
  the source contract permits it.
- A source operation may be deprecated only after a documented replacement and
  migration path exist. Deprecation begins with an actionable compiler
  diagnostic; removal requires a separately approved compatibility break.

No removal date or final source-compatibility guarantee is declared before the
public freeze milestone.

## Capture-schema evolution

- Matcore IR v1 remains the exact, deterministic capture/provenance DTO for its
  documented operation surface. Existing required fields, field meanings,
  verifier rules, canonical serialization, and provenance interpretation are
  not extended or weakened in place.
- A semantic fact that v1 cannot represent requires a new explicitly versioned
  schema or a verified internal bridge input. A new serialized schema must have
  strict parsing, deterministic serialization, a dedicated verifier, and
  explicit loss-checked conversion rules.
- Compatibility v0 and typed v1 remain distinct. Conversion never relabels one
  schema as another and never supplies a permissive default for information the
  destination requires.
- Unknown schema versions fail closed. Readers do not guess from fields or
  accept trailing undocumented semantics.

The internal Matcore MLIR dialect may evolve behind its verified bridge. Any
persistent byte format or externally accepted form still needs an explicit
version; internal SSA evolution does not silently change Matcore IR v1.

## C ABI evolution

- Existing exported symbols, parameter order, calling convention, record size,
  field offset, enum value, status-code value, and documented semantics are not
  changed in place.
- `abi_version` and `struct_size` are validated exactly for each existing
  record. Reserved input fields remain zero until a new record version assigns
  meaning; they are not an implicit extension mechanism.
- Growth uses a new `_vN` symbol and/or a new versioned record. The old symbol
  remains available with its old behavior unless a separately approved major
  compatibility break removes it.
- Fixed candidate arrays in plan reports v1/v2/v3 remain fixed. Registry growth
  requires a new report version or a future bounded query/iteration API; it
  must not overrun or reinterpret an older array.
- Published enum and status numeric values are never reused. Unknown inputs
  fail closed.
- Stable variant IDs identify the documented implementation family. Private
  microkernel symbols, blocking profiles, and task geometry are not public
  identities and may change without consuming public enum values.
- Additive APIs must preserve the stable C boundary: no STL, template,
  exception, or compiler-specific C++ object crosses it.

## Diagnostics and string lifetime

Every non-null `const char *` returned by the runtime C ABI is:

- NUL-terminated;
- read-only and borrowed;
- backed by runtime static storage or linked-provider storage; and
- valid only until the owning runtime or provider dynamic library is unloaded.

The caller does not free or modify these strings and must copy any value needed
after library unload. Exact status messages, legality reasons, selection
reasons, and provider display/configuration text are human-readable diagnostics,
not machine ABI. Programs branch on status codes, enums, versioned structured
fields, and explicitly documented stable IDs. Changing diagnostic wording alone
does not require an ABI version.

A future caller-buffer or structured-diagnostic interface may be additive. The
borrowed-pointer versions remain subject to the rules above.

## Deprecation procedure

Before deprecating a public source or C interface, a proposal must record:

1. the exact operation, symbol, record, field, enum, or stable ID affected;
2. the semantic reason and compatibility risk;
3. the supported replacement and a mechanical migration where possible;
4. the first version that emits a diagnostic;
5. package and installed-consumer coverage for old and new paths; and
6. the separately approved release or compatibility boundary at which removal
   may be reconsidered.

Deprecation does not permit silent fallback, enum reuse, schema relabeling, or
removal of an existing export. Before the final public freeze, the project does
not promise a fixed deprecation window; it promises explicit review, an
actionable migration path, and no unannounced in-place semantic change.

## Deferred decisions

This policy deliberately does not decide:

- a general transformed-operand object or cross-context sharing model;
- a caller-sized plan-report iterator;
- a device-neutral execution context;
- public execution-intent records;
- public dynamic-shape constraint representation; or
- final version-support durations and release guarantees.

Those remain inputs to the later public API / ABI / backend-contract freeze.
