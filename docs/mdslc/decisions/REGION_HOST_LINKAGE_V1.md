# Experimental region host linkage

Historical design checkpoint: the subsequently connected implementation is
described in the [region compiler guide](../REGION_COMPILER_V1.md); retain the
original pending-evidence statements below as decision provenance.

Status: selected implementation direction; connected installed-driver proof is
still pending. This record does not claim a released frontend or stable ABI.

## Decision

Keep an ordinary named C++ function as the host boundary. Admit its mathematical
body into frontend-neutral closed-region semantics, compile the immutable
original host translation unit with Clang before LLVM optimization, and replace
only the authenticated function's LLVM body with a checked ABI thunk to compiled
orchestration. Keep the function's identity, callers, addresses and host cleanup
context. Do not rewrite host source text.

The orchestration compilation is a separate compiler-owned translation unit.
User include paths, macros and preprocessing flags must not enter that unit or
alter private runtime layouts. Public owning results contain scalar state and
opaque ownership handles, not standard-library containers. These are separate
requirements: a matching host/helper ABI does not authenticate a private runtime
layout, and identical header bytes do not authenticate declarations modified by
ambient C++ attributes or out-of-line user definitions.

## Falsifiers and observations

* **OBSERVED:** replacing `void region() {}` with a longer same-line body changes
  a subsequent `__builtin_COLUMN()` from 46 to 113 under Clang 21.1.8. Both
  programs compile and execute. Original source SHA-256:
  `2459e19a80adad3a812c3f0958f7de96289acc3c47dd97e8d8788dae560938f7`;
  replacement:
  `19ecfe4591d7d4624e20efa8ce8eb47fcff7032b487b18ae13da97d58eb8684b`.
  The throwaway pair is retained locally in
  `/tmp/mdslc-host-location-falsifier.4Cdnoh`.
* **PROVEN WITHIN A BOUNDED CONTRACT:** `frontend.frozen_host_codegen` compiles
  original immutable host inputs through in-process Clang to LLVM IR and an
  actual executable. Source column, `std::source_location`, included constants
  and ordinary host construction/destruction match direct Clang execution.
  Mutating a captured dependency prevents any LLVM result from being returned.
* **OBSERVED:** writing intermediate artifacts into a captured include-search
  directory invalidates its recorded metadata. The corrected implementation
  stages outside the input closure; it does not exempt compiler writes from
  input authentication. Final output publication must follow the last input
  recheck and must not overwrite an input or dependency.
* **UNSUPPORTED:** arbitrary source rewriting plus `#line` repair is not a proof
  of preserved columns, macro expansion context or source-location behavior.
* **UNSUPPORTED:** LLVM verification, opaque pointer equality, or equal record
  size alone proves neither the source declaration contract nor its complete
  function ABI.

## Connection invariants

1. The selected entry, parameter ordering and concrete value-helper bindings
   come only from replayed native admission. Serialized IR and source-provided
   names are not execution certificates.
2. No LLVM optimization may consume the source-only body before replacement.
   Source-level Clang/Sema behavior outside the body remains authoritative.
3. Host and helper must have the same target/data layout, calling convention,
   complete structural ABI and return/parameter ABI attributes. Typed `sret` and
   `byval` contracts must be checked, not discarded.
4. Original body-derived optimization facts must not survive on the thunk or
   direct call sites. Unsupported source effect promises fail admission.
5. Value-only helper definitions may be retired only when the sealed helper
   subgraph has no remaining host users. Shape helpers retain ordinary host
   meaning and need not be removed.
6. Failed linking or verification returns no partially modified host module.
   LLVM type identities are context-owned: cloning modules in a shared context
   is not isolation from link-time type mutation.
7. Actual linked numerical candidates retain their own legality, numerical,
   provider, storage and failure contracts. Host orchestration is not authority
   for arbitrary user-supplied LLVM IR or target code.

## Remaining evidence

The ABI thunk, public admission, private runtime build binding, installed driver,
real source-to-generated executable and installed consumer must pass connected
adversarial tests before this decision is a product capability. Existing native
and provider routes remain independently validated compatibility mechanisms.
