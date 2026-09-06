# Closed-region semantic model: implementation lane

Date: 2026-09-06. Canonical starting point:
`a94b01067d390f0b3f997cd09692dba38cdba455`.
Integration branch: `mdslc/closed-region-admission-v1`.
This lane owns `MatcoreClosedRegion.{h,cpp}` and
`matcore_closed_region_test.cpp`; frontend, CMake, harness, adversarial source
fixtures and integration validation are independently owned.

## Established contract

- A transient, typed, frontend-neutral admission record contains no Clang AST
  pointers, public syntax, serialized optimizer schema or execution authority.
- Its derived module uses registered private `mdsl_admission` operations,
  standard Func/Arith operations, ranked f32 tensor values and existing internal
  Matcore descriptor/order types. No Linalg, buffer, vector or execution lowering
  is added. Unknown operations and attributes fail the closed module verifier.
- Resource handles remain distinct from immutable values. All resources MAY
  alias. Every publication advances a conservative global resource epoch;
  subsequent reads preserve that frontier. This epoch is not physical storage
  identity, a noalias fact, or an allocation/copy strategy.
- Read/publication requests a dense row-major f32 resource view. Compatibility,
  capacity, initialized objects, lifetime and race freedom remain required or
  unproven adapter/caller obligations. The opaque test handle is not a public
  matrix-view ABI. Logical tensor types do not acquire physical storage layouts.
- Shapes use explicit unsigned-64 semantics carried by signless i64 values and
  unsigned comparisons, never target-dependent `index`. Read/GEMM retain checked
  representability and shape obligations. Literal extents are bounded to the
  signed-64 tensor-shape domain; symbolic comparisons retain the full 64 bits.
- Each GEMM retains a checked-failure node, including dead results. The profiles
  spell f32 accumulation and operation-boundary rounding, strict increasing-k
  versus within-GEMM reassociation/FMA permission, IEEE special values, signed
  zero and unspecified NaN payload. Cross-GEMM reassociation is forbidden.
  FP-status/trap adaptation is explicitly unresolved and blocks execution.
- Mathematical GEMM is not a fallible provider invocation. Ordered reads,
  checked validation, publication and guaranteed observation carry effect
  dependencies. Publication may partially mutate before failure: no rollback
  or transactional guarantee exists. No actual failure injector is executed.
- Both shape-if bodies remain present, each with an ordered token and a
  conservative resource-state join. Branch-local values cannot escape; a helper
  returning values through branch joins is not claimed.

## Verification and falsifiers

The model verifier checks IDs, dominating values, dimensions, static geometry,
bounded control, profiles and source-site structure. The module verifier checks
registered vocabulary, exact obligations, resource identities/epochs, effect
chains, joins, static extent consistency and numerical invariants.

Paired verification rebuilds the expected module from the supplied admission
record and compares the exact untransformed semantic seam. It is not a general
optimizer-equivalence theorem and does not authenticate an editable record;
the separately owned native frontend seal/re-admission wrapper supplies source
authority. Synthetic unit-test records explicitly identify their non-source-
authenticated origin.

Adversarial controls cover lhs/rhs operand order, stale reads, old immutable
values across possible backing-store publication, guard bypass/discharge,
deleted dead-value checks/observations, branch epochs, signed shape comparisons,
invented static extents, numerical weakening/intermediate rounding, fake
noalias/atomicity, unknown memory-free UB, malformed identities and changed
source identities. The actual legacy CPU lowerer rejects both an ordinary
private module and one carrying forged legacy producer/target/capability/retry
labels, and clears prefilled dispatch records on failure.

## Validation

Exact Clang/LLVM/MLIR 21.1.8 Release, OpenBLAS OFF, root-orchestrated existing
`build-closed-release` build. No system package or toolchain changes.

- Independently executed by this lane after the final layout changes:
  `build-closed-release/bin/matcore_closed_region_semantic_tests`:
  **71 checks, 0 failures**.
- Independently executed by this lane:
  `ctest --test-dir build-closed-release --output-on-failure -j1 -R '^mlir.closed_region_semantics$'`:
  **1/1 passed**.
- Root additionally reported the complete new focused selection **3/3 passed**,
  including **279 frontend/adversarial checks, 0 failures**. This is root-owned
  evidence, not an independent frontend rerun by this lane.

The first manual syntax/build draft required correcting an MLIR constructor,
an ordered container for `mlir::Value`, and a StringRef/Twine test comparison.
Independent review then strengthened unsigned-64 shape preservation, read
extent/type consistency and explicit requested storage-view legality before
the passing checkpoint. An initial direct executable invocation used the wrong
build subdirectory; the correct executable is under `build-closed-release/bin`.

Full regression, sanitizer, hosted CI and source-admission claims belong to the
integration record. No performance, zero-copy, region execution, accelerator,
public API/ABI, arbitrary-view or provider atomicity claim follows from this
inspection-only lane.
