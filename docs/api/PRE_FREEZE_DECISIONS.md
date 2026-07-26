# Public API pre-freeze decision log

Date: 2026-07-26

Status: investigation input for a later, separately approved API / ABI /
backend-contract freeze. Nothing in this document freezes an interface, adds a
public operation, or authorizes source or binary compatibility claims beyond
the versions already tested.

## Scope boundary

Milestone 6 is auditing CPU execution and Milestone 7 is reserved for
evidence-backed internal GEMM work. Existing exported C symbols and installed
consumer behavior remain compatibility requirements throughout those
milestones. Proposed changes below must not be made merely to improve a
benchmark.

The current validated public surface consists of:

- valid C++ `.mdsl` source and `<matcore/mdsl.h>`;
- the explicit `matcore::mdsl::gemm(matcore::mdsl::out(...), ...)` operation;
- versioned C descriptors, policy, status, planning, workspace, prepacked-B,
  execution-context, capability, and topology records in
  `<matcore/runtime_c.h>`;
- the installed `mdslc++`, extractor, planner and benchmark tools; and
- the relocatable CMake package and runtime shared-library boundary.

## Interfaces that appear structurally sound

These conclusions are provisional. They mean that the audit has not found a
reason to redesign the underlying concept, not that names or layouts are
frozen.

1. **Explicit output and effects.** `out(C)` correctly makes mutation,
   initialization, alias rejection and synchronization visible.
2. **Typed tensor descriptors.** Rank, element dimensions and strides, dtype,
   memory space and mutability belong in an operation descriptor rather than
   implicit ambient state.
3. **Fail-closed policy.** Target and fallback policy must continue to reject
   unavailable variants rather than silently migrate data or select another
   target.
4. **Versioned C ABI records.** `abi_version`, `struct_size` and zeroed
   reserved fields provide a workable additive boundary. No C++ template,
   exception, STL container or implementation-specific C++ type should cross
   it.
5. **Query-before-execute resources.** Workspace size and alignment must remain
   queryable before output mutation. Caller ownership prevents benchmark and
   production paths from hiding large allocations.
6. **Opaque execution contexts.** Persistent worker state is appropriately
   opaque while its requested/actual thread counts, placement and generation
   are reportable.
7. **Stable variant identity plus plan evidence.** A selected implementation
   needs a stable ID, legality reason, resource requirements and explicit
   runtime-validation state. The planner must remain inspectable even if its
   internal cost representation changes.
8. **Separate capability dimensions.** Hardware, OS state, compiler support,
   implementation availability and physical runtime validation must remain
   distinct.
9. **Trusted canonical frontend declarations.** Post-Sema canonical identity,
   exact annotation payload and trusted-header origin remain the correct
   language opt-in boundary.

## Interfaces that may need redesign before freeze

1. **Fixed candidate arrays.** Plan-report v1/v2/v3 structs encode a registry
   count and candidate array directly in the ABI. More shape-specific kernels
   would require another report version even when the semantic operation is
   unchanged. Evaluate a bounded query/iteration interface or caller-sized
   record array.
2. **Request enum growth.** Every forced internal implementation currently
   consumes a public enum value. Consider whether stable string IDs or a
   versioned backend-selection descriptor are the better long-term diagnostic
   and test interface. Forced IDs must never become an accidental promise that
   a private microkernel exists forever.
3. **CPU-specific execution options.** Threading, affinity, SMT and NUMA fields
   are useful, but their placement in a GEMM-specific CPU record may not scale
   to a device-neutral execution contract. Separate semantic operation policy
   from backend execution hints.
4. **Borrowed diagnostic strings.** Process-lifetime strings are simple but do
   not provide a general serialization or foreign-runtime ownership model.
   Evaluate caller buffers or structured diagnostic codes without weakening
   actionable messages.
5. **Packed-B identity.** The v1 descriptor binds source address, shape,
   blocking constants and a provenance token. Address identity alone cannot
   prove immutable contents. Any persistent cache or transformed-operand API
   needs explicit content/lifetime/invalidation ownership.
6. **Parallel prepacking.** The single-thread packed path accepts prepacked B,
   while the parallel context path always packs B again. A future additive
   contract should let workers share a caller-owned, authenticated,
   read-only transformed operand without global cache state.
7. **Execution-context submission model.** The current context serializes
   submissions and exposes a worker generation, but there is no explicit
   asynchronous event, cancellation or concurrent-queue contract. Do not add
   such concepts until a real host/device integration use case exists.
8. **Topology summaries.** Public reports expose aggregate topology values,
   not a caller-sized logical-CPU/core/cache/NUMA mapping. Decide whether
   detailed topology is an introspection API or remains an internal planner
   input.
9. **Alpha/beta and initialization semantics.** Current GEMM is exactly
   `C = A * B`. General BLAS-like scaling would alter read/write effects,
   initialization, alias and numerical contracts and requires an explicit
   language/IR decision.
10. **Dynamic-shape execution.** IR v1 describes dynamic dimensions, but the
    current executable CPU surface is intentionally concrete F32 rank-2
    row-major GEMM. The later freeze must distinguish compile-time constraints
    from runtime descriptor checks.

## Missing internal backend abstractions

The performance audit identifies concepts that should remain private until
their semantics are proven:

- a shape/ISA-specific blocking profile selected independently from a stable
  user-visible variant family;
- a microkernel contract separating a prevalidated full tile from checked edge
  handling;
- an output-tile task graph able to decompose M, N or two dimensions without
  splitting K or changing reduction order;
- a parallel transformed-operand view shared read-only across workers;
- a measured dispatch-cost and serial-packing term in planner estimates;
- a worker-placement record that can represent cache/performance groups in
  addition to socket/core/NUMA identity;
- operation-specific arithmetic-intensity and bandwidth models; and
- a generic transformed-operand identity carrying layout version, dtype, ISA,
  dimensions, source identity, mutation generation and storage lifetime.

None of these internal abstractions should leak microkernel headers or packing
layouts into the installed public include tree.

## Ownership and lifetime questions

The freeze milestone must answer these explicitly:

- Who owns transformed matrix storage, and who proves the source remained
  unchanged?
- Is invalidation manual, generation-based, content-hash-based, or tied to a
  typed immutable object?
- May multiple execution contexts consume one transformed operand
  concurrently?
- Does context destruction wait for all submissions, and can submission ever
  become asynchronous?
- Which diagnostics and report strings survive context or runtime-library
  teardown?
- How are workspace and transformed-storage size limits enforced before
  allocation?
- How are topology/affinity requests represented on Windows and other
  non-Linux platforms without pretending that policy names have identical
  semantics?
- Which numerical-order guarantees apply when thread decomposition or an
  external provider changes?

Until these are resolved, no global mutable packed-weight cache is acceptable.

## Candidate device-neutral contracts

Candidates for later design work, not approved APIs:

1. A semantic operation descriptor containing typed operands, effects, shape,
   layout, accumulation and numerical policy.
2. A target/capability snapshot whose feature dimensions are versioned and
   backend-neutral while target-specific records remain extensible.
3. A plan query returning a stable implementation identity, legality,
   resources, synchronization behavior and structured rejection reasons.
4. Caller-owned workspace and transformed-operand objects with explicit
   provenance and lifetime.
5. An execution context that separates scheduling resources from semantic
   operation descriptors.
6. A selected-plan execution report that distinguishes requested, selected,
   compiled, physically validated and actually executed behavior.

## Operation readiness

| Operation | Language exposure | IR semantics | Executable reference | Optimized variants | Pre-freeze status |
| --- | --- | --- | --- | --- | --- |
| F32 GEMM | existing, validated | typed v1 | yes | reference, tiled, AVX2/AVX-512, parallel, optional OpenBLAS | continue performance hardening; not frozen |
| BF16→F32 GEMM | no public eDSL overload | typed/reference contract exists | yes, reference | none runtime-validated | not ready for public exposure |
| I8→I32 GEMM | no public eDSL overload | typed/reference contract exists | yes, reference | none runtime-validated | not ready for public exposure |
| GEMV | no public operation | private design only | no | no | audit input only |
| GEVM | no public operation | private design only | no | no | audit input only |
| ReLU-GEMM | named future operation only | not executable in this scope | no | no | not ready |

## Freeze entry criteria

The separate freeze milestone should not begin until:

- Milestone 7 has either met its declared native parity envelope or documented
  the bounded technical limit;
- internal blocking, packing and thread-decomposition experiments no longer
  require public request-enum churn;
- existing Linux and Windows installed consumers remain green;
- symbol and struct-layout compatibility tests cover every retained export;
- transformed-operand ownership and invalidation are decided;
- operation/version evolution and deprecation rules are written; and
- an independent ABI/backend-contract review has no unresolved high or medium
  finding.

