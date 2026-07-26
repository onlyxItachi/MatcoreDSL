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

## Milestone 7 private-backend evidence update

This additive update records the reviewed private CPU backend state at
`e49260e68f7e43124591e6515bfeb1fe84c3ea74`. It does not supersede the open
questions above, freeze an API, or assert final native-BLAS parity.

- The AVX-512 full-tile experiment is an internal 4x32 body over two adjacent
  v1 packed-B micro-panels. Its checked 4x16 edge path remains separate.
- AVX2 has a private prevalidated 4x16 body, but independent ABBA evidence
  rejected routing the serial executor through it as a performance promotion.
  Serial AVX2 therefore retains the checked entry; the private full-tile body
  remains in the parallel executor after its caller proves complete extents.
- The deterministic parallel planner now represents M-only, N-only, and
  two-dimensional output-task grids. It does not split K or change the
  accumulation order for one output element.
- Private infrastructure can cooperatively prepare disjoint final B panels
  inside one persistent execution-context submission. Final review found the
  broad activation region under-measured, so production selection is dormant.
  This remains within-call candidate preparation with a publication barrier,
  not a caller-owned cross-call parallel-prepacking API.
- The internal packed-B format/provenance model consolidates layout and storage
  authentication, but it still cannot prove that source contents remain
  immutable. It is not a general public transformed-operand object.
- Stable planner variant IDs remain distinct from private microkernel symbols
  and blocking profiles. Whether forced implementation IDs belong in the final
  public request enum remains open.
- Detailed task grids, cooperative-preparation state, and transformed-operand
  lifetime need a versioned structured-report decision before any freeze.
- [GEMM](../performance/kernels/gemm.md) now documents the implemented private
  execution contract. [GEMV](../performance/kernels/gemv.md) and
  [GEVM](../performance/kernels/gevm.md) remain design-only: there is no public
  declaration, capture, IR operation, ABI entry point, planner variant, or
  runtime implementation for either operation.
- Final parity outcome and quantitative envelope remain owned by the
  authenticated Milestone 7 performance report. No conclusion is inferred
  from an individual kernel or packing experiment.

These findings sharpen the later freeze questions around transformed-operand
identity, stable implementation-selection surfaces, structured execution
reports, and operation orientation. They do not justify removing or mutating
an existing exported symbol.

## Milestone 7 bounded disposition

The final locally validated Milestone 7 code checkpoint is `ff483af`; the
candidate bounded disposition pending hosted gates is
[native-blas-parity-v1.md](../performance/cpu/native-blas-parity-v1.md).

The private backend work validates that task geometry, packed-B preparation,
microkernel identity, and effective thread capacity can evolve without
changing the existing 15-function C ABI. Linux Release/Debug, strict C17,
installed source-inaccessible consumer, OpenBLAS-disabled, sanitizer, and
artifact gates remain green. This is useful evidence for keeping those public
surfaces additive.

The result is nevertheless **not ready for API/ABI/backend-contract freeze**:

- the complete native/OpenBLAS parity envelope and full-registry planner regret
  were not established;
- an intermediate four-thread short-wide diagnostic remained below the
  declared 3.0x target, while exact final-checkpoint scaling remains
  unestablished;
- caller-visible transformed-operand identity, content immutability, lifetime,
  invalidation, and size bounds remain undecided;
- exact task grids and capacity-limited requested/actual thread reporting need
  a structured versioned query contract before any freeze;
- forced stable variant IDs versus private kernel/blocking profiles remain an
  open policy decision; and
- GEMV and GEVM remain design-only, with no public declarations, IR
  operations, ABI entry points, planner variants, or executable backends.

Milestone 7 issue #15 and GitHub milestone #5 remain open. A later
exclusive-host forward/reverse collection may update the performance
disposition, but it must not silently mutate an existing export or convert a
private packed representation into a public lifetime contract.
