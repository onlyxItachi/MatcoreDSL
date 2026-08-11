# Public API pre-freeze decision log

Date: 2026-08-11

Status: investigation input for a later, separately approved API / ABI /
backend-contract freeze. Nothing in this document freezes an interface, adds a
public operation, or authorizes source or binary compatibility claims beyond
the versions already tested.

## Scope boundary

Milestone 6 is complete. Milestone 7's bounded partial implementation merged
through PR #16, while its performance tracker remains open. The new primary
work is the compositional semantic compiler and CPU-first beta described by
ADR-0009. Existing exported C symbols and installed consumer behavior remain
compatibility requirements throughout this work. Proposed changes below must
not be made merely to improve a benchmark or simplify one lowering.

The bounded Milestone G contract-resolution review is independently accepted
for the existing versions recorded here. This log remains input to a later,
separately approved public API/ABI/backend-contract freeze; it is not that
freeze. Native-BLAS parity alone neither starts nor completes a freeze.

The bounded CPU-beta decisions recorded below are governed by
[pre-freeze interface evolution policy v1](INTERFACE_EVOLUTION_POLICY_V1.md).
They close ambiguity in existing versions without declaring the broader public
surface frozen.

The current validated public surface consists of:

- valid C++ `.mdsl` source and `<matcore/mdsl.h>`;
- the explicit `matcore::mdsl::gemm(matcore::mdsl::out(...), ...)` operation;
- versioned C descriptors, policy, status, planning, workspace, prepacked-B,
  execution-context, capability, and topology records in
  `<matcore/runtime_c.h>`;
- the installed `mdslc++`, extractor, planner and benchmark tools, plus the
  conditional leaf `matcore-mlir` inspection tool when package capability is
  `ON`; and
- the relocatable CMake package and runtime shared-library boundary.

The installed CMake package also exposes the non-ABI capability variables
`MatcoreDSL_MATCORE_MLIR_AVAILABLE` and
`MatcoreDSL_DEFAULT_SEMANTIC_PIPELINE`. Its helper accepts
`SEMANTIC_PIPELINE capture-v0|matcore-mlir` and fails closed when the requested
pipeline is invalid, unavailable, or incompatible with the selected frontend.
These package controls do not make MLIR operation names or internal semantic
libraries public ABI.

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
10. **Capture/optimizer separation.** Matcore IR v1 is structurally sound as a
    versioned capture/provenance DTO. A deterministic, verified bridge into
    Matcore MLIR is preferable to turning the public JSON schema into a custom
    SSA optimizer.
11. **Semantic/execution separation.** Mathematical operation identity should
    remain independent of execution intent, target capability, selected
    provider, private blocking profile, and microkernel identity.
12. **Destination-tied semantic result.** The semantic GEMM result is the
    post-overwrite value of the explicit write-only destination, not an
    independently allocated tensor. Bufferization must alias those identities
    and preserve the observable destination write.
13. **Preconditions are not facts.** Required alignment and no-alias contracts
    may guide legality, but an optimizer may consume them only after static
    proof or a dominating guard that rejects before output mutation.

## Bounded existing-version decisions for CPU beta

These decisions define the existing v1 interfaces. They do not approve a
general transformed-operand API or the later public freeze.

1. **Packed-B v1 is caller-owned borrowed storage.** The caller owns both the
   original RHS bytes and packed storage. `matcore_packed_b_desc_v1` is only an
   address/shape/blocking/storage snapshot produced by `prepack_b_v1`; it owns
   neither region. Its provenance token authenticates metadata and addresses,
   not contents.
2. **Packed-B v1 invalidation is manual and fail-before-use by contract.** The
   caller keeps source and packed storage alive, at their original addresses,
   and unmodified. Mutation of either region, relocation/move, or deallocation
   invalidates the descriptor. The caller must repack before execution. The
   runtime cannot detect same-address content mutation and makes no hash or
   immutability claim.
3. **Packed-B v1 reuse is serial and synchronous.** Repeated serial calls are
   supported. Concurrent reuse of one descriptor or storage, including across
   execution contexts, is unsupported. Context-backed parallel execution does
   not accept this descriptor. A future shareable transformed operand needs a
   new owner, identity, synchronization, and invalidation contract.
4. **Existing returned C strings are borrowed.** Every non-null returned C
   string is NUL-terminated read-only runtime-static or linked-provider storage,
   valid only until the owning runtime/provider dynamic library unloads. Callers
   copy text that must survive unload and never free or modify the pointer.
   Exact diagnostic sentences are not machine ABI; status codes, enums,
   versioned fields, and explicitly documented stable IDs carry machine meaning.
5. **Existing versions evolve additively.** Matcore IR v1 remains an exact
   capture/provenance schema. Source-operation semantics, serialized schemas,
   and C ABI layouts are not changed in place. New semantics require explicit
   versions, strict conversion, and new `_vN` records/symbols where applicable.

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
4. **Structured diagnostics beyond borrowed strings.** The existing pointer
   lifetime is now explicit, but it does not provide general serialization or a
   foreign-runtime ownership model. Evaluate caller buffers or structured
   diagnostic codes without weakening actionable messages.
5. **General transformed-operand identity beyond packed-B v1.** The v1
   descriptor deliberately provides no content authentication and supports only
   caller-disciplined serial reuse. Any persistent cache, concurrent sharing,
   or general transformed-operand API needs an additive immutable identity,
   generation/lifetime, synchronization, and invalidation model.
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
11. **Numerical policy.** ADR-0009 freezes the internal
    `explicit-gemm-f32-v1` policy: F32 accumulation; contraction allowed;
    reassociation only inside one output's K reduction; implementation-defined
    K order; NaN/non-finite preservation without payload/order guarantees;
    relaxed signed zero; approximate math forbidden; explicit destination
    overwrite; no input/output alias or in-place operand mutation;
    round-to-nearest-ties-even; non-trapping exceptions; gradual subnormals
    with FTZ/DAZ disabled; and no exact exception-status-flag preservation
    guarantee. The internal MLIR encoding, exact source-evaluation profile,
    Linux x86-64 runtime preflight, native-worker admission, and supported
    backend conformance have passed focused independent tests. The eventual
    device-neutral public representation and physical Windows conformance
    remain pre-freeze decisions/gates.
12. **Execution intent.** `generic`, `inference`, and `training` require a
    versioned compilation/execution context. Intent alone must not imply
    immutability, prepacking permission, saved-intermediate lifetime, or
    hidden caching.
13. **Domain and composition.** The internal Matcore MLIR composition-v1 model
    now validates all/slice/indices/predicate domains and effect-aware
    GEMM-to-map use-def composition. It remains inspection-only: there is no
    public map operation or map/sine CPU execution. Future public
    map/reduce/contract contracts still require operation-specific lowering and
    ownership decisions.
14. **Recognition provenance.** The internal recovered-GEMM prototype now
    distinguishes authenticated explicit intent, recognized strict source, and
    source-proven guard-required source. It is analysis/equivalence inspection
    only and cannot authorize rewrite or execution. A future public structured
    diagnostic still must distinguish recognized, legally raised, rejected,
    and preserved source regions without collapsing them into one bit.

## Internal backend abstraction status

The semantic-foundation work has now implemented and independently reviewed
these internal, non-public foundations:

- a compositional Matcore MLIR operation/value representation with generic SSA
  operands/results, effect and alias interfaces, explicit numerical legality,
  and deterministic source provenance;
- a checked Matcore IR v1-to-MLIR bridge that preserves every represented
  field/dynamic-symbol relationship and supplies the verified
  `explicit-gemm-f32-v1` numerical profile;
- closed map/sine composition with all/slice/indices/predicate domains for
  optimizer inspection; and
- sealed recognition/permission evidence and structural-equivalence
  diagnostics for one canonical recovered C++ GEMM.

None of those facts authorize public map APIs, map/recovered execution, or a
public MLIR ABI. The remaining performance and contract concepts should also
stay private until their semantics are proven:

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
- operation-specific arithmetic-intensity and bandwidth models;
- a generic transformed-operand identity carrying layout version, dtype, ISA,
  dimensions, source identity, mutation generation and storage lifetime;
- a versioned execution-intent context separate from target policy and
  detected capability; and
- a structured public recognition report, if one is ever justified, distinct
  from the current internal analysis diagnostics.

None of these internal abstractions should leak microkernel headers or packing
layouts into the installed public include tree.

## Ownership and lifetime disposition

Resolved for the current v1 CPU-beta surface:

- The caller owns packed-B source and transformed storage. Address/metadata
  provenance does not prove contents; mutation, relocation, or deallocation
  requires explicit repacking.
- Packed-B v1 supports synchronous serial reuse only. Cross-context concurrent
  sharing is not part of this version.
- Returned diagnostic/report strings are borrowed NUL-terminated static or
  provider-lifetime storage. Callers copy before the owning dynamic library is
  unloaded. Exact wording is not a machine interface.
- Existing source operations, capture schemas, and C ABI versions follow the
  additive rules in `INTERFACE_EVOLUTION_POLICY_V1.md`.

The later freeze milestone must still answer these broader questions:

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
- Which numerical permissions originate in the language contract, which are
  compilation options, and which may a backend only consume after planning?
- How is the accepted Linux x86-64 rounding/trap/FTZ/DAZ and provider
  conformance contract implemented and physically validated on every future
  supported platform/provider before output mutation?
- Are floating-point exception-status flags intentionally outside the public
  semantic result, and how is that limitation diagnosed?
- How are untouched elements defined for partial-domain transformations, and
  when may a semantic result alias its destination?
- How are MLIR-cloned/transformed operations tied back to authenticated source
  provenance without treating locations as semantic identity?
- Can an inference context prove transformed-weight reuse only through an
  explicit immutable owner, rather than through an intent enum?

Until the remaining general transformed-operand questions are resolved, no
global mutable packed-weight cache is acceptable. The bounded packed-B v1
decision is not permission to infer immutability from an address.

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
7. A versioned numerical-semantics record that can encode the explicit eDSL
   profile without making it a permissive default for recovered source. Any
   unspecified or recovered-source field defaults conservatively to forbidden.
8. A versioned execution-intent record independent from mathematical operation
   identity and detected target capability.
9. A structured recognition report that separates pattern recognition from
   legal permission to replace ordinary C++.
10. A required-precondition record distinguishable from statically proven or
    guard-established alignment/alias facts.

## Operation readiness

| Operation | Language exposure | IR semantics | Executable reference | Optimized variants | Pre-freeze status |
| --- | --- | --- | --- | --- | --- |
| F32 GEMM | existing, validated | typed v1 plus verified `mdsl.gemm` | yes | reference, tiled, AVX2/AVX-512, parallel, optional OpenBLAS | semantic CPU route accepted; performance hardening continues; not frozen |
| F32 GEMM -> SIN map | no public map operation | internal verified composition-v1 | no map execution | none | optimizer/inspection only; not ready for exposure |
| Recovered canonical C++ F32 GEMM | ordinary C++ only | internal authenticated analysis/equivalence | ordinary C++ preserved; no recovered execution | none | analysis-only; no rewrite permission |
| BF16→F32 GEMM | no public eDSL overload | typed/reference contract exists | yes, reference | none runtime-validated | not ready for public exposure |
| I8→I32 GEMM | no public eDSL overload | typed/reference contract exists | yes, reference | none runtime-validated | not ready for public exposure |
| GEMV | no public operation | private design only | no | no | audit input only |
| GEVM | no public operation | private design only | no | no | audit input only |
| ReLU-GEMM | named future operation only | not executable in this scope | no | no | not ready |

## Freeze entry criteria

The following prerequisites are now resolved for their bounded internal or
existing-version scope:

- the Matcore MLIR core, multi-operation domain model, and focused explicit
  CPU runtime-dispatch lowering passed independent review;
- explicit and conservatively recovered GEMM share a verified mathematical
  representation for authenticated analysis/equivalence, while recovered
  forms remain analysis-only and fail closed at the executable boundary;
- Milestone F accepted Milestone 7's reviewed bounded technical limit without
  inventing a parity result;
- the Linux x86-64 explicit-GEMM numerical profile, compile environment,
  runtime FP admission, and supported backend conformance passed focused
  independent review; and
- bounded packed-B ownership/invalidation, borrowed-string lifetime, and
  additive version evolution passed the Milestone G independent contract
  review without changing an existing exported declaration or layout.

Those resolutions do **not** start the separate freeze milestone. Entry still
requires an explicit later authorization plus closure of the broader decisions:

- final CPU-beta Release, Debug, sanitizer, package/relocation,
  source-inaccessible, strict-C17/export, Windows, and hosted consumer gates on
  the exact candidate commit;
- evidence that private blocking, packing, and thread-decomposition evolution
  does not require public request-enum churn;
- a decision for transformed-operand ownership, identity, sharing, and
  invalidation beyond bounded serial packed-B v1;
- device-neutral execution intent, requested/actual resources, numerical
  policy, target policy versus capability, diagnostics, dynamic shape, and
  source-provenance ownership;
- final operation/version support, evolution, deprecation, and compatibility
  guarantees; and
- a fresh independent public API/ABI/backend-contract review with no unresolved
  high or medium finding.

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
bounded disposition merged through PR #16 is
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

## Semantic-foundation pivot update

ADR-0009 establishes an internal architecture freeze without freezing any
installed interface. It keeps Matcore IR v1 as the typed capture DTO and uses
Matcore MLIR as the compositional optimizer representation. Generic SSA values,
effects, aliases, numerical intent, closed map domains, execution intent, and
target context are represented in the internal dialect. The explicit F32 GEMM
semantic route has been validated through the existing CPU runtime-dispatch
boundary; map/domain composition and recovered GEMM remain inspection-only.
These bounded results are prerequisites, not a proposal to freeze the public
contract.

This changes the pre-freeze sequence in two important ways:

1. a parity pass is no longer sufficient to freeze a GEMM-shaped ABI before
   multi-operation semantics are understood; and
2. a parity miss is not by itself a CPU-beta blocker when the deterministic
   planner legally selects a faster authenticated provider.

The freeze review must keep private implementation choices private. In
particular, MLIR operation names, pass pipelines, blocking constants, packing
formats, microkernel symbols, and provider adapter details are not public ABI
merely because they are inspectable compiler internals.
