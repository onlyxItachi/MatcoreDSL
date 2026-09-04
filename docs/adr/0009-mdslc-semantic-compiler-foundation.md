# ADR-0009: MDSLC semantic compiler foundation

- Status: Accepted for the internal semantic-compiler architecture after
  independent Milestone A review; implementation proceeds only through the
  declared gates and public API/ABI/backend contracts remain unfrozen
- Date: 2026-08-11
- Scope: Additive standalone `compiler/` semantic optimizer foundation

## Context

The standalone compiler has already proved a valid-C++ frontend, authenticated
explicit GEMM capture, typed Matcore IR v1, deterministic CPU planning, normal
native artifacts, Linux and Windows packaging, and correct CPU execution. The
CPU Performance Deep Audit is complete. Milestone 7 added useful private CPU
hardening but remains an honest partial result: PR #16 merged normally at
`e5069758ad04bdb459de2026cad8498b47fda707`, while issue #15 and GitHub
milestone #5 remain open and no parity-completion tag exists.

That implementation proves the standalone architecture but does not yet
provide a compositional optimizer representation. Matcore IR v1 is a strong,
deterministic capture and provenance contract, but its fixed GEMM roles
(`output`, `lhs`, and `rhs`) and per-operation DTO structure are deliberately
not an SSA/region/use-def system. Growing that schema into a home-grown
optimizer would duplicate upstream compiler infrastructure and make future
multi-operation legality harder to establish.

The public API, stable C ABI growth model, and backend contract are still
explicitly unfrozen. This is therefore the least disruptive point to establish
the semantic optimizer boundary before broad backend expansion.

## Decision

### Objective ordering

Compiler choices follow this lexicographic order:

1. semantic correctness and legality;
2. compositional compiler architecture;
3. reuse of Clang, MLIR, LLVM, and appropriate vendor libraries;
4. fail-closed behavior;
5. a coherent CPU-first beta;
6. authenticated performance.

A lower item never justifies weakening a higher item. In particular, a
benchmark result cannot justify changing numerical meaning, alias/effect
legality, or the frozen Milestone 7 evidence envelope.

### Controlled consumption of information

Lowering is a controlled reduction of implementation possibilities and a
controlled consumption of semantic information. A fact may disappear only
when:

- its meaning is structurally encoded by the next representation; or
- every transformation that requires that fact has already been decided.

The verifier at each conversion boundary must establish this condition. A
bridge never invents a missing fact, treats an unknown fact as permissive, or
drops an unrepresentable fact silently.

### Responsibility split: WHAT, HOW, MACHINE

The architecture has three non-overlapping responsibility layers:

| Layer | Owner | Examples |
| --- | --- | --- |
| WHAT | authenticated capture plus the Matcore MLIR semantic dialect | GEMM, map domain, types, shapes, numerical intent, effects, aliases, provenance |
| HOW | MDSLC legality, planning, structured transformation, scheduling, and library/generated-code selection | Linalg/Tensor/MemRef/Vector and generic GPU substrates, OpenBLAS versus native, fusion legality, bufferization, tiling, vectorization, target constraints |
| MACHINE | LLVM and target-specific dialects plus platform/vendor toolchains | LLVM, NVGPU/NVVM, AMDGPU/ROCDL, target object generation, assembly, and linking |

The semantic dialect never contains a premature implementation name such as
`tensor_core_gemm`, `openblas_gemm`, or a particular microkernel tile. Such a
choice belongs to HOW until the selected lower representation structurally
encodes it.

An upstream dialect is not automatically MACHINE-level merely because it is
downstream. Linalg, Tensor, MemRef, and Vector remain HOW substrates while
alternative layouts, schedules, libraries, or targets are still possible. The
boundary is crossed only when LLVM or a target-specific representation such as
NVGPU/NVVM or AMDGPU/ROCDL structurally commits the relevant machine choice.
The transition may occur operation by operation; it is not a single
module-wide phase switch.

Clang remains the C++ parser, Sema implementation, source manager, and host
compiler. It is not the matrix optimizer. MLIR supplies SSA values, regions,
blocks, use-def chains, symbol infrastructure, parsing/printing, interfaces,
and pass management. MDSLC does not reimplement those facilities.

### Matcore IR v1 remains the capture DTO boundary

Matcore IR v1 remains accepted and versioned. It continues to own:

- authenticated operation identity and canonical declaration provenance;
- exact source locations, rewrite ranges, site identity, and deterministic
  serialization;
- dtype and accumulation dtype;
- rank, static/dynamic dimensions, strides, layout, alignment, and memory
  space;
- mutability, effects, alias requirements, synchronization, and source policy;
- the strict v0-to-v1 compatibility boundary used by the current frontend.

It is not deprecated, replaced, or relabeled as an optimizer IR. JSON v1
remains an inspectable capture artifact. No second JSON optimizer schema is
introduced. Any future capture schema requires an explicit version, exact
parser/verifier, deterministic serializer, and loss-checked conversion.

The existing v1-to-v0 projection and generated CPU pipeline remain available
during migration. The new path is additive until end-to-end equivalence and
artifact gates establish that it can become the normal semantic route.

### Matcore MLIR is the optimizer representation

The MLIR dialect uses textual namespace `mdsl`; implementation code lives
under a Matcore-owned C++ namespace. Initial operations are deliberately small:

- `mdsl.gemm`;
- `mdsl.map` with a single-element region;
- `mdsl.yield` and upstream `func.return` where structurally required;
- only the scalar elementwise operations required by the first canonical map
  proof, initially `mdsl.sin`.

The dialect uses generic SSA operands and results. It does not reproduce the
v1 `Output/Lhs/Rhs` enum as its value model. Built-in ranked tensor, memref,
scalar, and index types are preferred where they carry the exact required
meaning. Matcore attributes/interfaces add only semantics not already encoded
upstream.

The initial GEMM is destination-aware so the explicit `out(C)` mutation is not
lost. Its SSA result is the post-overwrite semantic tensor value tied to that
explicit destination; it is not a newly allocated independent tensor. The
destination is write-only for GEMM and is not read as an initialization or
accumulator operand. Bufferization must alias the result to the destination's
storage. The destination write remains an observable effect and cannot be
removed merely because the SSA result is unused.

The result participates in composition while the destination relationship,
write effect, and required no-alias preconditions remain explicit until a
verified lowering structurally consumes them. The verifier rejects any bridge
that cannot represent the overwrite, result/destination identity, shape,
preconditions, or synchronization contract exactly.

### Deterministic v1-to-dialect bridge

The bridge accepts only a verified Matcore IR v1 module and emits one verified
MLIR module in canonical operation order. For fields that v1 represents, it
must preserve:

- stable site identity and canonical operation name;
- tensor/scalar types and accumulation type;
- static dimensions and per-operation dynamic equality relationships;
- layout, stride, memory space, alignment, and mutability contracts;
- read/write effects, alias requirements, synchronization, and policy;
- source file/range provenance using MLIR locations plus stable Matcore
  attributes where a location alone is insufficient.

Dynamic symbols in v1 are scoped to one operation. The bridge must not
accidentally unify equal spellings from different operations. Unrepresentable
or inconsistent input is a conversion error, never a guessed default. The
dialect verifier runs before printing, transformation, planning, or lowering.

This is not yet a claim that current v1 contains every semantic fact required
for a fully lossless optimizer bridge. In particular, v1 records accumulation
dtype but lacks sufficient numerical-policy granularity. This ADR therefore
freezes the internal `explicit-gemm-f32-v1` policy below. Milestone B must
represent and verify every field rather than infer permission from a target,
implementation, or performance preference.

Recovered ordinary-C++ loops do not inherit the explicit-eDSL policy merely
because their algebra resembles GEMM. Their numerical policy must be proven
from source semantics and compiler options. Until that proof succeeds, raising
is rejected and the ordinary C++ remains unchanged. A bridge may be described
as lossless only after both structural fields and this numerical policy are
represented and verified.

### Canonical `explicit-gemm-f32-v1` numerical policy

The explicit `matcore::mdsl::gemm` F32 operation is a mathematical eDSL
operation, not a promise to reproduce the evaluation order of a handwritten
C++ triple loop. Its canonical semantic profile is:

| Property | Contract |
| --- | --- |
| input/output dtype | F32 |
| accumulation dtype | F32 |
| contraction/FMA | allowed |
| reassociation | allowed only among the K-reduction terms contributing to one output element |
| reduction order | implementation-defined within that K reduction; every K term is included exactly once |
| NaN/non-finite behavior | NaNs are not assumed absent and may not be optimized away; if a NaN participates in a contributing arithmetic path, the corresponding result remains NaN. Payload, sign, signaling state, and which NaN propagates are not guaranteed. Infinity arithmetic follows IEEE behavior for the implementation's permitted contraction and reduction order; no `no-nans` or `no-infs` assumption is permitted. |
| signed zero | relaxed; the sign of an exact zero result is not guaranteed |
| approximate math | forbidden; no approximate reciprocal, transcendental, reduced-precision substitution, or term dropping is authorized |
| rounding mode | round-to-nearest, ties-to-even is required |
| trapping exceptions | unsupported; an environment with unmasked/trapping floating-point exceptions is illegal for this profile |
| exception status flags | incoming flag state need not be preserved and no exact post-call flag set is guaranteed; flags may reflect the selected permitted contraction and reduction order |
| subnormals | IEEE gradual underflow is required; flush-to-zero and denormals-are-zero must be disabled unless a separately represented future policy explicitly permits them |
| mutation/aliasing | the explicit destination is overwritten, is not read as an accumulator input, and must not alias either input; input mutation and in-place operand transformation are forbidden |

Reassociation permission is local: it does not permit moving arithmetic across
another Matcore operation, changing effects, mixing output elements, adding or
dropping K terms, or relaxing dtype conversion. FMA permission is explicit and
does not imply general approximate math.

This profile matches the already validated explicit-eDSL execution model: the
native tiled/vector paths and mature BLAS providers may contract or partition a
GEMM reduction without pretending to reproduce increasing-K scalar evaluation.
Each backend still requires a conformance/legality check; the profile does not
make a provider legal merely because it is available.

The current runtime does not yet perform the complete rounding, trap-mask, and
FTZ/DAZ preflight required by this profile. This ADR records the required
semantics; it does not retroactively validate that missing guard. Before the
Milestone E execution route is accepted, each platform/backend combination
must prove conformance and the runtime must verify every dynamically observable
environment requirement it relies on. An unsupported rounding mode, enabled
trapping exception, or disallowed FTZ/DAZ state fails closed before packing or
destination mutation. Subnormal and non-finite correctness fixtures are part
of that gate.

The profile must not be inherited by recovered C++ loop nests. A loop's
contraction, reassociation, order, non-finite, signed-zero, and approximation
permissions come from its source semantics and effective compiler options. A
strict increasing-K C++ loop therefore cannot be replaced by an
implementation-defined reduction merely because the recognizer finds GEMM
indices. If the recovered policy cannot be represented and honored by an
available lowering, raising is rejected and ordinary C++ compilation remains
untouched.

Deterministic textual output is a test and inspection contract. It is not a
new source-language or persistent JSON schema.

### Required preconditions are not proven facts

Matcore IR v1's alignment and alias fields state requirements that a legal
execution must satisfy. They are not evidence that a particular runtime value
is aligned or that two buffers cannot overlap. The bridge therefore represents
them as required preconditions, not unconditional optimizer facts.

An optimization may consume an alignment or no-alias precondition only after:

- static analysis proves it for the actual SSA values; or
- a dominating runtime guard checks it on every path reaching the optimized
  operation.

A planner may form a conditional candidate from such a requirement, but the
selected implementation cannot execute until its guard succeeds. Dynamic
rejection must occur before packing, destination writes, or any other output
mutation. A check after speculative vector access or partial computation is not
a legal guard. Hoisting or combining guards must itself preserve effects and
observable failure order.

### Numerical semantics

Numerical intent is a first-class legality contract, not an optimizer hint.
The semantic layer must represent or conservatively derive at least:

- required accumulation dtype;
- reassociation permission;
- contraction/FMA permission;
- reduction-order permission;
- NaN behavior;
- signed-zero behavior;
- approximate-math permission; and
- in-place update permission.

Defaults are conservative. Unknown is not equivalent to permitted. A pass may
reassociate, fuse, approximate, change a reduction order, or select a provider
with different observable semantics only when the contract proves that choice
legal. Existing runtime behavior must be audited against the new explicit
contract rather than retroactively declaring every current variant strict.

### Effects, aliases, mutation, and ordering

Operations expose read, write, and read-write effects through MLIR interfaces
and Matcore attributes where necessary. Alias/no-alias and destination/in-place
relationships remain explicit. SSA dependencies express necessary data order;
the dialect does not add a total program order where dependencies and effects
are sufficient.

Conversely, an optimizer may reorder operations only when use-def, effect,
alias, synchronization, and numerical contracts collectively prove it legal.
Volatile, atomic, externally observable, or otherwise unsupported source
effects block raising or transformation.

### Domain semantics

`mdsl.map` has an explicit domain, initially one of:

- `all`;
- a verified slice;
- explicit indices; or
- a predicate/mask.

The word `mask` is reserved for active-element predicate semantics. Whole
tensor application is `domain(all)`. Domain verification checks rank, bounds,
index type, predicate shape, effect compatibility, and result shape. A partial
domain may not silently define untouched elements; preservation or destination
semantics must be explicit.

The first multi-operation proof is semantic GEMM followed by `sin` over
`domain(all)`, plus one partial-domain fixture. It is a composition/legality
proof, not authorization for a broad operation catalog or fusion optimizer.

### Recognition is not permission

Explicit annotated `matcore::mdsl` calls remain the strongest capture signal.
Ordinary C++ idiom recognition is a separate, conservative path with two
predicates:

1. a source region is recognized as a mathematical operation; and
2. replacing that region is legal under dependence, alias, numerical, effect,
   optimization-barrier, and source-range rules.

Recognition without permission leaves ordinary C++ unchanged. The compiler
may emit an inspectable diagnostic such as `recognized: gemm; rewrite:
rejected; reason: optimization barrier`, but recognition failure is not a C++
compilation failure. Macro, volatile/atomic, unsafe alias, dependent-template,
unsupported control-flow, and user-barrier cases fail closed.

One canonical, non-dependent, ordinary C++ GEMM loop nest will be prototyped.
It raises directly to the same `mdsl.gemm` representation as explicit capture;
the frontend does not rewrite it into a textual eDSL call. Equivalence is
semantic and provenance-aware, not byte identity.

### Execution intent and target context

Mathematical semantics remain separate from execution context. Versioned
compilation context represents at least `generic`, `inference`, and `training`.
The same `mdsl.gemm` operation is used for each. An intent may affect legal or
profitable planning only when required lifetime, mutation, saved-intermediate,
or reuse facts are explicit. `inference` alone never proves weight immutability
or authorizes a hidden transformed-weight cache.

Source policy and detected capability are also separate:

- policy records user constraints such as required target and fallback mode;
- capability records discovered architecture, OS state, compiler support,
  implementation availability, and physical validation.

Neither is baked into mathematical operation identity. A target-specific
choice becomes implicit only after a selected target lowering structurally
encodes it.

### CPU-first lowering proof

The first target consumer is CPU. The new semantic route must prove:

```text
valid C++
  -> authenticated capture
  -> verified Matcore IR v1
  -> verified mdsl MLIR
  -> legal CPU plan/lowering
  -> existing validated runtime or structured upstream lowering
  -> ordinary object/executable
  -> correct execution
```

The proven native planner/runtime remains a legal lowering destination. It is
not replaced merely to exercise MLIR. An initial bridge may select the existing
C ABI/library route, provided the decision is explicit, loss-checked, and
tested. A later structured Linalg/Vector/LLVM route must earn acceptance with
the same artifact and correctness gates.

In that route, Linalg, Tensor, MemRef, and Vector are still HOW-level
representations. The MACHINE boundary begins only after lowering commits to
LLVM or an equivalent target-specific dialect/toolchain contract.

Native/OpenBLAS parity remains valuable evidence, not the definition of the
CPU beta. The planner selects the best validated legal candidate in its
supported search space and continues to report when that candidate is
OpenBLAS.

### Coherent isolated MLIR 21 toolchain

The standalone frontend already uses Clang and LLVM 21.1.8 from Ubuntu package
revision `1:21.1.8-6ubuntu1`. The system installation contains that coherent
LLVM/Clang surface and MLIR 22, but not MLIR 21. An APT simulation showed that
a system-wide MLIR 21 development install would remove the existing MLIR 22
surface. The milestone therefore extracted the exact `libmlir-21`,
`libmlir-21-dev`, and `mlir-21-tools` Debian payloads at
`1:21.1.8-6ubuntu1` into a user-owned, versioned toolchain prefix. It did not
mutate the system package database.

The extracted payload is 761 MiB. Its package SHA-256 values are:

- `libmlir-21-dev`: `ee47ca5eb635afc6d482b683a3d250541ddd446b02c1a66c6dc89743243ae1fd`;
- `libmlir-21`: `5fec86b613126963f0247c6c65ce112f26da96a18bf2d6958534569fc939ba97`;
- `mlir-21-tools`: `b33d0a9ede6939be48580c0ed12faa38e4a26b0e9a2736e8f9f7e4f55c88f397`.

The development prefix is a configure input, not an installed-package or
runtime path contract. It must never be hardcoded into generated artifacts,
installed CMake exports, diagnostics intended for users, or ABI records.

Matcore MLIR support therefore has a hard configure-time coherence gate:

- compiler executable, LLVM headers/libraries, Clang headers/libraries, MLIR
  headers/libraries, TableGen tools, and CMake packages must all be 21.1.8;
- discovery uses supported LLVM/Clang/MLIR CMake packages and imported targets;
- the MLIR feature is isolated under `compiler/` and can be disabled explicitly
  for the already-supported non-MLIR compatibility build;
- a requested MLIR build fails clearly if the coherent surface is absent;
- it never silently links MLIR 22 or changes the legacy root CMake contract,
  which separately requests MLIR 18.1.3.

The toolchain lane configured with the extracted MLIR CMake directory and the
system LLVM/Clang 21 CMake directories. It validated both an in-process
`clang-cpp` + MLIR + LLVM executable and a TableGen-generated toy dialect/op.
It also linked and ran a narrow static `MLIRIR`/`MLIRSupport` executable with no
`libMLIR` dependency and no development-prefix RUNPATH. Production targets
should prefer the narrow imported MLIR components they consume rather than a
monolithic shared library. No LLVM source build is required.

## Milestone dependency graph

```text
A Semantic architecture freeze
  -> B Matcore MLIR core
       -> C multi-op/domain semantics
       -> D explicit/recovered equivalence prototype
  B + C + current CPU planner/runtime
       -> E CPU MLIR lowering proof
  existing frozen performance contract -----------> F M7 evidence closure
  A + B + C + D + E + bounded F
       -> G pre-freeze contract resolution
  E + F disposition + G + full product gates
       -> H CPU beta
```

Milestone F may run in parallel once an exclusive host is available, but it
cannot change semantic contracts or block architectural work merely because a
handwritten kernel trails OpenBLAS. G requires a bounded, honest F disposition,
not necessarily native parity. H does not begin public GPU expansion.

## Required design artifacts

ADR-0009 is the architecture source of truth. Later lanes must make their
operation-level contracts reviewable in focused documents rather than hiding
them only in TableGen or C++:

- `docs/mdslc/MATCORE_MLIR_DIALECT_V1.md`: operation/type/attribute syntax,
  interfaces, invariants, canonical examples, and verifier failures;
- `docs/mdslc/NUMERICAL_SEMANTICS_V1.md`: detailed encoding/tests for this
  ADR's canonical explicit-GEMM policy, source-derived recovered-loop policy,
  and provider/lowering conformance;
- `docs/mdslc/MATCORE_V1_MLIR_BRIDGE.md`: exact field mapping, dynamic-symbol
  scope, provenance, deterministic printing, and rejection boundary;
- `docs/mdslc/CPP_GEMM_RECOGNITION.md`: recognized form, permission proof,
  barriers, preservation behavior, and diagnostics; and
- `docs/mdslc/CPU_MLIR_LOWERING.md`: legal plan boundary, existing-runtime
  reuse, structured-upstream alternatives, artifacts, and package behavior.

These are proposed implementation documents, not additional persistent IR
schemas or public API specifications.

## Merge gates

| Milestone | Required before normal merge |
| --- | --- |
| A | repository/GitHub truth recorded; ADR, roadmap, status, and pre-freeze log agree; coherent toolchain gate documented; independent architecture review; docs/hygiene checks |
| B | coherent MLIR 21 configure; generated dialect builds; parser/printer round-trip; deterministic v1 bridge with reviewed explicit-GEMM numerical policy, destination-tied result, and precondition/fact distinction; source-derived numerical proof required for recovered loops; verifier negatives for every represented contract; no v0/v1 regression; independent semantic review |
| C | GEMM-to-SIN(all) and partial-domain goldens; SSA/use-def verification; domain, effect, alias, mutation, and numerical negative tests; deterministic round-trip; no broad op catalog |
| D | explicit and canonical-loop semantic equivalence; recognition-only diagnostics; alias/dependence/numerical/barrier/macro/volatile/atomic fail-closed fixtures; unchanged ordinary-C++ artifacts on rejection |
| E | `.mdsl` through verified MLIR to legal CPU route; static proof or dominating runtime guards for alias/alignment; round-to-nearest, non-trapping, gradual-subnormal environment checks before mutation; backend numerical conformance; independent oracle; ordinary object and executable inspection; forced-illegal failure; Release, Debug, supported sanitizers, install, relocation, and external consumer on Linux; Windows compatibility plan/gate remains green |
| F | unchanged authenticated forward/reverse envelope on a quiescent host, or an explicit bounded technical-limit report; no benchmark-contract mutation; issue #15 updated honestly |
| G | transformed-operand ownership/identity/lifetime/invalidation, report iteration, variant identity, execution context, dynamic shape, diagnostics ownership, execution intent, and operation evolution decisions reviewed; compatibility tests cover retained exports |
| H | clean Release/Debug/sanitizer/package/consumer/Windows/native-artifact/IR/recognition/planner/performance-sanity/hygiene suite; independent adversarial review; beta claim matches executed scope |

No milestone is accepted solely from an implementing agent's self-review.

## Branch and worktree plan

Milestone A starts from clean `main` at the authenticated pivot and uses
`mdslc/semantic-compiler-foundation-v1` in an isolated sibling worktree.
Implementation lanes have non-overlapping ownership:

- architecture/ADR and semantic invariants;
- dialect/bridge/verifier;
- frontend recognition and permission;
- CPU lowering/integration;
- Milestone 7 performance evidence; and
- independent adversarial review.

Focused commits are integrated into the milestone branch and merged normally
only after its declared gates. Subsequent B--E work may remain in the same
semantic-foundation branch when commits preserve those boundaries, or use
short-lived child branches from its reviewed tip. Milestone 7 evidence remains
on its own issue/evidence lane. Contract-resolution and beta publication use
new branches from then-current clean `main`; they do not rewrite or rebase the
accepted history.

## Pre-freeze rework risks

The highest semantic and ABI rework risks are:

1. mapping explicit overwrite/destination effects into tensor SSA without
   losing observable mutation;
2. preventing alignment/no-alias preconditions from being treated as facts
   before proof or a dominating guard;
3. specifying and dynamically enforcing the floating-point environment tightly
   enough that existing native and
   external-library variants can be proven legal;
4. representing dynamic symbol equality without accidental cross-operation
   unification;
5. distinguishing source recognition from permission to remove a loop;
6. expressing partial-domain untouched elements and in-place behavior;
7. preserving exact source/provenance identity through MLIR cloning and
   transformations;
8. preventing execution intent from becoming an implicit immutability or cache
   promise;
9. preventing target policy, detected capability, and selected implementation
   from collapsing into one field;
10. transformed-operand source identity, ownership, cross-context sharing,
   lifetime, invalidation, and size bounds;
11. fixed candidate arrays and public forced-variant enums coupling ABI growth
    to private microkernels;
12. structured diagnostics and requested-versus-actual resource reporting; and
13. operation/version evolution before a public beta contract is named.

These are design gates, not reasons to leak private packing or microkernel
details into public headers.

## Rejected alternatives

- Replacing Matcore IR v1 with unversioned MLIR text: loses the accepted
  capture/compatibility contract.
- Evolving the JSON DTO into a custom SSA/region IR: duplicates MLIR and
  creates overlapping optimizer representations.
- Lowering explicit GEMM directly to a CPU/GPU implementation in the
  frontend: consumes alternatives and semantic facts too early.
- Recognizing ordinary C++ loops by source text or treating recognition as
  permission: risks semantic miscompilation.
- Encoding inference/training or CPU/GPU as different mathematical GEMM ops:
  confuses WHAT with HOW.
- Linking installed MLIR 22 into the Clang/LLVM 21 frontend or changing the
  legacy MLIR 18 dependency: creates an incoherent toolchain and unrelated
  regression risk.
- Blocking CPU beta on handwritten BLAS parity: mistakes an R&D metric for the
  semantic product contract.

## Consequences

The project gains one documented compositional optimizer representation while
preserving its verified capture DTO and CPU product path. The near-term cost is
a strict MLIR dependency gate and more explicit numerical/effect contracts
before fusion or recovered-idiom replacement is allowed.

This ADR does not freeze the public API/ABI/backend contract, claim MLIR
lowering is implemented, complete Milestone 7, expose GEMV/GEVM, or authorize
GPU work. Those claims require their own executed gates.
