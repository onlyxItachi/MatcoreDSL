# Ordered two-GEMM frontend admission lane

Base: `5f455bacde0959983b2b888f15fd5dabd4b1ceaa`.
Lane: `agent/two-gemm-region-frontend-v1`.

This is an opt-in, native-LibTooling **inspection admission**, not permission to
rewrite, fuse, bufferize, vectorize or execute a region. The ordinary explicit
GEMM extraction/runtime path and public header are unchanged. The integration
owner supplies the CLI, registered Matcore region operations, structural
sequencing verifier and complete validation record.

## Provenance and admission contract

`Options::inspect_two_gemm_regions` requests transient typed candidates in
`Result`. Successful extraction seals the full result and effective options in
the existing immutable `AuthenticatedNativeFrontendEvidenceV1`; the mutable
candidate vector is not an authorization boundary. No second serialized
optimizer IR was introduced.

An admitted pair consists of exactly two direct authenticated GEMM expression
statements, adjacent in one function-body compound or a direct try-body
compound. It has the bounded dependence `C = A * B; E = C * D`, with distinct
output descriptor bindings. Parameters, local objects and qualified nonlocal
object declarations are supported. Transparent automatic local reference
chains are supported only through exact declaration-reference initializers.
Distinct descriptors remain **MAYalias in physical storage**. Repeating an
input descriptor is legal; distinct parameter identities do not prove disjoint
descriptors or storage at runtime.

Bindings preserve three separate facts:

- Actual canonical C++ declaration identity, derived from its physical source
  identity, parsed bytes, declaration kind and byte location.
- Resolved descriptor-binding identity, after only the admitted reference-chain
  proof. This is not a pointer, allocation or noalias identity.
- Exact original source expression and its stage ordinal, plus the original
  capture ordinal/site and region source span/function identity.

Per-stage ordinals are **not** sufficient to encode sequential failure and
commit semantics. The paired MLIR implementation must structurally snapshot,
guard, read and commit stage zero before stage one snapshots/reads. In
particular, no all-region guard hoisting, tensor forwarding across unproved
aliases, or rollback of the first successful output is authorized here.

The scanner admits nonoverlapping pairs greedily. Unsupported neighbouring
calls retain actionable rejection reasons. Host statements, observations,
descriptor mutation, even an empty statement, different functions, nested
scopes/control and unsupported binding forms split admission. RHS-only producer
dependence is intentionally deferred rather than generalized speculatively.

## Capture-context identity

The opt-in path checks every SourceManager physical parsed-file buffer against
its physical file identity and bytes during AST finalization and rechecks the
identity/digest after LibTooling completes. Lookup-only files without a parsed
FileID are not claimed as parsed dependencies. The context digest binds the
loaded native Clang full version, configured compiler/resource directory,
canonical main path and source hash, effective LibTooling arguments and sorted
parsed dependency paths/content hashes. Main-source content remains checked by
the existing extraction path as well.

This distinguishes changed included semantic context despite unchanged main
source/site IDs. It does not change existing weak-wrapper identities, hash the
entire compiler binary, constitute hostile-process cryptographic attestation,
or authorize reuse against arbitrary later filesystem changes. The paired
inspection consumer must bind this sealed context digest. It is not an
executable-artifact cache key.

## Adversarial corrections during implementation

1. Qualified same-spelling declarations: only the region opt-in compatibility
   report retains exact source spelling before legacy name-based checks.
   Ordinary capture retains its historical behavior; region admission uses
   actual declaration identity.
2. Persistent references: `static auto &alias = C` cannot identify the current
   invocation's parameter after the first invocation. Only automatic local
   reference initializers are chased.
3. Macro-expanded declarations: source spelling locations can collapse distinct
   declarations from repeated expansions. Such declaration/function identities
   are rejected rather than guessed.
4. User GEMM definitions, including definitions after the candidate calls,
   reject region admission. User redeclaration attributes cannot inherit the
   trust of a canonical trusted declaration: each noninherited attribute's own
   spelling and expansion origin must also be trusted. The header's own
   annotation macro remains supported. These gates do not change the ordinary
   historical interception path.

## Validation and coverage

Before integration, native implementation and direct-API unit source passed
Clang 21 syntax checks; `git diff --check` passed. No linked or execution test
is claimed by that syntax result. The integration owner runs the new target and
all affected established surfaces, recording exact outcomes in the campaign
checkpoint.

The direct-API test performs real LibTooling extractions, covering consecutive
dynamic GEMMs, binding and stage order, immutable-seal isolation, transparent
reference chains, same-input overlap, unknown physical overlap, qualified
same-spelling declarations, ordinary-mode boundaries, try-body admission,
host/mutation/empty barriers, conditional/loop/scope/function barriers,
RHS-only rejection, call/cast/persistent reference rejection, macro declaration
identity, nonoverlapping three-call scanning, user definitions and attributes,
and changed dependency/option context identity. It deliberately asserts no
runtime noalias, shape specialization, numerical equivalence, generated
execution or performance claim. Structural ordering and paired mutation tests
belong to the independently implemented MLIR lane.

Remaining limitations include nontrivial descriptor expressions, arbitrary
control-flow regions, exception-handler/coroutine modeling, dynamic view/type
frontend extensions, general alias/dependence analysis and execution-artifact
identity. None is silently treated as solved by this bounded admission.
