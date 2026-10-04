# Authenticated publication-to-read forwarding

Status: implemented on the isolated `mdslc/publication-forwarding-v1` branch;
integration, exact-head complete qualification, hosted checks and canonical merge
are still pending. The [implementation report](agent-reports/publication-forwarding-v1.md)
separates executed tests from tests that have only been authored. This is one
bounded target-independent resource optimization, not whole-region fusion,
residency, a new mathematical surface or a performance claim.

## Authority and ownership

The original authenticated frontend-neutral `Program` and exact untransformed
paired MLIR witness remain unchanged. `ClosedHostDerivedPlan` is a separate
compiler-private, in-process immutable derivation. Its private payload owns the
original evidence seal. Only `deriveClosedHostPlan` issues a plan after rebuilding
the exact witness and replaying authenticated source pairing. No editable graph,
serialized attributes, imported certificate or diagnostic proposal issues one.

Consumption by `emitClosedHostV1(evidence, plan)` calls `verifyClosedHostPlan`,
which replays exact pairing and recomputes the complete bounded forwarding list.
The revision, optimization mode, semantic/host identities, all record fields,
list count/order and plan identity must match. A different selected function in
the same source TU cannot substitute its plan. A moved-from plan/evidence is
rejected before payload access. Public plan queries return immutable data and
diagnose moved-from use.

`verifyClosedHostForwardingProposal` is a bool-returning diagnostic/test seam
over an explicitly untrusted vector. It never constructs or returns a plan,
and even a successful answer carries no execution authority. The issuer always
derives its own records, never stores caller proposals. This is not a general
proof interpreter or an overlapping serialized optimizer schema.

The existing compiler-owned orchestration, original frozen host, checked thunk,
primitive issuance, candidate DSO isolation and pinned installed-header/artifact
ownership remain mandatory. The additive private runtime method does not change
`SessionAbiV2` or `ValueAbiV2` object layout. Installation reuses the existing
private-header snapshot and export-map mechanisms; no optimizer header is installed.

## One bounded legality rule

Each record identifies an original read frontier, its resource id, a dominating
publication frontier and the retained immutable value. The successful publication
frontier is that resource's checked version. Generated orchestration still checks
the publication's result before it can reach the forwarded read.

- Every publication invalidates all older resource bindings, including distinct
  descriptors: all host resources remain MAY-alias. The new publication then
  establishes only its own resource/value/version binding.
- Reads, GEMMs and owning observations do not write caller resources, so they
  do not invalidate that binding. Required checks and observations still execute.
- `ShapeIf` is conservatively an entry and join barrier. A branch may forward a
  read after its own dominating publication; incoming, sibling-arm and escaping
  branch bindings are not reused. Both static arms consume their original
  frontier numbers, including each shape-if's non-runtime frontier.
- Resource identity is the same sealed source parameter binding, not pointer
  equality, descriptor equality, descriptor inequality or inferred physical
  disjointness. No no-alias permission is introduced.

This rule deliberately forgoes valid opportunities across branch boundaries or
distinct disjoint runtime descriptors. It requires no general liveness or alias
framework. Retained immutable values survive subsequent publications exactly as
before; mathematical operand roles and per-operation rounding boundaries remain.

## Required runtime checks and frontiers

The generated read remains at its original source frontier. The private
`Session::readForwarded` runs the same original read guard precedence:

1. Sticky status, completion/reentry/platform and increasing frontier checks.
2. Requested rows/columns and element/byte extent representability.
3. Requested shape equals the bound resource descriptor shape.
4. Original descriptor validation: access enumeration, descriptor extent,
   capacity, nonempty pointer/alignment and address-span representability.
5. Only after those checks, retained-value validity and exact dimensions.
6. Nonallocating immutable ownership retention and original-frontier retirement.

The ordinary read-write permission rule is unchanged: a read may use a read-only
descriptor. Empty dimensions still perform representability/access checks but
never dereference null storage. A failure preserves the preexisting output Value,
original failure site and already completed publications/owning observations;
later operations do not run. There is no hidden snapshot or candidate fallback.
Complete caller floating-point controls and status remain under the original
adapter contract. Forwarding performs no floating-point computation.

The omitted snapshot removes allocation opportunities. Allocation exhaustion
can therefore differ from the no-optimization realization, but each outcome must
retain an allowed completed effect prefix. It does not remove a source-required
shape/resource check, move a failure ahead of earlier effects or weaken the
normal-return guarantee that a failed publication leaves its own destination
unchanged. See the [resource contract](FOUNDATION_RESOURCE_DECISION_V1.md).

## Explicit implementation selection

Both single-source and `--program` invocations accept:

```sh
mdslc-region source.mdsl --region pipeline --optimization none -o baseline
mdslc-region source.mdsl --region pipeline --optimization publication-read-forwarding -o optimized
```

`none` is the default. Candidate/target selection remains orthogonal; forcing an
unsupported candidate still fails without fallback. Unknown and duplicate
optimization options are rejected before output publication. Semantic witness
identity remains the same for the same source; private orchestration/helper
identities additionally bind optimization mode and the checked plan identity,
so optimized and baseline implementations are not interchangeable.

## Qualification gates and current evidence

Focused runtime ASan/UBSan execution has passed **67 independent checks**, no
failures or sanitizer diagnostics. The actual private method's mangled definition
was inspected and matches the additive export entry. The source fixture passes
Clang 21 syntax with warnings-as-errors. These are bounded runtime/static checks,
not source-connected driver qualification or a complete compiler test result.

Committed tests require canonical-list/forged-proposal rejection, same-TU
selected-function and new-source mismatch refusal, moved-from plans/evidence,
all-MAY-alias invalidation, unequal nested branch frontier holes, original
source sites, immutable ownership, full read-guard precedence, sticky failures,
empty/overflow shapes, FP state and deterministic allocation-opportunity tests.

The parameterized driver runner compiles exactly the same physical source with
`none` and forwarding. It checks both independent arithmetic/failure oracles and
the actual executable's undefined `readForwarded` reference: present only in the
optimized artifact. Distinct alias reads cannot erase insufficient-capacity
checks, and an intervening partial-overlap publication cannot reuse stale values.
The multi-source and installed package routes repeat that connected discriminator.
Forced candidate unavailability is explicitly reported separately; qualification
may require actual execution rather than accept a fail-closed refusal.

Integration owner must append exact clean candidate SHA/tree, complete Release
and affected sanitizer results, independent acceptance, actual per-policy/device
execution, exact-head hosted results and normal PR/merge identities before this
document can claim a qualified canonical boundary. No timing or speedup follows
from removing an allocation opportunity.
