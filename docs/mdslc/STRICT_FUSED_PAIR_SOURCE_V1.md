# Bounded source-connected strict fused pair v1

## Scope and authority

This explicit realization is selected with
`--optimization strict-fused-pair --candidate generated-strict`. `none` remains
the default. Publication-read forwarding remains a separate selection; the
two optimizations are not composed implicitly. Automatic, native, ISA-specific,
permitted, provider and GPU candidates are incompatible with the new selection.
The checked issuer/adapter is initially Linux x64, Clang/LLVM/MLIR 21.1.8 only.
There is no new source syntax, mathematical operation, broad DAG optimizer,
target dispatch policy, residency claim or performance claim.

The original sealed source Program and exact untransformed whole-region witness
are unchanged. The immutable in-process `ClosedHostDerivedPlan` owns that
evidence, binds its complete semantic/host identities, revision and explicit
mode, and records the original pair/value/Read provenance. Consumption replays
source authentication and recomputes the complete ordered canonical list.
Unknown modes, stale/moved evidence, a different selected function, missing or
forged fields and count/order differences cannot issue or consume authority.
The proposal verifier returns only a diagnostic boolean; caller vectors are
never stored. Neither a serialized module, inspection verifier, editable
Transform program nor imported LLVM body is an execution-plan issuer.

Eligibility is exactly two adjacent semantic operations, `C=gemm(A,B,strict)`
and `E=gemm(C,D,strict)`, in one lexical body. A/B/D must be dominating immutable
Read results before the first operation. C must have exactly one retained
semantic use: the second operation's lhs. The recursive count includes every
GEMM/publication operand, Read dimension and ShapeIf condition in both arms.
Rhs carry, C*C, live C dimensions, multiple uses, late external reads and any
operation/barrier between the pair block this window. Independent nonoverlapping
windows may be selected; this is not a dataflow scheduling framework. An explicit
request with zero eligible pairs fails compilation, never silently falls back.

Canonical `rows`/`cols` and pure helper Shape bindings are already eliminated by
authoritative admission when dead. A discarded pure query has no original
runtime operation, guard, effect or frontier. It may therefore remain eligible.
No parallel Sema query ledger was added: only actual retained Program references
must be counted, including references in otherwise untaken arms.

## Original guard and failure frontiers

The generated helper calls the private
`gemmStrictFusedPair(f1,f2,A,B,D,first,second,E)`. Both original frontier/source
records remain, including helper call stacks and static ShapeIf holes. There is
no C Value declaration/materialization. The adapter requires consecutive positive
frontiers without wrap; malformed private calls fail before output mutation.

At f1, preserve the original order: begin/sticky/reentry/platform, A/B validity,
numeric-enum validity, first contraction shape, original candidate legality and
the narrower strict-pair compatibility/availability, then FULL logical C signed
dimensions, product and byte-span extent. Removing full C allocation is a
permitted realization change under resource-contract item 8, not permission to
drop its extent guards. Enter, validate, check and completely restore f1's own
FP scope before retiring its logical guard frontier. D and the second numeric
profile are not inspected before that retirement. Separate ActiveCall scopes
prevent the second logical begin from becoming reentry.

At f2, check D validity, numeric-enum validity, the second contraction shape and
candidate/strict-pair legality before checked private E allocation. Allocate
bounded scratch only for work that actually invokes the leaf. Enter f2's own FP
scope, execute exactly the trusted leaf, check controls, restore all caller
controls/status, and only then issue E. Every failure leaves the caller's old
output handle unchanged; assignment may alias any input handle safely.
Earlier publications and owning observations survive either failure. Failure
is sticky and no later source operation or allocation is performed.

For example A[INT64_MAX,0], B[0,2], D[2,0] must fail C's extent at f1 despite the
empty inputs/final result. C[INT64_MAX,1] also fails the byte-span bound. Conversely
C[INT64_MAX,0] is legal and E[INT64_MAX,2] fails at f2. All-zero huge dimensions
with an empty final result are legal and must finish without an enormous row
loop. Candidate availability and both original FP scopes are still required on
empty routes.

## Trusted private realization

The issuer accepts no user module. It emits the five canonical dense rank-2
memref C-interface descriptors ordered A[M,K], B[K,N], D[N,P], E[M,P] and
workspace[min(4,M),N]. This fixed four-row panel is a private HOW recipe. E and
scratch are fresh checked adapter-owned storage, mutually disjoint and disjoint
from inputs; inputs may alias. Capacities, signed/byte products, offset zero,
row-major strides, live descriptor objects and source object/race preconditions
remain mandatory. There is no unchecked MLIR tensor allocation or full C buffer.

The adapter skips the leaf when E is empty, after both original logical guards.
This avoids an overflow/hang from `i+=4` at huge zero-output M. For nonempty E,
its checked byte extent bounds the row increment. When N=0 the consumer reduction
is +0 and the adapter uses initialized E. When K=0 but N>0 and E is nonempty,
it MUST run the consumer: producer +0 multiplied by Inf/NaN is NaN, not +0.
Null pointers for empty A/B are legal and the issued leaf must not dereference
them. No single-GEMM zero-K shortcut is borrowed for this pair.

Each producer result is rounded/stored as f32 before consumer load; both
contractions retain their separate increasing-order multiply/add and positive
zero initialization. No FMA, reassociation across the pair or altered operand
order is permitted. Stage graph equivalence and strict topology checks, exact
LLVM ABI/pointer/arithmetic checks, and the actual optimized-object call gate
are issuer obligations, not source proofs from inspection attributes.

Deferred physical producer arithmetic is justified only for the pinned strict
CPU no-recoverable-error/control-preserving leaf. Its optimized object may call
trusted conforming memset for bounded private zero fills: no arbitrary host
effects, recoverable failure or FP-control changes are permitted. Arbitrary libc
interposition is excluded, like arbitrary allocator hooks in the existing
contract. Providers, GPU adapters and injected test callbacks do not satisfy this
argument. The f2 post-control check is not a general theorem attributing arbitrary
two-stage candidate failures. Restoration failure still terminates instead of
returning with corrupted caller state.

No Session/Value record layout changed. Private Implementation code 14 is
appended for a real fused invocation. The f1 report is honestly guard-only:
actual none, no invocation attempted and no intermediate Value issued. f2
reports one fused invocation, or the existing empty/zero-reduction realization,
and only claims a Value after success. No original-two-invocation/materialized-C
claim is made. Helper/artifact identity includes the explicit mode and checked
plan hash; installed header snapshots and intrinsic private DSO ownership remain.

## Acceptance evidence and remaining gates

The contract-first independent runtime test was authored before reading this
adapter implementation. It ran 204 checks with ASan/UBSan and genuine issued
single/pair ASan objects, zero failures and no sanitizer diagnostics. This is
focused private runtime evidence, not by itself source-connected or package acceptance.

Focused source qualification uses the same .mdsl source with none versus
strict-fused-pair, requires the optimized executable's actual undefined reference
to `gemmStrictFusedPair` and its absence in the baseline, then executes independent
two-stage strict-f32 oracles, original failure/source/effect prefixes and whole FP
checks. The 22 source cases include panel tails, reduction/FMA/intermediate-rounding
discriminators, first-K0 Inf/NaN, N0 +0, M0/P0, identical input views, both source
contraction failures, full-C overflow, byte overflow, final extent overflow and
huge legal empty termination. Single-/multi-source paths have now been physically
checked on the exact fresh `de5d09f4c59ec035482121086f0422b2476a76ff` build: each
mode ran 659 checks and 22 cases with zero failures; both actual call-reference
discriminators passed. The five-test set also passed the independent runtime's
204 checks, the plan's 283 checks and the native-only macro-off control.
Installed driver and relocated source/build-inaccessible package paths remain
pending integration gates. Plan tests
cover dead/live/helper dimensions, branch holes, helper ledgers and multiple
nonoverlapping windows; diagnostic/manual proposal acceptance is not authority.

Independent exact-commit review accepted `de5d09f` for composed qualification
without blocking findings; it inspected saved execution evidence without reruns.
Integration/full local, clean build and exact-head hosted evidence are pending.
The integration owner records actual results and
canonical checkpoint here after qualification; no merge or production-fusion
acceptance is implied by this implementation/design document.
