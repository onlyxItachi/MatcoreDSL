# Publication-to-read forwarding: independent adversarial work

## Scope and ownership

Baseline: `43ce3cb2557aad4dc3b810ea715c00531c76628f`.
Worktree: `mdslc-work/region-optimization-v1/adversarial`, branch
`test/mdslc-forwarding-adversarial-v1`.

This reviewer independently authored
`compiler/tests/closed_host/forwarded_read_independent_test.cpp` from the
original resource/runtime contract and the agreed `readForwarded` declaration,
before reading the implementation diff. The forwarding implementation, its
plan/emitter tests and CMake integration belong to a different agent. This is
test/code authorship separation, not a claim of formal or universal correctness.

The initial review was read-only. Canonical main and both implementation/research
worktrees were clean at the baseline. No full build, GitHub mutation, source API
extension, candidate-policy change or optimization implementation was performed
by this reviewer.

## Required authority and ordering boundaries

- Preserve the exact original sealed Program and replay-paired untransformed
  witness. An immutable derived plan is separate, bounded compiler authority;
  a serialized module or mutable proposal must not issue it.
- Recompute and compare the full canonical forwarding record list, identities,
  mode and revision when consuming a plan. Reject moved-from or mismatched
  evidence, including another selected region from the same source snapshot.
- Every intervening publication invalidates all previous resource facts:
  distinct descriptors and partially overlapping subranges remain MAY-alias.
  Clearing state at shape-branch entry and join is conservative and sufficient
  for the initial implementation.
- Forwarded reads retain the original requested extent, requested/descriptor
  shape and descriptor access/capacity/pointer/span checks, in that order and at
  the original frontier. Only the snapshot allocation/copy is removed.
- Frontier enumeration includes shape-if records and both statically present
  branch arms, even though only one arm executes and shape-if itself has no
  runtime adapter call. Unequal nested branch lengths must be tested.
- Retained values remain immutable when later external publications overlap,
  and observation remains an owning snapshot rather than a reuse hint.

## Independently authored runtime falsifiers

The new test covers:

- Original-versus-forwarded required-check status/error precedence, including
  invalid zero-extent requests, requested-shape mismatch, invalid access,
  short capacity, null/misaligned storage and address-span overflow.
- Unchanged existing output Value on failure; retained-value validity/shape;
  no owned allocation before invalid-input rejection.
- First successful publication and observation retained across late forwarded
  shape failure; sticky failure suppresses subsequent reads and effects.
- Equal successful frontier traces with actual retained storage; deterministic
  removal of the late snapshot-allocation failure opportunity while preserving
  every earlier injected allocation failure and completed effect prefix.
- Bit-preserving publication/read retention, partial overlap followed by an
  ordinary late read, and value lifetime beyond Session/producer destruction.
- Null empty storage, retained-input/result self-assignment, noncommuting lhs
  and rhs consumers, and same-handle `C*C` arithmetic.
- Zero/nonincreasing frontiers, a retired session, synchronous reentry through
  the test-only candidate seam, and complete caller FP-state preservation on
  successful and failing normal return.

Direct malformed-input tests exercise the private adapter defensively. They do
not claim runtime authentication of a forged or stale compiler derivation. In
particular, the partial-overlap test deliberately uses an ordinary read after
the aliasing publication; static plan tests must establish invalidation.

## Validation at initial test handoff

Test authorship is complete; execution is **not yet validated**. The integration
owner explicitly reserved build capacity and instructed this agent not to build
initially. The forwarding implementation/header is on its separate branch.
The implementation owner will register the test against the test-only runtime
variant. A clean compile and focused execution are required after composition;
neither uncompiled source nor review is passing-test evidence.

## Separate bounded fusion review

The research row-panel schedule is not production authority. Initially, default
One-Shot bufferization was observed by the research owner to introduce unchecked
`malloc`/`free` per panel. That was a blocking integration defect under the
existing no-allocation generated-leaf contract. The later caller-workspace
research refinement and remaining trust boundary are recorded below.

For exactly a pure strict CPU lhs-chain interval, with all external reads before
the pair and no surviving producer use, it is plausible to retire the first
GEMM's required checks before deferred physical pair computation. This requires
the original full producer extent/product, numerical/candidate legality and
valid FP enter/restore at its original frontier, not just panel-shape checks.
Eliminating full-producer allocation is permitted by resource-contract item 8.
The second checks and private allocations must retain their own frontier.

This reasoning is conditional on a verified call-free, allocation-free strict
CPU leaf whose first arithmetic segment cannot report a recoverable candidate
failure or alter FP controls under trusted execution. The existing strict issuer
enforces no leaf calls and no fast math; the new pair must establish its own
equivalent boundary. It does not extend to provider, GPU or test callbacks.
One combined post-call failure cannot simply be attributed to the second GEMM
without this argument. No all-program theorem, broad fusion, residency or
performance/parity claim follows.

### Split guard/retirement argument and counterexamples

The existing runtime enters `ClosedFpEnvironmentV1` only after GEMM value,
profile, shape and candidate checks and output allocation. It then checks
`valid()`, runs the candidate, checks `controlsUnchanged()`, restores the complete
environment, and issues the result only on success. The Linux x64 scope saves
libc state, MXCSR and x87 control/status; the ARM64 scope saves full FPCR/FPSR.
Restoration failure terminates rather than returning a contract-breaking status.
The post-call control check deliberately ignores arithmetic sticky flags, which
are internal and restored before normal return.

The current strict CPU issuer rejects leaf calls and fast-math flags. Under that
specific trusted, call-free arithmetic realization, ordinary multiply/add does
not modify FP controls or return a recoverable candidate error. This supplies
the missing bounded argument for deferring first physical arithmetic across the
second guard, not a general permission to defer arbitrary candidates. The fused
issuer must independently verify the corresponding no-call/no-allocation,
numerical-order and control-preservation boundary after its own derivation.

An admissible split therefore has these obligations:

1. At the first original GEMM frontier, validate both original values, its
   original numerical profile, contraction relation, selected-candidate legality
   and the full logical producer extent/product. Perform a valid original FP
   enter/control-check/restore before retiring that frontier. Do not report that
   the full producer was physically materialized when it was not.
2. Only then validate the second GEMM at its original frontier. A second shape,
   candidate, FP or private-allocation failure leaves the first required checks
   completed and retains all effects preceding the pure interval.
3. Allocate the bounded panel and final output through checked adapter-owned
   resources. The removed full-producer allocation may remove an exhaustion
   opportunity. It cannot remove required extent checks, introduce hidden
   unchecked allocation, or retire a second failure before an earlier effect.
4. Execute exactly the issued strict pair and retain the producer's f32 rounding
   boundary and each contraction's separate increasing-order multiply/add.
   Restore caller FP controls and status on all defined normal returns.

A concrete required extent falsifier is `A[INT64_MAX,0]`, `B[0,2]`, `D[2,0]`.
Each input read is empty and the final output has zero columns, but the first
logical `C[INT64_MAX,2]` is not representable. The first GEMM must fail before the
empty consumer can shortcut computation; checking only panel/final extents
would incorrectly succeed.

A candidate that can change controls, call a provider, invoke an injected
callback, or report failure after partial computation is a counterexample to
the simple split. One combined post-call failure would lose whether the original
first or second candidate failed. Such candidates need a stage checkpoint or
remain ineligible. No independent fusion tests were added before a production
API and derivation exist.

## Provisional staged implementation review

After freezing the independent test authorship, this reviewer inspected the
forwarding agent's uncommitted plan/runtime/emitter and single-/multi-source
driver changes. No blocking correctness defect was found in those inspected
versions: plan issuance is closed and immutable, consumption replays the exact
original witness and recomputes all records, every publication replaces the
single MAY-alias fact, branch entry/join clears it, and emitted forwarded reads
repeat the original runtime guard ordering. Optimization remains independent of
candidate choice and defaults to `None`. The added exact export demangles to
the agreed six-argument private method without changing Session layout.

One test-fixture defect was reported: the host-less inspection admission control
used a helper that adds `#include`, while the hermetic inspection route rejects
all preprocessing. The test must first establish successful preprocessing-free
inspection admission before checking that it cannot issue executable authority.

This is not final exact-commit acceptance or executed validation. In particular,
the existing two-GEMM example carries `c` directly rather than reading the
published resource, so running it with the optimization flag alone cannot prove
that a forwarding record was used. Qualification needs a compiled source case
with a genuine publication-to-read edge and artifact/execution evidence across
the claimed candidate routes.

## Exact-commit forwarding review

Verdict: **ACCEPT the bounded forwarding implementation for integration**.
No blocking correctness defect remains in the reviewed changes. This is not
complete qualification, merge authorization, a formal all-program proof or
acceptance of future fusion.

Reviewed exact commits:

- Independent test cherry-pick:
  `cc078cf769ded78d739ed1797e1dc119fdb4c9c6` (authorship origin `f381ef6`).
- Implementation: `14367dc825333e8465299149a7250dc8ac889688`.
- Plan, connected driver and installed tests:
  `6548a2758442757be5b9f74b2b1b237fb9ea9581`.
- Compiler subtree at the reviewed head:
  `c3e25487ee6ce1d80a7afbb1d2108ea3f11eab7c`.
- Independently reviewed CI registration change:
  `2c66487a08000efc6bea20d1c76e8940160af616` in the qualification lane.

The working compiler files matched the reviewed test head exactly. The two
implementation/test commit diffs passed `git diff --check`. The reviewer did
not compile or run tests during this review; the implementation owner separately
reported its direct ASan/UBSan runtime execution of 67 independent checks with
zero failures. Source-connected execution, composed/full suites and exact-head
hosted results remain the integration owner's distinct qualification gates.

### Final authority delta

`ClosedHostDerivedPlan.cpp` lines 29-51 rebuild the original witness and replay
exact source pairing. Lines 62-87 retain global MAY-alias invalidation and the
conservative branch barriers. Lines 171-195 reauthenticate and compare complete
identities and canonical records at consumption.

Removing a redundant third witness replay from `ClosedHostEmitter` does not
weaken the boundary: its line 121 calls `verifyClosedHostPlan` before reading the
Program or consuming `semanticIdentity()` at line 128. The new getter returns
only the immutable original digest and rejects moved-from access; it cannot
construct or mutate a plan. Runtime lines 514-533 retain the original guard
ordering before nonallocating ownership retention. No source/witness verifier
was relaxed and no Session layout changed.

### Connected test and CI review

The new source specimen actually reads C after publishing C; it is not the old
direct-value-carry example. Its partial-overlap branch publishes through D before
reading C, which falsifies stale forwarding across a MAY-alias write. The other
branch checks forwarding, a separate alias read's short capacity, requested
shape/extent failures, retained observation ownership, exact source locations
and branch-hole frontiers.

The test inspects the actual executable with `nm --undefined-only --demangle`:
the optimized executable must reference `readForwarded`, and the `none` binary
must not. This rules out the test accidentally compiling two baseline
realizations. The same source is then executed against independent exact-small-
integer math and failure-prefix oracles. The two-TU program test verifies option
propagation through the separate program compiler; the installed package repeats
the connected discriminator using its installed driver and generated candidate.
The initially defective host-less fixture now first admits a preprocessing-free
inspection source and then rejects executable-plan issuance.

Optional candidate tests deliberately permit six explicit first-GEMM
`candidate_unavailable` refusals. That is fail-closed refusal evidence, not
forwarding/candidate execution. Native-strict, generated-strict, automatic,
multi-source and installed qualification require six actual executions. Physical
ISA/provider/GPU claims must use the execution counts or force
`REQUIRE_EXECUTION=ON`, not infer execution from a passing CTest alone.

The CI count changes are consistent with the registered x64 test graph:

- OpenBLAS OFF driver scope: previous 41 plus eight candidate policies and one
  program test equals 50.
- OpenBLAS ON driver scope: previous 42 plus nine policies and one program test
  equals 52.
- ASan/UBSan OFF-provider scope: previous 175 plus nine driver tests and two
  proof/runtime tests equals 186. The installed forwarding check extends an
  existing CTest and therefore adds no separate registry entry.

The new expressions include both proof/runtime tests and all driver policy
tests. Native qualification additionally requires the four mandatory connected
execution tests to exist. These are static registration/count checks, not an
assertion that the new hosted jobs have executed.

## Refined fusion research trust boundary

Research commit `198ee59` subsequently mapped the certified panel allocation to
caller-owned workspace. The reviewer inspected its saved artifacts without
building or executing them: `workspace.ll` SHA-256
`56f108e0bb4ecefb74c854b9d02968d5281a0467d246fa62f69549082275d0e0`
has only the `llvm.smin.i64` intrinsic and C-interface-to-leaf call;
`workspace.o` SHA-256
`02067bbea0241541eef823de874eab20dd3654ebd9127185db4c15a360461eac`
has exactly the undefined symbol `memset`. The research owner's 6,786 executed
checks are its evidence, not an independently rerun result here.

Thus the earlier call-free condition is a sufficient initial model, not an
accurate blanket machine-code description or a necessary prohibition on known
memory helpers. LLVM's lowered zero fill invokes libc. The bounded deferred-
arithmetic argument needs an explicitly trusted conforming `memset` contract:
writes stay within checked private storage, no recoverable failure or arbitrary
host effect occurs, and FP controls are preserved. Arbitrary interposition is
outside that trust boundary. A production issuer must verify/allowlist the
actual helper boundary and retain adapter-owned checked workspace; pre-
optimization LLVM alone cannot establish those machine/library facts.

The caller-workspace experiment resolves the previously observed research
allocation defect, not source authority, runtime source-frontier retirement,
production artifact ownership or sanitizer qualification. No production fused
candidate is accepted by this review.

## Integration harness failures and bounded corrections

Integration execution subsequently exposed test-harness defects missed by the
earlier static test review. The integration owner reported that the coherent
focused runtime test passed 67 checks and the plan test passed 130 checks; these
are owner-executed results, not additional independent runs. The first connected
driver attempts did not execute their source oracles and must not be counted as
forwarding execution evidence.

The single-source harness used `if(POLICY MATCHES ...)`. CMake interprets the
literal `POLICY` as its reserved policy-query form, so the test failed before
compilation. The integration owner's test-only correction quotes the variable's
value, `if("${POLICY}" MATCHES ...)`. Independent follow-up inspection also
found that both CMake runners appended complete oracle summaries to a list even
though the summaries contain semicolons. Consequently, `list(GET outputs 0)`
and `list(GET outputs 1)` select segments of the first summary, not the complete
baseline and optimized summaries. The inspected scalar `baseline`/`optimized`
assignments preserve the complete text and are the bounded correction. Neither
issue requires a compiler/runtime change. These root-owned changes were still
uncommitted at this review snapshot; the exact integrated head and rerun results
remain the integration owner's qualification evidence.

### Existing strict-FP multi-source admission boundary

The original multi-source fixture failed even with optimization `none`:
`CALLSITE: region call ABI or effect promises differ from compiler witness`.
The independent reviewer inspected the implementation owner's already-emitted
Clang 21 IR without using another compiler slot:

- `builds/forwarding-callsite.87IiCa/strict-host.ll`, SHA-256
  `43e159eddafd338fcc70e529c1e9bdbe31c1ae182647d580ae2bb8eb7a604df8`:
  protected region call at line 622 has function attributes
  `nounwind strictfp` (attribute set 14, line 1797).
- `builds/forwarding-callsite.87IiCa/witness.ll`, SHA-256
  `8a66e7bb068c9bf5abd74f0a3a28f6ad7e6e06705643acd30f158c625fbdec1c`:
  the corresponding canonical pragma-free call at line 43 has `nounwind`
  (attribute set 3, line 56). Its return, argument, `sret` and `byval` attributes
  match the protected host call.

The extra `strictfp` comes from the fixture's `FENV_ACCESS ON` call context.
`sameAuthenticatedHostCallInterface` deliberately compares complete function
attributes as well as the rest of the ABI, so this is the existing exact-call
admission bound, not an optimization-dependent failure. Ignoring `strictfp` in
the comparator would change that production authority boundary and is not an
acceptable test repair.

Test-only commit `929cb18dce0944cb637f9cd24dc2aa1ec0328a7a` is accepted for
integration. It keeps the original direct-strict-FP fixture as a negative in
both optimization modes, requires the exact `CALLSITE` diagnostic and requires
that no executable was published. For the positive program fixture it inserts
an ordinary `noexcept` forwarding wrapper immediately after the unchanged
region body and before `FENV_ACCESS ON`. The wrapper performs no arithmetic and
makes the protected direct call in the existing admissible context. `run()`
still has the full strict-FP before/after environment oracle; single-source and
independent runtime FP tests are unchanged. Inserting the wrapper after the
region also preserves the original region line/column failure assertions.

The commit passed static diff inspection and `git diff --check`; this reviewer
did not compile or execute it. Acceptance of this bounded test correction does
not replace the required connected, installed and hosted qualification reruns.

## Contract-first independent strict-pair runtime falsifiers

The integration owner next authorized a separate independent test,
`compiler/tests/closed_host/strict_fused_pair_independent_test.cpp`. It was
authored against the agreed private API before inspecting any fused runtime
implementation. Existing unfused runtime semantics, the public resource
contract and the issuer's arithmetic stages were available to the reviewer;
this is independent runtime test authorship, not independence from all design
discussion or the issuer review.

The agreed API is `gemmStrictFusedPair(f1, f2, A, B, D, first, second, E)`.
The frontiers are consecutive without wrap; both profiles are `strict_f32` and
the request is explicitly `generated_strict`. An unknown candidate is invalid;
the otherwise legal automatic/native requests and a configured test callback
cannot silently substitute for the trusted pair. Allocation-only fault
injection remains supported. The test requires a separate matched test runtime
with genuine single-GEMM and fused issued objects, not a callback, native
fallback or changes to the existing native-only test runtime's availability.

The new test covers:

- Original f1/f2 guard precedence and sticky status, including the full logical
  C signed-element and byte-span extents. `A[INT64_MAX,0]`, `B[0,2]`, `D[2,0]`
  must fail f1 even though all reads and final E are empty. Conversely,
  `A[INT64_MAX,0]`, `B[0,0]`, `D[0,2]` must retire f1 and fail E extent at f2.
- Legal enormous empty outputs, including `M=INT64_MAX,K=N=P=0`, must complete
  without entering the row loop. A finite CTest timeout is required to catch
  accidental traversal. A nonempty E with `N=0` contains positive zero; a
  nonempty E with `K=0,N>0` still executes the consumer, including `0*Inf` and
  `0*NaN`, and honestly reports physical invocation.
- An independent two-stage volatile-f32 oracle, noncommuting matrices, both
  reduction orders, FMA discrimination, intermediate f32 rounding and forbidden
  cross-GEMM reassociation. Full and tail row panels, identical input storage,
  output assignment aliasing each input handle, preceding external alias
  publication and later result publication/lifetime are checked.
- Preserved earlier publication/observation prefixes, unchanged caller result
  handles on failure, retired sessions, nonadjacent/wrapped frontiers, callback
  refusal and reentry through an existing outer test candidate.
- Every allocation fault in the actual successful pair's measured allocation
  range. No removed full-C allocation may replace a required f2 failure; no
  extra allocation may occur after sticky failure. The tests do not prescribe
  a realization-independent allocation count.
- Complete raw caller FP controls/status across success, each guard stage and
  every actual pair allocation-failure point. Caller downward rounding, flush
  modes and sticky flags discriminate normalization to nearest-even/gradual
  underflow and exact restoration.

The source-neutral shape-use question was also resolved by code inspection.
`ClosedRegionAdmission.cpp` represents canonical `rows`/`cols` calls as pure
Shape bindings and returns before appending an operation. Discarded bindings
have no baseline runtime action, ordered failure or observation. Only retained
`Dimension::ValueRows`/`ValueColumns` references become semantic dimension
operations in read dimensions or shape-control operands. A recursive whole-
Program liveness/use check therefore preserves the bounded no-live-C-shape-use
gate; an additional Sema query ledger is unnecessary frontend coupling. Dead
pure queries may remain eligible, while live references anywhere in either
control arm must block this first fused segment.

This test source has not been compiled or executed by its author. Registration,
actual generated-leaf execution and sanitizer instrumentation are distinct
pending gates. No production fused source path is accepted merely because these
falsifiers have been written.

## Exact-commit isolated fused issuer review

Verdict: **ACCEPT the isolated compiler-owned strict CPU pair issuer** at
`8ce0bcd349cf2339162ce423c82377d3c827d642`. No blocking correctness defect
remains in the reviewed bounded component. This is not acceptance of the
pending source plan/runtime implementation, merge authorization, a formal
all-program theorem, performance evidence or all-target fusion.

The working tree was clean and matched that exact commit. The reviewer read
the complete issuer, its private contract, CMake integration, CLI and tests;
`git diff --check` passed. Exact producer/consumer graphs are built inside the
closed issuer. Actual scheduling and bufferization use upstream Transform and
One-Shot, with exactly one checked row-panel allocation request mapped to
caller workspace. The textual stage references compare the resulting graph;
they are not executable loop payloads or an external import path. Both strict
reductions retain their ordered separate f32 operations and the intermediate
rounding boundary. Inputs remain MAY-alias; E and scratch require private
disjoint storage.

The earlier preoptimization LLVM provenance/count checks alone did not cover
every integer IV, branch, address expression or function attribute. The final
commit supplements them with pinned complete normal/ASan LLVM graph hashes,
ignoring only the diagnostic ModuleID comment, plus concrete IV/GEP/unsafe-FP-
attribute corruptions. This is an exact coherent-21.1.8 drift gate. Neither a
successful bool verifier nor a manifest issues source/runtime authority; the
only executable artifact factory accepts its MLIR context and sanitizer mode,
not supplied IR or a caller Transform program.

The initially overbroad empty-output leaf precondition was corrected. The
unchecked row loop must not receive empty E; the future adapter must still
retire both original required guard frontiers before taking that shortcut.
For nonempty E, its validated byte extent bounds M and makes the row increment
by four representable. The full logical C extent remains mandatory at f1 even
when E is empty. K=0 is not a final-output-zero condition; N>0/nonempty E still
executes the consumer. The independent runtime test covers these distinctions.

The optimized-object gate runs as part of object generation, not only CTest.
It requires both exact strong ABI exports, permits only the normal `memset`
import or the explicit ASan instrumentation set, and checks separate scalar
arithmetic without FMA. The trusted memset contract explicitly requires bounded
private writes, no recoverable failure/arbitrary host effects, and preserved FP
controls. Arbitrary interposition is outside that boundary.

Without compiling or executing, the reviewer independently checked the saved
normal/ASan LLVM hashes `bc1cdf4c...b237c` / `02432a53...8ea52` and object hashes
`c1644975...243a6` / `10892a50...7d095` against the owner's full hashes in
`strict-fused-pair-issuer-v1.md`. `llvm-nm` confirms the stated actual imports.
Disassembly shows fixed stack-frame adjustments for both leaf/wrapper modes;
the preoptimization loop-local descriptor metadata allocas do not survive as
an accumulating per-panel stack allocation in these checked objects. The
owner's saved CTest log records 9/9 passed: 61 candidate checks, 13,617 checks
per normal/ASan execution mode, and actual generated A/B/D ASan read controls.
Those runs were not independently repeated here. Of 3,360 exhaustive geometries,
840 empty-E cases intentionally do not invoke the leaf; the owner's recorded
2,561 total invocations per mode include the additional nonempty specimens.

The separate CI delta `31401bf6002e10d09213f2e613ab2f29734def4f` is also
accepted by static inspection. The nine registered tests are one candidate,
one CLI, two execution, two object and three ASan input-read controls. Release
requires all nine before its full suite. The ASan expression adds exactly this
new namespace; the existing `generated_cpu\.` expression does not match
`generated_cpu_fused_pair.`. Therefore the selected count changes from 186 to
195. Issuer files in that integrated commit match the reviewed issuer bytes.
Hosted, composed/full, installed and later source/runtime qualification remain
the integration owner's distinct gates.

## Exact-commit source-connected strict pair review

Verdict: **ACCEPT for composed qualification** the bounded source/runtime
implementation at `de5d09f4c59ec035482121086f0422b2476a76ff`. No blocking
correctness or authority defect was found in this exact clean checkpoint. This
acceptance now includes inspection of the owner's actual connected single- and
multi-source executions, not just emitted text or an inspection verifier. Full
integration, installed/source-inaccessible execution and exact-head hosted
qualification remain pending; this is not yet a source-fusion merge
recommendation, a universal proof or performance evidence.

The reviewer froze the independent runtime test at `d7b0571` before reading the
new runtime implementation, then reviewed that implementation and its final
wiring separately. Test authorship was independent of the implementation, but
the test author and this reviewer are the same agent. No compile, test execution,
GitHub mutation or implementation-worktree write was performed by the reviewer.

The complete retained semantic use count includes Read dimensions, both
ShapeIf operands/arms and every GEMM/publication use. A pair requires the
adjacent strict lhs chain and exactly one intermediate use, with all three
external values defined by dominating immutable Reads. Branch-local read maps
are copied rather than leaked into siblings or continuation. Multiple windows
are disjoint; the exact original operation/frontier enumeration, including
untaken-arm holes and helper call stacks, is retained by both derivation and
emission. The plan still owns original authenticated evidence, recomputes its
complete ordered records/identities when consumed and gives no issuance
authority to caller proposal vectors. Neither the original Program nor its
whole-region witness is transformed.

Runtime review confirms the full logical C signed/product/byte extent at f1,
before D/second-profile inspection. The original first FP enter/control check
and complete restoration occur before retiring f1; separate ActiveCall scopes
permit the original second begin. At f2 the original input/profile/contraction/
candidate precedence precedes E and bounded workspace allocation. Output
assignment is last, so every failure preserves the previous handle even when
it aliases A, B or D. Immutable snapshots survive external MAY-alias writes.
The huge legal empty E route retires both logical guard frontiers without a
row loop; N=0 is positive zero, but K=0 with N>0/nonempty E still invokes the
consumer and can produce NaN from zero times Inf/NaN.

Removing full-C allocation is the explicit resource-contract item 8 permission,
not permission to remove C's required checks. Deferring its physical arithmetic
uses the pinned no-recoverable-error, control-preserving strict leaf and the
explicit bounded-private-write conforming memset obligation. No arbitrary
provider, callback, interposer or two-stage candidate-failure attribution is
covered by this argument. Callback injection is rejected before invocation;
allocation-only controls remain. f1 honestly reports no intermediate Value or
invocation, and f2 reports only the realization actually performed. The final
runtime/header delta from `5f34a25` consists only of these trusted-helper comments.

The `8fccd8a` wiring places the genuine issued normal object and its generation
dependency in the production registry and a separate matched testing registry;
it does not change the old native-only test registry. That native-only control
now also rejects the pair as unavailable without output mutation and preserves
sticky failure. Installed source compilation retains exact header/private DSO
ownership. The package additions copy the genuine source fixture before deleting
their owned disposable producer directories, use the relocated installed driver,
require the real executable's pair reference only for the optimized form, and
require all 22 source oracle cases plus a finite huge-empty execution timeout.
Those new installed paths have been reviewed, not executed by this reviewer.

The saved direct ASan+UBSan command in `strict-fused-pair-source-v1.md` compiles
the actual runtime/FP sources and the unchanged independent test, with genuine
single-GEMM and pair ASan objects. The reviewer inspected the executable's
actual method/two leaf definitions and both objects' ASan read/write imports.
The executable SHA-256 is
`1cd3717bf363d263372f51900c7f3dcfd1120a9e3faf227f02a86f1819dd130c`;
the pair object is the previously reviewed
`10892a501e67ebdb741f01cd4c96fe7ee15b95c7b3a4389dfa93bf155147d095`.
The owner's combined stdout/stderr repeat log records 204 checks, zero failures
and no sanitizer diagnostics; its SHA-256 is
`b3dbfd4215733646f28f02fba3e0aad666be1ea66cb9bcecb121030032fc6c16`.
The owner confirms exit zero under a 30-second timeout and pipefail. This is
independent artifact/log inspection of owner execution, not a reviewer rerun.

A fresh separate Release build's generated CTest configuration records clean
source `de5d09f`. Its saved combined focused log has SHA-256
`6793d583c9baae6540c17e9b05d49217f196651ecf64f5ead51bd006975e94d2` and records
five of five passing: 283 plan checks, 204 independent runtime checks, the
native-only production control, and both connected driver forms. Each form
executes the baseline and optimized program with 659 checks, zero failures and
22 cases. The reviewer also inspected both actual executable pairs with nm:
only optimized executables import `gemmStrictFusedPair`, and baseline executables
import the original `gemm`. These tests exercise strict reduction/rounding,
source-failure locations and retained effect prefixes, not only equality of
successful output. The test harness's exact whole-output scalar comparison
does not reintroduce the earlier CMake semicolon-list defect.

Root owns CI integration: preserve the isolated issuer's mandatory nine-test
gate and add the four source registrations explicitly. Existing focused regexes
do not accidentally select those four names. Adding all four to the reviewed
195-test ASan scope requires 199; the two driver names increase the driver
presence count by two. This is a static registration calculation, not hosted
execution evidence. Source docs/AGENTS correctly keep pending qualification,
explicit-only Linux x64 policy, original witness/frontiers and lack of broad
fusion/performance claims distinct.

## Forwarding qualification closure

The final recommendation for forwarding PR #78 is **MERGE exact head
`20c8473fae24d7396d384612064e9b69be0a0c29`**, under the integration owner's
normal exact-head merge procedure. A live clean candidate-tree check and diff
confirm that its compiler library/tool/include production bytes are identical
to the independently reviewed `14367dc8`; subsequent changes are the reviewed
tests, qualification harness corrections, CI and documentation. No host
callsite/effect verifier was weakened to admit the strictfp fixture.

The integration owner reports local 393/393 with zero skips and all exact-head
hosted lanes green, including Debug and the alternate schedule. Those results
close the formerly pending forwarding qualification gates; they were not
rerun or fetched from GitHub by this reviewer. This recommendation neither
authorizes broader optimizations nor implies acceptance of pending fused-source
package/full/hosted qualification.
