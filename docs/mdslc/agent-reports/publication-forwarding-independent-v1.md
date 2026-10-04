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

The research row-panel schedule is not production authority. Default One-Shot
bufferization was observed by the research owner to introduce unchecked
`malloc`/`free` per panel. That is a blocking integration defect under the
existing no-allocation generated-leaf contract; it needs separately verified,
adapter-owned checked workspace before qualification.

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
