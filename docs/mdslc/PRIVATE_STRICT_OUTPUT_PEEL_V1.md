# Private strict output pattern: full-N and tail v1

Starting canonical checkpoint: `243f6772ebf8cc7600d05776a2adbb8d8a0515d0`.
Engineering scope: [Issue #102](https://github.com/onlyxItachi/MatcoreDSL/issues/102).
Status: focused private qualification passed; full regression/hosted integration
remain pending. Not a merge or speed claim.

This records the original private boundary. The separately requested source
connection has its own [contract and qualification](SOURCE_TIME_OUTPUT_PEEL_V1.md);
it does not promote primitive manifests into source authority. Historical
qualification below must not be read as source-connection evidence.

## Why this boundary exists

The authenticated source-time performance experiment retained a loss and an
inconclusive overall matrix. Its stock and 5x3 LLVM artifacts already carry the
justified fresh-output `noalias` fact. Adding another alias assertion would not
address the observed conditional accumulator accesses.

The reviewed [compile-only research checkpoint](https://github.com/onlyxItachi/MatcoreDSL/blob/4f3cd4c/docs/mdslc/agent-reports/strict-output-tile-peeling-prototype-v1.md)
is not an executable input. Its durable reports are on
[`research/mdslc-strict-tile-codegen-v1`](https://github.com/onlyxItachi/MatcoreDSL/tree/research/mdslc-strict-tile-codegen-v1/docs/mdslc/agent-reports):
P2 explains why ordinary O2 did not promote all accumulator accesses; P3 shows
that upstream full-N/tail peeling permits three independent register accumulators
for 5x3. Code size and inlining also change, so neither establishes a speedup.

## Additive HOW, not a new mathematical operation

The compiler-private issuer accepts the separately explicit
`--output-tiles-peel-n=M,N`, with each integer in `[1,64]`. It derives only the
same canonical built-in strict GEMM; no submitted MLIR, object, shader, callback
or cached instance is accepted. Existing `--output-tiles`, scalar/row-contiguous
and all source-driver defaults retain their original derivations.

Let `T` be the N tile and `F=T*floor(N/T)` for nonnegative logical N:

- Full N origins cover `[0,F)` with step T and exactly T local columns.
- A separate tail covers `[F,N)` with local width `N-origin`, less than T.
- Both have the original bounded M tile and complete increasing K, step one.
- Every output keeps positive-zero initialization and separate f32 multiply/add.
  The full destination is filled once before either compute domain, including K=0.
- A/B operand order, actual C accumulator/destination indexing and the returned
  original destination remain unchanged. No packing, padding, allocation, copy,
  publication, synchronization or new numerical/alias permission is introduced.

Construction uses pinned MLIR 21.1.8 `transform.loop.peel`, not a private tiler or
accumulator optimizer. Tile N=1 statically selects the original no-tail graph.
It must not call peeling and suppress a failure: the pinned upstream operation
can refuse step one. Upstream owns later canonical lowering, independent-output
SLP, instruction selection and register allocation.

The independent scalar envelope checks each full/tail domain, split identity,
bounds, indices, arithmetic and destination. Exact parameter-bound upstream replay
is an additional complete-graph defense, not a replacement for that envelope.
Semantic mutants must fail in the envelope itself; a replay-only mismatch is
tested separately. Unit N has one arithmetic copy; other tiles have exactly two
disjoint copies. LLVM footprint checks remain a post-lowering guard, not a proof
that arbitrary two-copy code is correct.

## Authority and platform boundaries

Only the private baseline Linux x64/AArch64 issuer route is eligible for this
qualification. Nonbaseline ISA requests are rejected. Target emission alone is
not physical validation. The original private milestone added no source-time
flag or source-consumption classifier. No build-default schedule, candidate
registry or runtime adapter is switched to this realization. Existing source
requests still select the old unpeeled pattern; a separate source request must
meet the separately authenticated connection contract above.

Raw private callers must provide valid dense f32 descriptors, live/race-free
inputs, a disjoint private output and compatible FP controls. A/B may overlap.
They retain the existing signed next-IV bounds:
`M <= INT64_MAX-(tile_m-1)`, `N <= INT64_MAX-(tile_n-1)`, checked extents and
addressable byte counts. The source adapter's byte guards/empty bypass alone do not
authorize source issuance. Raw K=0 still fills C with +0 with null A/B data;
empty output performs no data access. No source failure frontier is modified.

## Required qualification

Before merge: valid-IR full/tail gap/overlap, wrong bound/step/index, changed K,
accumulator/destination, numerical and extra-effect mutations; unchanged original
witnesses and old artifacts; malformed/unsupported CLI requests; actual issued
normal/generated-ASan leaf execution with canaries and overlapping inputs;
full/tail rounding/order/FMA/subnormal/Inf/NaN/zero-sign controls; raw empty/K=0;
deliberate sanitizer violations in both full and tail bodies; affected regression,
independent review and relevant hosted checks. Report skipped/inapplicable controls
separately. This contract is not evidence that those gates have already passed.

In particular, tile N=3 needs numerical witnesses at mixed N=4,5,7,8; N=129 alone
never exercises its tail. Cover odd/even K and the full/tail M boundary too.

The initial local focused suite passed **44/44 CTests, zero CTest skips**. The
inspector passed 627 checks, rejected 56 valid-MLIR semantic adversaries before
replay, and reported zero unsupported upstream mutants. CLI checks refused 22
invalid requests and issued eight parameter/target combinations. Fourteen
normal/generated-ASan programs covered seven tile pairs: 27020 cumulative cases
and 50958208 checks, with zero mismatches; reused cases are included, not added
twice. Fourteen deliberate corruption controls and fourteen generated full/tail
ASan violations were detected. Witnesses requiring more K terms than available
are explicitly inapplicable, not counted as passes (1380 across the positive runs).

The compiler-issued 5x3 LLVM is byte-identical to the reviewed P3 specimen,
SHA-256 `7864ea0110a3567ad29c0f27d8a64369f05fb4f440625d457a4712b8021d3496`.
This joins real private issuance to the observed mechanism; it does not connect
a source program or measure performance. See the [implementation record](agent-reports/strict-output-peel-n-implementation-v1.md)
and [independent fixtures/review](agent-reports/output-peel-adversarial-tests-v1.md).

## Exactly one subsequent boundary

After private qualification, connect an explicitly requested peeled realization
to the existing authenticated source-time issuer with a distinct instance identity
and exact consumption classifier. Preserve source closure, artifact ownership,
effects and failure order; do not silently replace v1 requests. Only then does a
new preregistered source-call comparison become meaningful. No tuning, threshold,
performance win, BLAS parity or GPU schedule follows from private correctness.
