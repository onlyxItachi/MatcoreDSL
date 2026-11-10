# P3: full-N/tail peeling, compile-only falsification

Canonical starting checkpoint: `243f6772ebf8cc7600d05776a2adbb8d8a0515d0`.
This follows the [P2 upstream trace](strict-output-tile-upstream-trace-v1.md),
not a retiming or reinterpretation of P1's negative/inconclusive campaign.

## Observed mechanism

Pinned MLIR/LLVM/Clang 21.1.8, frozen P1 5x3 bufferized/Transform inputs.
Upstream Transform replay and ordinary lowering reproduced the original LLVM
and object bytes exactly after restoring only the already-established target
triple and aligned-C argument 15 `noalias`. Raw and derived files remain separate;
the inspection binder is not an importer, body verifier or execution authority.

The sole prototype change is upstream `transform.loop.peel` on the outer N tile
handle after generalization/interchange, followed by canonicalization. See the
[pinned upstream operation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td).
The full domain is `[0,F)`, `F=3*floor(N/3)`; the separate tail covers `[F,N)`.
Full B/C subviews have constant inner extent 3. M tails, complete increasing K,
unflagged separate f32 multiply/add and the once-only full +0 fill remain.

Unchanged O2 now produces three scalar C accumulator PHIs and exit stores at
LICM. Later independent-output SLP packs columns 0/1 into two lanes and retains
column 2 scalar. Explicit k-then-k+1 chains and an odd-K epilogue remain, with no
cross-output reduction, FMA or new numerical permission. No per-K N-tail branch
or C memory recurrence remains in that full-tile body.

This is **compile-only mechanism evidence**, not measured performance. Inlining
also changes: the leaf grows from 778 to 1717 bytes; the wrapper changes from a
763-byte inlined body to a 78-byte thunk. These are genuine confounds. Nothing was
linked, loaded or executed; no product source/default/toolchain was changed.

## Independent review and adoption boundary

Independent review rechecked all 68 command-log hashes, identities, full/tail
coverage, zero/empty/K=0 structural behavior and final scalar/vector lane mapping.
All 17 compiler processes succeeded. Two retained nonzero inspection comparisons
record only a scheduled printer newline difference; complete LLVM/object controls
are byte-identical. No unresolved high/medium report finding was reported.

The current product verifier accepts only the unpeeled graph: seven scalar loops
and one arithmetic copy. The prototype has eleven loops and two disjoint copies.
Simply loosening counts is not a proof. An additive private realization needs an
independent branch-aware coverage/dataflow envelope plus parameter-bound replay,
valid-IR mutants, raw-leaf zero/null/alias/canary and numerical/sanitizer execution.
Unit N tiles need an explicitly proved no-tail case. General tile parameters need
their own proof; one 5x3 machine-code witness does not establish their code shape.

The reviewer found a concrete test gap: existing N=129 order/FMA controls exercise
only the full branch for tile N=3. New controls must put those distinctions in
full lanes and tail widths 1 and 2, including mixed N=4,5,7,8 and odd/even K.
Source-time consumption remains a separate exact-classifier/ownership boundary;
no source candidate follows from this prototype.

## Retained identities and limits

Raw: `/var/tmp/mdslc-p3-peel-kiv52W/`. Report SHA-256:
`eae7012b9808a15a91e85a34e0034d947c34ba2e401bc9a10f73cdaa5c3dd97b`;
commands: `e5a199912ad6c4479f1a30e6a78b76811a4736df705fca9b78021f87eef41ea7`.
Compiler-process budget use: 0.745654 seconds, not a performance sample;
819347 raw bytes. Local retained archive under canonical ignored
`benchmark_reports/strict-tile-peel-prototype-2026-11-10-v1/trace.tar.gz`:
`50eb801164d12af7e5c09270b8b24a99fbf406b4b4c367d84c468b634ca4e869`.

The new README explicitly corrects P2's broad environment wording: P2 stripped
only `LD_PRELOAD`/`LD_LIBRARY_PATH`, not every loader/compiler variable. Neither
prototype claims comprehensive environment attestation. Frozen P2 bytes were not
rewritten. This caveat does not change P1's separate stronger environment checks.
