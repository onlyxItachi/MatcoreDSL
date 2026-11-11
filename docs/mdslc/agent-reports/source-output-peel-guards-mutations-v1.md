# Source output-peel: guard implication and mutation controls v1

## Scope and evidence status

This lane audited the existing adapter at frozen source
`dfc35f9e13653db16c7f9a825db2bb1b2a82ea22`, then changed only the
noninstalled `compiler/tests/source_pattern/mutation_test.cpp` and this report
in the separate `source-peel` worktree. The additive source request API is
provided by `5028b5a`: `SourcePatternRealizationV1::{OutputTiledMKN,
OutputTiledPeelNMKN}`, with the old realization the trailing-field default.

No runtime, frontend, issuer, build configuration, installed interface or
numerical permission was changed by this lane. No compiler, linker, generated
program, benchmark or physical test was run by this lane for this task. Source
inspection and `git diff --check` are not execution qualification. The integration
owner owns compilation, real mutation execution and final frozen regression.

## Conditional runtime guard implication

The existing adapter does not need a new universal dimension limit merely to
accommodate the private peeled schedule. Its implication is conditional on
**actually entering the generated leaf**, not just admitting a mathematical
empty call. It also presupposes the compiler's independently verified recipe,
correct target/ABI, and request-bound intrinsic DSO ownership; guards cannot
make arbitrary submitted machine code safe.

Relevant frozen code:

- `compiler/lib/runtime/closed_host_v1.cpp:50`: private owning `ValueStorage`;
  `:117`–`:154`: dense rank-two descriptor construction, including duplicate
  allocated/aligned pointers, zero offset and strides `(columns,1)`.
- `:364`–`:394`: extent, capacity, access, nonempty pointer/alignment and address
  interval checks. `:469`–`:499`: checked fresh allocation and eager snapshot.
  `:505`–`:540`: normal and compiler-derived forwarded reads.
- `:542`–`:622`: GEMM guard order, output allocation, FP scope, empty/K-zero
  bypasses, and generated-strict descriptor/call construction. `:709`–`:717`:
  FP-control check/restoration, sticky status, result commit and success frontier.
- `compiler/lib/runtime/closed_host_v1.h:102`–`:123`: caller storage/lifetime,
  all-MAY-alias external views, thread confinement, and allocator trust contract.
- `compiler/lib/platform/closed_fp_environment_v1.cpp:39`–`:96`: native baseline
  strict numerical scope and exact caller environment restoration.

Let `I = INT64_MAX`, and
`B = min(SIZE_MAX,PTRDIFF_MAX)/sizeof(float)`. On the qualified LP64 tuple,
`B = 2^61-1`. Successful owning inputs establish `M*K <= B` and `K*N <= B`;
successful fresh output allocation establishes `M*N <= B`. Each dimension also
fits signed i64. The production generated call is reached only when `M,N,K > 0`,
so each of `M,N,K` is at most `B` there.

For tile parameters `1 <= Tm,Tn <= 64` this proves:

- The final bounded-M outer increment is at most `M+Tm-1 <= B+63 < I`.
  Local-M/local-N induction and their final increments are at most 64.
- With `F = Tn*floor(N/Tn)`, `0 <= F <= N` and
  `0 <= N-F < Tn`. Full-N increments terminate at `F`; the one tail increment
  is at most `F+Tn <= N+Tn-1 <= B+63 < I`. The full and tail index domains are
  disjoint and cover `[0,N)` only because the issued peeled recipe separately
  proves this exact construction.
- Increasing, untiled K has final increment `K <= B`. Dense element indices
  are at most `M*K-1`, `K*N-1`, and `M*N-1`; their products, additions and
  byte-scaled offsets are representable. No triple-work `M*N*K` bound is used
  or implied.

`extent` checks occur before allocation. A nonempty resource additionally needs
declared capacity, an ordinary live host-float object range, natural alignment
and nonwrapping pointer arithmetic. Only the arithmetic checks are mechanically
proved by the adapter; real object lifetime/capacity and concurrent access remain
the explicit caller preconditions. A read snapshots into owned immutable storage,
or the separately proved forwarding path retains an already valid owned value.

The output is a new successful allocation and remains separate from live input
allocations under the existing conforming allocator contract. A and B may be the
same `Value`, may retain the same owned storage, and their original external
views may overlap. A caller may pass the result variable as an input variable:
the old input storage remains live until the final successful result move. The
descriptor's duplicate allocated/aligned pointers are not independent buffers
or a no-alias proof. No extra A/B-disjointness precondition follows.

### Empty and K-zero qualification

An unconditional claim that all accepted dimensions satisfy a tile-increment
margin is false. For example, `A=(I,0)`, `B=(0,0)` has `M=I,N=K=0` and may be a
valid empty call, although `M > I-(Tm-1)` when `Tm>1`. Similar examples put `I`
in N or K with zero-area inputs/output. These calls are safe because the adapter
selects `empty_output` before the generated leaf. Null pointers are permitted
for declared zero-area views. They are not permitted for nonempty leaf inputs.

Positive output with `K=0` still checks the complete output extent and allocates
fresh storage first. Its value initialization yields positive zero and the
`zero_reduction` branch bypasses the leaf. Forced candidate availability and
numeric compatibility are checked even for empty math. The test-only injected
candidate callback precedes these production bypasses, so such a callback is
not a valid witness of the production no-invocation property.

### Numerical, failure and effect order

`Session::begin` (`:444`–`:459`) rejects sticky failure, reentry, completed/taken
state and nonpositive/nonincreasing frontiers before candidate work. GEMM checks
input validity, numeric enum, K equality and candidate legality before fresh
output allocation. No result is issued before the final FP-control check,
restoration and successful status check. Allocation failure preserves the old
result. Earlier publications survive later failures; no new publication is
performed by GEMM. `publish` (`:892`–`:909`) completes all fallible preparation
before its first output copy; owning observation (`:911`–`:931`) prepares its
snapshot before retiring the effect frontier.

The Linux x64 scope establishes RN, gradual underflow and masked traps through
the canonical environment plus checked MXCSR/x87 controls. AArch64 captures the
whole FPCR/FPSR and establishes the accepted zero control/status state. Sticky
arithmetic status during computation is not an illegal control change. The
whole caller environment is restored on normal and failure exits; restoration
failure terminates. Allocation/deallocation outside this scope retain the
existing linked-allocator FP/effect trust boundary. Bounds alone neither prove
ISA legality nor authorize relaxed arithmetic or source entry.

## Noninstalled mutation acceptance changes

The original test main is parameterized by optional `--peel-n`; no option retains
the old `OutputTiledMKN` request. All original source/header phase mutations,
other-tile LLVM/object/DSO replacements, stale-source DSO, semantic-MLIR mutation,
manifest prepopulation, ownership-path race, typed source/entry/policy/profile
checks and positive final publication now also run with the new realization.
The old mode retains those controls and its original publication/execution
fixture. Precreating consumer directories before admission is fixture hardening,
not a product behavior or source/runtime semantic change.

New-mode-only paired controls issue old and peeled realizations from **one
authenticated physical source, the same selected entry and equal tile values**:

- Nonunit `3x5`: both are real compiler-issued donors; LLVM/object differences
  are asserted before opposite-realization replacements are attempted.
- Unit `5x1`: LLVM and object bytes and their recorded hashes must be equal;
  source schema, issuer manifest, request binding, SONAME and DSO must still
  distinguish the requests. Equal executable math does not merge source-instance
  authority. Replacing identical LLVM/object bytes is deliberately not counted
  as a rejecting mutation.
- Both directions independently reject a valid opposite DSO with the original
  source-input binding, and an opposite immutable binding with the original
  actual DSO bytes. A separately authenticated physical-source donor is rejected
  by both the typed input factory and direct compilation boundary.
- Both directions reject the opposite complete compiled result and a copied
  result carrying only the opposite compiler-produced pairing seal. The exact
  source/instance/LLVM pairing diagnostic is required, before native linkage;
  precreated `host.ll`/`result` sentinels remain empty and no executable appears.
  Each original instance/result pairing is rechecked afterward.
- Bidirectional phase controls replace nonunit LLVM/object bytes and both
  nonunit/unit DSO bytes with the valid opposite donor. Manifest prepopulation
  is explicitly tested as the exact no-clobber refusal, not as a parsed-manifest
  authority check. A stale-source DSO donor remains covered separately in each
  selected-mode original suite.

All relevant fixture, bundle, helper, publication and direct-consumer directories
are created before source admission. New typed negatives require the exact
source-seal, candidate-byte or compilation-pairing diagnostic, not merely any
exception containing `source-pattern`. Existing artifact-phase negatives still
require completed hooks, exact changed paths and the expected refusal class.
The synthetic unfinished-hook and path-prefix controls remain in both modes.

Direct parsing and source-invocation validation also reject
`static_cast<SourcePatternRealizationV1>(-1)` with the exact unknown-realization
diagnostic before issuance. This tests the typed selector rather than relying
only on CLI parsing to make unknown values unreachable.

## Smallest additional guard falsifiers

These are source-derived test proposals, **not tests executed by this lane**:

1. Seed owned zero-area values in another Session, then use `A=(B+1,0)` and
   `B=(0,1)`: GEMM must reject output extent before allocation/leaf entry despite
   zero reduction. At `M=B`, configure first-allocation failure before the new
   Session's first operation: expect allocation failure and one allocation
   attempt without attempting a huge allocation. `configureForTesting`
   (`:968`) is not legal after that Session has begun.
2. Exercise the huge empty examples above with a production generated-strict
   adapter, no injected candidate: require `empty_output` and no invocation.
   For small positive-output `K=0`, require all positive-zero results and no leaf.
3. Put strict order/FMA/subnormal/NaN witnesses in both full columns and tails
   of widths 1 and 2; include tile-N 1/64, `N=Tn-1,Tn,Tn+1,2*Tn-1`, bounded-M
   full/tail, and odd/even K. Raw private-leaf empty/K-zero checks remain a
   separate obligation from the source adapter's bypass behavior.
4. Exercise A=B, result/input variable alias, overlapping external reads and
   publication destinations, failure preserving old results/earlier effects,
   duplicate frontiers, allocation failure and hostile caller FP controls.

This audit supports the existing guard implication at the prospective source
connection boundary. It is not that connection's implementation, a whole-family
execution receipt, a performance claim, or a relaxation of the private leaf's
valid-descriptor and signed-arithmetic preconditions.
