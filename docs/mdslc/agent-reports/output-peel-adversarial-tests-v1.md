# Private output-N peel: independent adversarial fixtures

Status: root-executed local Linux x86-64 inspection/CLI and 5x3 raw-leaf controls
passed as recorded below. This agent independently inspected retained logs/XML
and build rules, without compiling or executing anything. Broader family,
ARM, hosted CI and source-route qualification are not claimed by this receipt.
The work is additive to unchanged v1 tests and uses the new private issuer's
actual objects, not the P3 fact binder or imported LLVM.

## Inspection fixture

`compiler/tests/generated_cpu/output_peel_test.cpp` directly exercises
`deriveStrictGemmOutputPeelNV1` / `verifyStrictGemmOutputPeelNV1`. It requests all
36 pairs of M/N tiles in `{1,2,3,5,16,64}`, checks deterministic replay and exact
parameter identity, unchanged canonical WHAT stages, and continued old-v1
acceptance. N-tile 1 is explicitly the exact old Transform/graph: there is no
N tail and the failing upstream peel operation must not be invoked for it.
These points are not empirical coverage of every integer tuple or target.

Semantic adversaries start from the independently checked 5x3 graph. Each must
change the payload, remain valid MLIR, and fail with `peeled scalar envelope:`
before replay. Invalid-IR failure or replay mismatch does not satisfy this gate.
Both full and tail products are attacked independently:

- off-by-one/wrong-dimension split; omitted, overlapping or gapped ranges;
- wrong origins/steps, full width tile+1, padded tail width, wrong B/C indices;
- clipped/late/stride-two K, swapped inputs, wrong C row/source;
- wrong accumulator/yield, contraction/reassociation, missing/repeated/-0 fill;
- allocation effects, invented alias/source authority and wrong result identity.

Separate B-K and C-M view-length +/-1 mutants attack both copies, and separate
tail C-N length +/-1 mutants keep the other operands' bounds unchanged. Some
such lengths disappear under memref alias folding; they must be proven at the
structured subview checkpoint, not inferred from surviving data addresses.

Unused index operations are not automatically harmless: dynamic `dim(A,1+1)`
and `dim(A,M)` can access an axis outside rank two, while a semi-affine apply
dividing N by the dynamic expression `K*0` is undefined. These still have to
pass native MLIR verification before a structural refusal counts. If upstream
MLIR disallows the semi-affine form, its explicit unsupported count is reported
and is not a semantic falsifier. The two dynamic-dimension cases require valid
MLIR. The benign unused constant below remains a distinct replay-only control.

A separate harmless dead index constant is required to reach only the replay
diagnostic. This prevents the exact replay check from hiding a weak independent
envelope. Failed inspections must leave originals and inspected inputs unchanged.

## Raw issued-object fixture

`compiler/tests/generated_cpu/output_peel_execution_test.cpp` includes the
unchanged independent `output_pattern_execution_test.cpp` with its main renamed.
The executable links the root-wired **new private issued object**, reusing its
volatile separate-f32/increasing-K oracle, descriptor checks, exact-bit output
checks, NaN-class policy, input immutability, output canaries, ordinary lane-
distinct rectangles, overlapping A/B inputs and empty/zero-K controls. This
does not exercise source-time driver binding or adapter failure semantics.
Additional valid empty calls use null C data with real descriptors for M=0
and/or N=0, nulling A/B only when their own extents are empty. Positive-output
K=0 calls retain nonzero-primed C and require every result bit to be +0.

The additional corpus fixes the inherited 5x3 gap: N=129 is divisible by three,
so that old fixture's explicit long order/FMA cases alone miss peeled N tails.
Every applicable frozen-bit witness now runs at parameter-relative full/mixed/
tail widths, explicitly including N=4,5,7,8, M=tileM and tileM+1, K=1..6 and
additional K=33/65 order/FMA cases. The same witness reaches full lanes and N
remainders one/two where those remainders exist. Maximum-tail widths and exact
tile multiples are included. N-tile 1 has no N tails; N-tile 2 has no width-two
tail; M-tile 1 has no M remainder. Inapplicable short-K cases are counted, not
claimed as executed witnesses.

Frozen constants defend non-FMA order, increasing-K cancellation, RNE addition,
positive/negative minimum subnormals, subnormal products, inexact RNE underflow,
overflow, Inf*0, opposed infinities, NaN class, and the positive-zero seed. For
the seed case **all** products are -0, without +0 padding that could hide a wrong
negative fill. Sensitive terms appear at the end of K to exercise odd/even
unroll epilogues. Oracle results must match the analytic constants before their
use against the issued leaf. An extra last-K-only integer product depends on
both row and column; all preceding prefixes are zero and products are exact
f32 integers at these bounded sizes. Ordinary nonuniform cases separately guard
against addressing mistakes that uniform arithmetic witnesses could mask.

The fixture retains `MDSLC_PATTERN_TILE_M/N`, the existing raw C-wrapper symbol,
`--corrupt-output`, and `--oob`. New `--oob-full` uses M=1, K=2, N=tileN and a B
allocation of exactly 2*N-1 floats while declaring the full shape. This deliberate
private-precondition violation must be diagnosed by emitted-leaf ASan; it is not
a supported-call result. The retained N=1 `--oob` reaches the N-tail route for
nonunit N tiles; the new probe reaches a full tile. Root owns their separate
CTest registration and diagnostic matching.

## Qualification boundary

No compiler/runtime/default/source-route or performance harness changes belong
to this lane. Existing source-time classifiers intentionally remain unchanged;
new source execution requires a later exact classifier/binding qualification.
Only the local inspection/CLI and 5x3 normal/generated-ASan results below are
asserted here; no broader family, ARM, source, runtime or performance pass follows.
P1 remains INCONCLUSIVE_TIMING; these fixtures neither retry it nor infer speed
from a register-accumulator code shape. The initial private variant is baseline
x86-64/AArch64 only as selected by root; nonbaseline ISA admission is not tested
or implied.

## Retained first-build diagnostic

Root's first focused compilation is retained in sibling build directory
`build-peel-release/focused-build-first.log`. It reported two test-only overload
errors: `ConstantIndexOp::create` inherits the MLIR 21 `ConstantOp::create`
interface, which requires a typed attribute rather than a bare integer. The
follow-up uses `ConstantOp::create(..., builder.getIndexAttr(1/42))`, preserving
the late-tail gap and benign replay-only constant. No build was rerun by this
agent, and the failed receipt remains evidence rather than a passing result.

## Root-executed first native controls

Retained evidence is in the sibling
`/home/hamza-usta/mdslc-work/correctness-campaign-v1/build-peel-release`:

| Receipt | SHA256 |
| --- | --- |
| `peel-first-controls.log` | `aa3018a86c6eb1edaea18c8f7ba5f12cdc49d0edb118d7745033674e33615be8` |
| `peel-first-controls.xml` | `12a66652dfde32b60fcf7947c670dbc295ce6686eb011ebec596fae31d90ebe6` |
| `peel-5x3-controls.log` | `c12968ac1953008f1dd4d468836e4b9c2971816a01dff5255047c57382339db9` |
| `peel-5x3-controls.xml` | `29ffe3d09fbfa7e9ea27f00eeac90f5c73ecc604b4ff1d8855410cecd3e9f521` |

`peel-first-controls` records 2/2 CTests passed, zero skipped: inspection reports
627 checks, 56 upstream-valid semantic mutants, zero failures and zero
unsupported mutants. Thus the semi-affine negative was actually valid MLIR and
rejected by the independent envelope. The separate benign-42 replay control
also passed its assertion. CLI reports 22 refusals and eight issued
parameter/target sets; AArch64 emission here is not native ARM execution.

`peel-5x3-controls` records 6/6 CTests passed, zero skipped: normal positive,
normal corruption detection, generated-ASan positive, ASan corruption detection,
tail OOB detection and full-tile OOB detection. The negative helper assertions
require the generated-leaf heap-buffer-overflow diagnostic, not an arbitrary
failure. These deliberately invalid capacity calls are not supported-call cases.

Each of the two positive processes reports **1,618 cases / 558,043 checks / zero
failures**. This is a cumulative count: it already includes the reused **269
cases / 372,457 checks** printed first. Do not add that subset again. The added
corpus contributes 1,349 cases / 185,586 checks per positive process.
Strict-witness participation is full=852, tail-width-one=426 and tail-width-two=426;
these counts overlap for mixed full+tail shapes and must not be added as disjoint
cases. Eighty inapplicable short-K witness combinations were excluded explicitly,
not silently counted as executed witnesses or reported as skipped CTests.

The actually issued `tests/generated_cpu/peel-5_3-normal.ll` hashes to
`7864ea0110a3567ad29c0f27d8a64369f05fb4f440625d457a4712b8021d3496`, exactly equal
to preserved P3 `peeled-bound.ll`. This is a native closed-issuer reproduction;
the execution target does not consume the P3 binder or that external file.

## Independent wiring inspection

No blocking mistake found in the reviewed wiring:

- The new private CLI branch selects only `OutputTiledPeelNMKN`, validates the
  existing 1..64 parameter carrier and refuses conflicting schedules/candidates,
  imports, unknown flags and nonbaseline ISAs. The direct issuer repeats the
  baseline-only and closed-target checks; no source-driver/default selection
  was added.
- `OutputPeelTests.cmake` registers a separate 44-test family: seven profiles,
  six controls each, plus inspection and CLI. The generated 5x3 Ninja rules bind
  the matching tile macros and actually link the new issued object. Its normal
  leaf uses `-O2 -ffp-contract=off`; the ASan leaf additionally receives issuer
  instrumentation attributes and Clang address instrumentation. Harness flags
  separately retain contraction-off/rounding semantics and sanitizer linkage.
- Both sanitizer capacity probes are registered separately; corruption still
  matches the retained oracle's intentional nonzero failure receipt.
- The LLVM footprint remains exactly two definitions, separate unflagged
  multiply/add and the original narrow intrinsic/call allowlist. Only the
  already-verified peeled nonunit-N recipe expects two arithmetic copies;
  unit N and every old recipe still expect one. Existing aligned-C-only noalias
  preservation and module verification remain mandatory after lowering.
- Workflow filters give the new underscore-prefixed tests their own exact-44
  gate, leaving the retained dot-prefixed 236-test sanitizer scope and separate
  two-test SPIR-V gate intact. Registration/source inspection is not evidence
  that either hosted gate has run or that all seven new profiles have passed.
