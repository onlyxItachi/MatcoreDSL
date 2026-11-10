# Private output-N peel: independent adversarial fixtures

Status: source authored and inspected; **not compiled or executed by this agent**.
Root coordinates all builds and physical execution. This report is not a pass
receipt. The work is additive to the unchanged v1 tests and uses the new private
issuer's actual objects, not the P3 fact binder or imported LLVM.

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
No sanitizer, native, ARM, source, runtime or performance pass is asserted here.
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
