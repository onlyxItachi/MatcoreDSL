# Private strict output-N peeling: implementation lane

Starting canonical checkpoint: `243f6772ebf8cc7600d05776a2adbb8d8a0515d0`.
Scope: additive compiler-private HOW derivation/inspection, not source-driver,
runtime, default/build selection, GPU, or automatic performance policy changes.
The existing unpeeled v1 Transform, derivation and verifier code remains intact.
Parent owns issuer enum/CLI/LLVM footprints and build integration; a separate
agent owns unit/numerical fixtures. This report describes this implementation
lane and does not independently qualify their work.

## Interface and upstream construction

`gemm_pattern::OutputTilePatternV1` is reused unchanged, with checked M/N tile
parameters in `[1,64]` and no K parameter. The additive functions are:

- `strictGemmOutputPeelNTransformV1(tiles,error)`;
- `deriveStrictGemmOutputPeelNV1(canonical_bufferized,tiles,error)`;
- `verifyStrictGemmOutputPeelNV1(scheduled,tiles,error)`.

For tile-N greater than one, the deterministic Transform program reuses native
tiling, generalization and M/K/N interchange, then native `transform.loop.peel`
on the outer tile-N handle. A native canonicalizer exposes static full-tile N
width and the separate remainder. Failed upstream mutation is confined to a
clone and discarded. No private tiler, accumulator transform, imported shader,
callback or arbitrary schedule/LLVM execution authority is introduced.

Tile-N equal to one is an explicit statically selected identity: exactly the
old v1 Transform and graph, with one contraction copy. It never attempts peel
and suppresses no failure. This matters because the pinned21.1.8 implementation
rejects constant step<=1, despite the more permissive-looking Transform TD
option. The exact native negative control is saved under
`/var/tmp/mdslc-peel-family-eYz0rM/unit-negative.{json,stderr}`: the 5x1 script
with an actual peel returns1, reporting failure to peel the last iteration.
The product unit branch therefore does not depend on that unusable operation.
See pinned [loop specialization](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/SCF/Transforms/LoopSpecialization.cpp)
and [Transform implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/SCF/TransformOps/SCFTransformOps.cpp).

## Independent proof before replay

Independent failure diagnostics begin with `peeled scalar envelope:`. Final
parameter-bound mismatch instead reports `peeled payload differs from
parameter-bound upstream replay`; tests can distinguish actual structural
rejection from replay-only refusal without a new import/inspection API.

The proof has two required sides of the native alias-folding boundary:

1. Before folding: complete module/function ownership, closed native/inherent
   attribute and operation vocabulary, exact strict generic maps/iterators and
   scalar bodies, single fill, and each copy's unique A/B/C subviews. Slice
   offsets, both lengths, source identity, unit strides, type-validity and sole
   generic operand use are checked. Even a length later erased by alias folding
   must already match the proven M/K/N domain.
2. After native Linalg-to-loops and alias folding, without unit-loop
   canonicalization: each copy must have exact tile-M/tile-N/local-M/K/local-N
   ancestry; bounded M and complete step-one K; ordered separate A/B/C loads,
   multiply, add and C store; exact original indices and one-use float SSA.
   One complete positive-zero fill precedes the shared M loop, and return is
   exactly the original C. Counts supplement these role proofs: one-copy/seven
   loops for unit N, two-copy/eleven loops otherwise, never an arbitrary range.

For tile N=t>1, the main bound must be exactly `F=t*floor(N/t)`. Main origins
are `[0,F)` step t, with local width t. The sole remaining domain starts at the
same F, ends at N, and uses local width `N-origin`. Both copies share the same
original M tile loop and full increasing K fold. For valid nonnegative N,
`0<=N-F<t`, so full and tail are disjoint and complete, including N=0. M=0
executes no contraction; K=0 preserves the original complete +0 fill.

All dimension queries are checked for approved matrices/views and proven axis
0 or1, including unused queries that a lower pass could delete. Index glue
must belong to the transitive dependency closure of independently proved
bounds/slices/accesses, or an explicit safe native unit-N extent-minus-one
helper. Harmless dead constants remain replay-only cases. An unused dynamic
dimension index, unproved semi-affine divisor or unrelated index program is
not accepted merely because native folding would erase it.

Index relations use bounded read-only linear forms with depth/node budgets;
intermediate coefficients stay within2^20 and pairwise products fit signed
i64. The F expression is matched structurally, not evaluated on untrusted
divisors. This is a narrow verifier vocabulary, not a replacement optimizer.
No input/descriptor noalias, nonnull, alignment or numerical permissions are
added. Integer overflow and floating fast-math properties are checked, as are
memory nontemporal properties and all other extra attributes.

The unchanged raw-leaf signed stepping contract remains explicit:
`M<=INT64_MAX-(tile_m-1)` and `N<=INT64_MAX-(tile_n-1)`, valid nonnegative dense
descriptors and fresh C disjoint from live inputs. A/B may alias. F<=N makes
its multiplication representable; the tail never pads N or partitions K.
The existing source adapter's byte bounds/bypasses are not a new source route.

## Evidence and preserved construction failures

- Prior P3 compile-only evidence is frozen at
  `/var/tmp/mdslc-p3-peel-kiv52W/REPORT.md`, SHA256
  `eae7012b9808a15a91e85a34e0034d947c34ba2e401bc9a10f73cdaa5c3dd97b`.
  It established three scalar C accumulators at LICM and later independent
  output-lane SLP for the 5x3 prototype, not product or performance acceptance.
- Standalone pinned native-MLIR controls in
  `/var/tmp/mdslc-peel-family-eYz0rM` accept 1x64 and64x2 construction with the
  same main/tail graph shape; the unit-step actual-peel control rejects. These
  inspect native construction only, not this C++ verifier or physical arithmetic.
- A native scalar-inspection checkpoint confirmed alias folding greedily
  deletes subviews even without a separate canonicalizer. The initial draft's
  post-fold view-count proof was therefore wrong and replaced by the required
  pre-fold slice proof plus post-fold access proof. No successful gate is
  claimed for that draft. Parent retains the first compile log at
  `build-peel-release/focused-build-first.log`; the draft core compiled, while
  two independently owned test construction overloads required correction.
- Review identified possible unused invalid dimension/semi-affine glue hidden
  by DCE. The complete dimension validation and proved-index closure above
  address that independent-proof gap; replay is not counted as its proof.

At this first implementation handoff, the parent-coordinated focused build,
mutant tests, raw-leaf numerical/empty/alias checks, generated-code sanitizer
controls and full regressions have not yet passed for the corrected source.
This lane ran no C++ build or generated physical program. Do not infer source
driver acceptance, package/target qualification, speed improvement, default
adoption, or permission to weaken the existing v1 verifier from this report.
