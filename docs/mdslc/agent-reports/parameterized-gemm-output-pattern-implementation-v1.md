# Parameterized strict GEMM output pattern v1: implementation lane

Scope: CPU-first compiler-private HOW extension on the isolated
`mdslc/parameterized-gemm-pattern-v1` integration worktree, based on `4c222dd`.
This lane owns `MatcoreGemmOutputPattern.h/.cpp` and
`MatcoreCpuGemmCandidate.h/.cpp`. Root owns build/CLI/CMake integration and the
independent test lane owns the new unit and execution fixtures. No commit or
merge is issued by this lane without integration-owner instruction.
For the implementation commit, root explicitly transferred ownership of its
completed `compiler/lib/regions/CMakeLists.txt` and private issuer CLI wiring;
those two integration files are included with this lane's core files/report.

## Implemented boundary

`OutputTilePatternV1` carries checked integer M/N tile parameters in `1..64`;
there is deliberately no K tile, padding, vectorization, numerical permission,
target mapping, imported IR, or execution-authority field. `{0,0}` means no
pattern only at the existing candidate selector. Nonzero parameters on any
other schedule, absent/partial/out-of-range pairs, and unknown choices fail
closed before issuance.

The new `OutputTiledMKN` schedule starts from the unchanged compiler-owned
strict semantic, structured and bufferized stages. Actual payload generation
uses only the pinned upstream Transform operations: tile the canonical matmul
by `[M_tile,N_tile,0]`, generalize its tiled operation, and interchange its
local traversal to M/K/N. Full positive-zero fill remains outside all tiles.
Transform runs on an isolated clone; partial failure does not mutate the
original or leave a candidate for reuse.

The target-neutral pattern verifier binds the complete scheduled operation
graph to exact upstream replay at the normalized parameter pair, ignoring
locations only. A separate narrow inspection lowers a clone with upstream
Linalg-to-SCF plus memref alias folding, without canonicalization that might
erase unit output loops. It checks tile origins/steps/full bounds, bounded
tails, scalar M/K/N nesting, the complete increasing `0..A.K` fold, ordered
input indices, positive-zero overwrite, exact old-C accumulator/store identity,
absence of tensor allocation/copy/provider/foreign operations, and unchanged
private symbol/type/attribute ownership. Replay and inspection are derivation
checks, not independent proof of the entire upstream implementation or source
authentication for arbitrary serialized IR.

CPU lowering consumes the checked scheduled stage and folds subview aliases
before the existing LLVM pipeline. Only the pinned pure i64 `llvm.smin` tail
intrinsic is additionally admitted; scalar f32 multiply/add and no-fast-math
postconditions remain. Descriptor facts still mark only fresh C's aligned data
pointer noalias; A/B may alias. Baseline Linux x64/AArch64 and the existing
isolated x64 ISA refinements reuse the same parameterized schedule, with target
triple/features/symbol changes only below the scheduled stage. ISA availability
does not authorize execution.

The manifest binds the pattern family, exact normalized parameters, bounds,
Transform and scheduled hashes, pipeline and retained caller requirements.
Scalar and existing row-contiguous branches retain their old construction and
manifest text; byte identity must be checked against baseline artifacts by the
integration owner rather than inferred from this source review.

## Integer stepping and source failure trace

A private raw tiled invocation must have representable next SCF tile indices:
`M <= INT64_MAX-(M_tile-1)` and `N <= INT64_MAX-(N_tile-1)`, in addition to the
existing descriptor/numerical/storage contracts. This matters for enormous
empty descriptors whose zero element count does not bound each dimension.

No source-level guard or new rejection is introduced. The existing authenticated
source adapter already bypasses the generated leaf for empty output and zero K
(`closed_host_v1.cpp`, `SessionAbiV2::gemm`, empty-math branch). Its existing
nonempty output extent check bounds f32 output elements by `PTRDIFF_MAX/4`.
Positive M/N are therefore each at most `PTRDIFF_MAX/4`; for tile sizes at most
64, the final next index is at most that bound plus 63 and is safely signed-i64.
The source Program, unchanged whole-region witness, capability checks,
FP controls, allocation/failure frontiers and ordered publication/observation
path remain untouched. This raw-leaf stepping precondition is not a semantic
shape restriction or a new runtime qualification limit.

## Validation and observed construction failures

This lane has run read-only inspection, an already-built baseline issuer, the
installed 21.1.8 `mlir-opt` transformation/inspection, and `git diff --check`; it
has not run a compiler build or physical execution suite. Root owns the
coordinated resource-budgeted builds.
Root reported the initial issuer compile succeeded, but first tiled issuance
was rejected by the new scalar coverage verifier. The inspection was then
changed to retain unit loops, resolve subview dimension bounds and not assume
loads precede generated affine index glue. A rebuilt request then identified
the M-tail comparison specifically. An installed-tool spike from the actual
built canonical buffered stage produced the intended full-fill, tile-M/tile-N,
local-M/K/local-N scalar form, base A/B/C indices and exact dynamic minima. The
remaining comparison canonicalized the actual affine expression but not its
expected counterpart; both now use the same affine simplification. Root then
reported rebuilt tile `3,5` issuance and the narrow CLI suite succeeded.

The first independent issuer-unit run reported 207 checks / 22 valid-IR
corruptions and one failing test expectation: tile `1,1` was required to differ
from the old scalar LLVM body. This is not a valid requirement: upstream may
erase unit local loops, recovering the same scalar MNK body. The parameterized
scheduled graph and Transform still differ, and nonunit patterns must generate
different structure. The test owner corrected that assertion to retain the
scheduled/Transform graph check while permitting the valid unit-tile LLVM
simplification.

### Parent-run narrow qualification

Root then ran the production issuer and new independent fixtures in
`/home/hamza-usta/mdslc-work/pattern-generation-v1/build-release` with the coherent
21.1.8 tuple. This lane inspected that build's
`Testing/Temporary/LastTest.log`; all of the following are actual parent-run
results, not inferred passes or executions performed by this lane:

- `generated_cpu.output_pattern.*`: **32/32 passed, zero skips**. This consists
  of the issuer/CLI checks and five controls for each of six raw-leaf profiles:
  normal execution, normal deliberate oracle corruption, ASan execution,
  ASan deliberate oracle corruption, and generated-leaf ASan out-of-bounds.
- Issuer: **242 checks, 22 well-formed IR corruptions, zero failures**.
- CLI: **17 refused requests and four issued parameter pairs**.
- Every normal and ASan execution below ran **269 cases** with **zero
  failures**. The strict ISA helper did not skip the AVX2/AVX512F profiles.

| Private generated profile | Checks per normal/ASan execution |
| --- | ---: |
| Baseline `3x5` | 344,782 |
| Baseline `4x16` | 361,755 |
| Baseline `1x1` | 321,149 |
| Baseline `64x64` | 1,279,267 |
| x64 AVX2 `4x8` | 360,083 |
| x64 AVX512F `4x16` | 361,755 |

These establish bounded issuer/mutation/CLI and physically executed raw-leaf
arithmetic, canary/input preservation and negative controls on this host.
They are not an authenticated source-route, complete package suite, AArch64
physical pass, GPU result, benchmark, universal ISA/shape qualification or
hosted-CI claim. Host UBSan compilation of the test harness does not imply
UBSan instrumentation of all generated LLVM arithmetic. Full source/package
and independent core audit remain integration-owner gates. Spike artifacts
live only in the build tree. Core implementation files are frozen for those
coordinated checks unless a concrete failure requires a correction.

## Explicitly deferred

GPU launch geometry/adapter changes and GPU tiling qualification are not part
of this increment. The pinned direct Transform
`gpu.map_forall_to_blocks` rejects dynamic forall trip counts; future dynamic
GPU specialization requires a separately established route and checked launch
law. Existing scalar GPU issuer/runtime and fused-pair recipes are unchanged.
SPIR-V remains an independent milestone. No performance gain, automatic
selection, CPU worker parallelism, arbitrary contraction execution, broad
fusion, device residency, imported-artifact authority, release or merge claim
follows from this pattern.
