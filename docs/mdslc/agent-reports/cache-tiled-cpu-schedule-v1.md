# Connected strict cache-tiled CPU schedule

2026-09-09. Base `9769dbafc18d4b8efd05c2e3519543eb57531604` on isolated branch
`mdslc/cache-tiled-cpu-schedule-v1`. This is a candidate integration checkpoint,
not a canonical merge, public interface freeze or performance-selection claim.
The preceding reproducible upstream research is on branch
`research/mlir-structured-hpc-v1` at
`6ce4b3f6035f1a4ae8f5395753f9e0202eabdffc`.

## What is connected

The existing closed issuer can explicitly select `CacheTiledMKN` through private
`matcore-cpu-gemm-candidate --output FILE --schedule=cache-tiled` or advanced build
setting `MDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE=cache-tiled`. Scalar remains the
default and row-contiguous remains selectable. The semantic primitive, source
admission, whole-region paired witness, emitted source orchestration, runtime
guards, owning values, publication/observation/failure contracts, private DSO and
provider policy are unchanged. No arbitrary source or serialized IR is accepted
by the issuer. Generated strict remains the same registered candidate class.

The exact upstream Transform sequence tiles a verified bufferized matmul by
`[M=4,N=64,K=32]` with sequential outer M/N/K loops, generalizes the inner matmul,
then interchanges its traversal to M/K/N. One initial positive-zero fill remains
outside all tile loops. Each inner contraction updates the existing C subview,
never an independently zeroed partial sum. The selected `4/64/32` constants are
internal scheduling choices, not mathematical metadata or tuned thresholds.

For every fixed output element, increasing outer K chunks and increasing inner
K enumerate the original reduction exactly once in order. Independent outputs
can be interleaved because the adapter requires immutable inputs and a private,
disjoint destination. Compatible dimensions, capacities, index/byte extents,
strides, FP controls and lifetime remain caller/adapter obligations. This adds
neither a physical alias proof about arbitrary descriptors nor new assumptions
to the source semantic representation.

## Trusted transformations versus checked postconditions

**Trusted compiler machinery:** pinned MLIR 21.1.8 Linalg tiling, generalization,
interchange, structured lowering, strided-metadata expansion and LLVM lowering.
The implementation registers Linalg's external tiling interface explicitly; the
all-dialects CLI had supplied that registration in the research experiment.
Each transform works on an isolated clone. Failure discards the clone; no
partially transformed payload replaces the original verified input.

**Mechanically checked envelope:** the original bufferized primitive is verified
first. The scheduled result permits only the expected op families, one top-level
fill, three sequential loops, three subviews, one generic contraction, and one
scalar f32 multiply/add pair without numerical flags. No allocation, copy,
external call, vector contraction or parallel loop is admitted. An exact replay
of the compiler-owned schedule from fresh canonical semantics is compared using
upstream `OperationEquivalence` with only `IgnoreLocations`; operation properties,
attributes, types, ordering and SSA bindings are retained.

Replay detects changed bounds/steps, minima, subview sources/offsets/sizes, scalar
operands, reduction indexing, zeroing position and destination return. It is
**fixed-pipeline identity/drift checking, not an independent theorem about MLIR
or a generic loop-equivalence engine**. The bounded semantic argument and actual
adversarial execution are separate evidence. No new proof DSL or private tiling
algorithm was introduced. Exact nested arithmetic operand order is tested on
the pinned equivalence implementation, not generalized to other versions.

After LLVM translation, the prior two-definition/one-multiply/one-add gate and
no-fastmath/no-provider/unknown-call refusal remain. Cache tiling alone permits
exactly three leaf calls to the standard `llvm.smin.i64` intrinsic, with intrinsic
ID, declaration status and exact `i64(i64,i64)` signature checked. These encode
integer tail minima, not fallible provider calls. No other intrinsic/declaration
exception was introduced. The manifest binds unchanged semantic/structured/buffer
identities and distinct Transform/scheduled/LLVM identities in actual pass order.

## Validation performed before this implementation commit

| Scope | Actual result |
| --- | --- |
| Structural candidate test, final test source | 70 checks, zero failures in Release and Debug ASan/UBSan |
| Existing generated source/leaf/object/negative scope, Release OpenBLAS ON | 9/9 passed, 6.47 s, before four final test-only mutations |
| Same scope, Debug ASan/UBSan OpenBLAS OFF | 9/9 passed, 13.11 s, before four final test-only mutations |
| New independent actual-source boundary matrix | Cache Release and cache ASan each 225 cases / 563,296 checks passed; independent scalar and row Release controls also passed |
| Scalar LLVM preservation | Byte-identical to established scalar artifact, SHA `53c75982ef45ed9cc103e3a6f4c47ade84e944b7d19af17e26b0cd60b0fa6f54` |
| Row LLVM preservation | Byte-identical to row artifact, SHA `8ef2b5b27e7f5fcd5fc58f6269927d341a8c31173a904c00263a1d2a1aa3e6b7` |
| Duplicate private schedule CLI choice | Exit 2 before artifact creation |
| Repository whitespace | `git diff --check` passed |

Structural negatives include changed K bound/start/step/tail minimum, wrong C
source/offset/size, zeroing inside K, allocation/copy, contraction permission,
replaced/swapped scalar accumulator, swapped multiplication, altered indexing,
library-call injection and wrong returned destination. The independent source
matrix spans M=3/4/5/7/8/9, N=63/64/65/127/128/129 and K=31/32/33/63/64/65,
plus K-chunk rounding, FMA across chunk boundaries, nonfinite/subnormal cases,
second GEMM N=65, alias-sensitive late reads and failure/publication prefixes.
It was authored and executed by the independent adversarial lane, not this
implementation author; its fixtures/report are a separate integration commit.

The first embedded tiling attempt failed for absent external interface
registration; the corrected implementation supplies the upstream registration.
The first LLVM gate then correctly rejected the integer-min intrinsic before
its exact, schedule-specific handling was added. One broad CTest invocation
also selected two source tests before their driver target existed; both refused
missing DRIVER and were rerun successfully after the real driver build. None of
these setup/implementation failures is counted as successful evidence.

## Exact artifacts and remaining gates

Implementation CPP SHA:
`ac15b8fac8d21814879d06bf02b54e6e307fb2ec16820c51e50d8f645c432369`;
header `5cd8bc7e66a169ac716b140453a038c8f6a646ccc4032d6bc088ec8c83a79bc0`;
final structural test
`ae29d972a96d72500d7ded6bb55200ad94e36102b8f3f15895990138e4da467b`.
Builds are `build-cache-release` and `build-cache-asan` in this branch worktree.

| Artifact | SHA-256 |
| --- | --- |
| Issued cache LLVM | `cfe09efce2b8d05eb66e67f7a3b2a129fe85a4b9789da498fec748b6dd3f6c33` |
| Release object | `02a8657e898705e156c593a9771e4c69886390895027a256062944798f596fd8` |
| ASan production object | `0fac7af5fb462fbfe3d38f0898f25b695d91e567cffdf63b99b196ede28bd30f` |
| Release driver | `18af6cab78d570980211ebcf671e146785a941625b845f58efe6120283c728ad` |
| Debug ASan/UBSan driver | `85a7bc7e7e4d02b65ba54bcfb828023d54abc344906c5eb9a61823aa8b382ffd` |

Independent/root architecture review found no concrete semantic blocker for
this bounded composition. Full integration regression, hosted CI and any normal
merge remain the integration owner's gates; they were not run by this lane.
The connected production object has not yet received a quiet performance
comparison. Research timings for another vector candidate do not transfer here.
No default dispatch decision, threshold, BLAS parity, transformed whole-region
execution, storage reuse, fusion, API/ABI freeze, Windows or accelerator claim.
