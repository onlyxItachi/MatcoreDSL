# Upstream structured CPU schedules: bounded research evidence

2026-09-09 local date. Starting canonical main
`0e8c040809ae1fe6b0f1c01e41fc509b07212be7`, clean on inspection.
Research branch `research/mlir-structured-hpc-v1`; no production source,
candidate registration, runtime policy, public syntax or execution authority is
changed. Reproducible inputs live in
[`compiler/experiments/mlir_hpc_v1`](../../../compiler/experiments/mlir_hpc_v1).
Generated artifacts remain outside the repository.

## Verdict

**OBSERVED:** stock coherent MLIR/LLVM 21.1.8 can tile a dynamic GEMM, perform
masked vectorization, lower it to an ordinary x64 object, and execute strict
increasing-K f32 arithmetic with masked tails. No private loop/vector compiler
was necessary. **Not every upstream schedule preserves Matcore strict semantics.**
One contraction alternative physically violates the strict FMA discriminator.

**INFERRED selection:** do not integrate this particular explicit-vector schedule
as the default. It passed correctness but lost all six measured shapes to the
independent row-contiguous MKN schedule, and lost two to the original baseline.
Keep the reusable upstream mechanism and the negative results; schedule policy
must follow evidence, not the presence of more vector operations.

## Exact experiment

Both successful inputs start from default-space identity-layout rank-2 f32
memrefs, one positive-zero `linalg.fill`, then `linalg.matmul` into the same C.
The private leaf assumes checked compatible nonnegative dimensions, truthful
capacities/lifetimes/strides, immutable A/B and C disjoint from both inputs. A/B
may alias each other. The host adapter remains responsible for those conditions,
nearest-even/masked exceptions/gradual underflow, private allocation, completion,
publication and restoration of the caller FP environment.

1. `tiled_vector.mlir`: Transform `tile_using_for [4,8,1]`, then
   `structured.vectorize vector_sizes [4,8,1]`.
2. `tiled_interchange_vector.mlir`: generalize/interchange to MKN, tile
   `[4,1,8]` with outer-loop order MNK, then vectorize `[4,1,8]`.

Both retain an outer increasing-K loop with step one. Explicit vectorization
produces separate `arith.mulf` and a unit-extent `vector.multi_reduction`, not a
`vector.contract`. This bounded schedule needs no padded tensor or extra
arithmetic in the reduction domain. Inactive tail lanes may contain poison;
masked memory accesses protect bounds and those lanes are not published.
Exact private-candidate FP exception flags are not a source-observable contract.

The shared lowering pipeline, in order, is:

```text
canonicalize
lower-vector-multi-reduction
lower-vector-mask
convert-vector-to-scf
convert-linalg-to-loops
expand-strided-metadata
lower-affine
convert-scf-to-cf
convert-vector-to-llvm (vector-contract-lowering=parallelarith)
convert-arith-to-llvm
convert-ub-to-llvm
finalize-memref-to-llvm
convert-func-to-llvm
convert-cf-to-llvm
reconcile-unrealized-casts
mlir-translate --mlir-to-llvmir
clang-21 -O3 -ffp-contract=off -c
```

The CLI reproduction uses `test-transform-dialect-erase-schedule` only to remove
the research Transform program after interpretation; it supplies no scheduling
mechanics. Production can keep schedule and payload modules separate using the
existing interpreter API. This report does not propose a product test-pass
dependency. The constants are research scheduling choices, not semantic fields.

## Actual validation

Run `bash compiler/experiments/mlir_hpc_v1/run.sh [OUTPUT_DIRECTORY]`.
Default tools are the established extracted 21.1.8 prefix and `/usr/bin/*-21`;
the script exposes explicit environment overrides. On a previously validated
x86-64-v3 execution host, set `MDSLC_RESEARCH_RUN_V3=1` to run the rejecting FMA
counterexample as well. It is not unconditionally executed on other machines.

Final reproduction directory: `/tmp/mdslc-mlir-hpc.kr7V2m/final`.

| Check | Actual result |
| --- | --- |
| First vector schedule, optimized object | 59,869 checks, zero failures |
| First schedule, generated ASan plus harness ASan/UBSan | 59,869 checks, zero failures |
| Interchanged vector schedule, optimized object | 59,869 checks, zero failures |
| Interchanged schedule, generated ASan plus harness ASan/UBSan | 59,869 checks, zero failures |
| Invalid input-capacity negative, each instrumented leaf | Heap-buffer-overflow READ in `research_gemm`, expected nonzero |
| Strict-FMA counteroracle | Actually differs; not an ineffective fixture |
| Repository whitespace | `git diff --check` passed |

The suite covers 700 deterministic random rectangular/dynamic geometries, NaNs,
infinities, signed zeros and subnormals; 320 systematic zero/tail geometries;
both carried sides in dependent noncommuting rectangular GEMMs; legal shared A/B;
bitwise immutable input checks; leading/trailing destination sentinels; and
volatile separate-f32 multiply/add increasing-K output oracles. NaN payloads are
unspecified, so NaNs compare by classification. Exactly sized inputs under ASan
exercise tail reads; malformed leaf capacities are deliberately outside the
contract and establish actual instrumentation, not a leaf guard.

The independently authored execution lane at harness
`9f3aed9917ee6a49d765f2681721d67b25fd5393` also passed 24 strict cases, including
larger square 512, 7x13 FMA/cancellation/subnormal/signed-zero tails, output
corruption detection and exact immutable/sentinel checks. That lane is distinct
from this agent's own oracle and from authenticated public-driver execution.

## Rejected contraction alternative

`negative_contract.mlir` uses static 1x2 times 2x1 matmul,
`vectorize_children_and_apply_patterns`, and outer-product vector contraction
lowering. Translation produces `llvm.fmuladd`, despite no source permission for
fused arithmetic. For A `[-1, 0x1.000002p0]`, B `[1, 0x1.fffffep-1]`:

- Baseline x86-64 executable prints actual `00000000`, expected `00000000`.
- Same LLVM IR compiled for x86-64-v3 on the actual Ryzen AI 9 HX 370 prints
  actual `337ffffe`, strict expected `00000000`, and exits exactly 1.
- `-ffp-contract=off` does not remove contraction already authorized by an LLVM
  intrinsic. The source meaning was lost before machine instruction selection.

Pinned upstream code explains the observation: the floating-add contract helper
creates `vector::FMAOp` for vector accumulators, and Vector-to-LLVM maps that to
`LLVM::FMulAddOp`. This is an incompatible route for this bounded strict contract,
not a claim that every vector contraction or upstream lowering is incorrect.
[Contract lowering](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/Vector/Transforms/LowerVectorContract.cpp#L111-L147),
[FMA lowering](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp#L1122-L1131).

## Independent warm primitive timings, including losses

The execution agent ran a coordinated quiet window, pinned CPU 2 on the same
Ryzen. Nine batch-mean samples per cell, warm reused buffers, single process;
SMT sibling/desktop/governor/thermals were not isolated or reconfigured. These
are primitive-only medians in microseconds, including output initialization but
not snapshots, allocation, publication, dispatch or FP scopes.

| Shape M x N x K | Original | Explicit vector | Row-contiguous |
| --- | ---: | ---: | ---: |
| 16 x 16 x 16 | 0.996 | 1.637 | 0.459 |
| 128 x 128 x 128 | 970.079 | 794.675 | 126.348 |
| 512 x 512 x 512 | 127040.196 | 51369.660 | 7960.349 |
| 128 x 256 x 64 | 875.531 | 796.967 | 127.225 |
| 16 x 512 x 256 | 1916.593 | 800.616 | 123.374 |
| 65 x 67 x 63 | 94.520 | 113.904 | 24.943 |

Raw evidence lives in the execution lane at
`benchmark_reports/hpc-execution-comparison-v1/evidence.json`, run started
`2026-09-08T22:46:01.872478+00:00`. It freezes all objects and commands and retains
layer/numerical distinctions. No BLAS parity, universal threshold, cost model,
end-to-end speedup or target-generalization claim follows from this table.

## Provenance, limits and next experiment

Tool hashes: `mlir-opt`
`18c322575dc399010d8186ff72cbf10047ff7ce99d8e11aec162f33820c89ed3`;
`mlir-translate`
`a9d4edaa6c7bb110d869c39a198d46a4806506db1c094fb18c8d7ad5e42b959f`;
Clang `412bbe8c60571a1eb06f48fde89635033621caeb01a9b4ee76d46711bae8e932`.
Both successful schedules optimize to the identical object
`5d472f0eaa7ffbf51fb16c1016f35ed990606f8ed18511191b2f0ba51c1aa231`,
7332 text bytes. Their unoptimized LLVM IR differs; LLVM already canonicalizes
that difference. Sanitized objects are respectively
`61b64425a215daed08cf153884dbfb0a2f78ca6b93e0e2f5610b454277835405` and
`5f155d287175b096965c9953c057a6d7f21440bd91675a49d270f8f112f3c4b0`.

Initial incomplete pipelines left `ub.poison`/affine operations or transfer
operations after conversion: fixed by explicit UB lowering and pass ordering,
not a custom lowering. One attempted ASan pass invocation used an invalid
analysis scope and failed; the final script uses `forceattrs sanitize_address`
then actual Clang ASan instrumentation. An initial external debuginfod lookup
stalled negative symbolization; the final script disables that optional service.
No failed attempt is counted as execution evidence.

Masked vector-size requirements and partial Transform failure behavior remain
upstream responsibilities, not new Matcore machinery.
[Pinned Transform/Linalg contract](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td).
Matcore must prove private-output and numerical legality before selecting a
schedule, then mechanically verify the issued derivation. This standalone
research input does not establish that issuance, a transformed whole-region
contract, fusion/reuse, provider policy, Windows, 22 compatibility or any device.
No whole-project or hosted CI was run for this research-only commit.

Next bounded research question: upstream non-unit K cache tiling with a single
initial fill and strictly increasing inner-K accumulation into the current C,
using inner MKN plus LLVM auto-vectorization. Independent partial sums or zeros
at each K tile would be a different numerical operation and must be rejected.

## Follow-up: non-unit K cache tiling, not yet performance-selected

The preceding experiment was committed at
`5a8577211b46c028fff390a5d82124d9c30a7683` before this follow-up began.
`tiled_mkn.mlir` now demonstrates that exact next research question:

```text
fill C once with +0
Transform tile matmul [M=4,N=64,K=32] using sequential scf.for
generalize the inner tiled matmul
interchange only its iteration order to M,K,N
standard Linalg-to-loops, metadata, affine/SCF, LLVM lowering
LLVM -O3 auto-vectorizes independent N outputs
```

**OBSERVED:** no explicit vector reduction is required. Scheduled IR has one
fill outside the tile loops, sequential M/N/K chunk loops, and an inner generic
with parallel/reduction/parallel iterators whose accumulator is the existing C
subview. The scalar body is separate f32 multiply/add without fastmath. The
object executes ordinary SSE `mulps` and `addps`; `memset` is its only undefined
symbol, implementing the zero fill. It contains 2510 text bytes. This is neither
a new private tiler nor an added provider/candidate registration.

The bounded mathematical justification is direct: for each fixed output (i,j),
K chunks are visited in increasing order; each chunk visits its actual K extent
in increasing order and starts with the previous rounded C(i,j). Changing the
relative order of different outputs cannot affect immutable A/B or disjoint C.
This argument depends on the checked private-output contract and on the actual
sequential lowering; a generic reduction label alone is not the proof.

The research oracle now explicitly straddles K=32 and K=64 boundaries with
7x13 outputs: initial `2^24`, then `1`, then `-2^24`. Required sequential
accumulation is +0; independently summing the later tile then combining gives 1.
Both counteroracles really distinguish the contracts. Systematic tails now
include K=31/32/33 and K=63/64/65, with exactly allocated input sizes.

Final follow-up reproduction:
`MDSLC_RESEARCH_RUN_V3=1 bash compiler/experiments/mlir_hpc_v1/run.sh /tmp/mdslc-mlir-hpc.kr7V2m/cache-final`.
All three schedules passed **77,985 checks each in ordinary execution and each
with generated ASan plus harness ASan/UBSan** (six runs). All three malformed
capacity controls diagnosed an actual generated-load heap-buffer-overflow. The
prior FMA alternative still returned the exact expected rejecting result.
The first two optimized object hashes remained unchanged; this is expanded
validation, not a retrospective change to their original 59,869-check evidence.

| Follow-up identity | SHA-256 |
| --- | --- |
| Schedule | `257d15b35ebd0be290526bc09792c88d53692f0067cf22f573b96cc3bc4b116b` |
| LLVM IR | `78edbeafc74b4dd45a9c47360f6b533533a2a93117a20515533fc62dcf736e8c` |
| Ordinary object | `2096d76115c45fbf0bba91fa2776ddf19e71a8311982a6f3689ee28dc1dad556` |
| Instrumented object | `85707a3d2d1cc3af2f482d581c92b9046aef0a36e8467b156c9275d3738524ee` |
| Expanded oracle | `2a05acb227798c3a4cc222a108cabea3e696d1add6194387d6368d2374e5ef9e` |

**UNRESOLVED:** this non-unit-K schedule has no independent timing result yet.
The earlier table describes the explicit-vector object, not this new object.
An independent execution/performance comparison must precede any selection;
4/64/32 are a research specimen, not a planner recommendation. The owning
integration lane retains the simpler row-contiguous candidate and unchanged
source/effect orchestration while it reviews and tests that bounded change.
