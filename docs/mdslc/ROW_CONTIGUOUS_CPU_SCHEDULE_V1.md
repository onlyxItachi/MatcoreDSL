# Strict row-contiguous CPU GEMM schedule

Status: **merged bounded engineering checkpoint**, [PR #60](https://github.com/onlyxItachi/MatcoreDSL/pull/60),
canonical merge `0b2c21f126288501fe27afea9320ec9fce392dcc`.
Campaign start: `0e8c040809ae1fe6b0f1c01e41fc509b07212be7`.

## What changes

One alternative compiler-owned realization uses upstream MLIR 21.1.8 Transform
`structured.generalize` and `structured.interchange [0,2,1]` to traverse GEMM
as M/K/N rather than M/N/K. Standard Linalg loop lowering and LLVM optimize the
result. This is not an explicit MLIR Vector pipeline, tiling, fusion, transformed
whole-region execution, automatic storage reuse or a new source operation.

The internal opt-in build selector is
`-DMDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE=row-contiguous`. The default remains
`scalar`; an unknown selector is a configuration error. Both choices use the
existing generated-strict candidate identity and private ABI, because they
implement the same operation contract. Build-selected artifacts and the manifest
identify the physical implementation. This is not runtime cost-based dispatch.

## Numerical and storage argument

For each output `(m,n)`, K starts at zero and increases by one over the full
reduction range. The accumulator starts at positive f32 zero; every update is
one separate f32 multiply followed by one f32 add. Moving the independent N
loop inward does not regroup the reduction, introduce partial sums or erase an
operation's rounding boundary. The pinned upstream scalar-loop lowering is
part of this argument; a Linalg reduction iterator label alone is insufficient.

Before lowering, the verifier requires exactly:

- the compiler-owned dynamic rank-2 f32 function and original result destination;
- one positive-zero fill, one generic contraction and the destination return;
- indexing maps `(m,k,n)->(m,k)`, `(m,k,n)->(k,n)`, `(m,k,n)->(m,n)`;
- iterator kinds `[parallel,reduction,parallel]` and the canonical multiply/add
  scalar body, with no fast-math flags or extra operations/attributes;
- no new allocation, copy, publication, external call or asserted alias fact.

One-Shot Bufferize is still applied and checked before this schedule. The
private destination remains isolated from both inputs. The inputs may alias
one another, including being the same Value. Dynamic/empty/zero-K behavior and
all caller guards remain unchanged. Physical input validity, capacities and
race freedom are not inferred by this static verifier.

The issuer accepts only its own authenticated built-in primitive, not imported
MLIR or serialized authority. Transform runs on a clone; failure discards that
candidate and leaves the original stage unchanged. Semantic, structured,
bufferized, Transform, scheduled and LLVM identities are separately recorded.
Source admission, exact untransformed region witness, static orchestration,
private candidate DSO and canonical provider-policy ownership remain unchanged.

## What upstream supplies and what Matcore retains

MLIR supplies named-op generalization, interchange and loop lowering. LLVM may
auto-vectorize independent N lanes: the initial Release object contains SSE
`mulps`/`addps` and scalar remainder arithmetic, without fused instructions.
An independent object test checks the achieved unsanitized form. Instrumented
builds are not assumed to make the same optimization decision.

Matcore supplies the operation contract, closed derivation, schedule-specific
postconditions, artifact identity and existing runtime/effect guards. This
experiment requires no replacement bufferizer, private schedule language or
LLVM backend modification.

An explicit masked MLIR Vector 4x8x1 alternative also executed correctly in the
research lane, but did not win the sampled performance comparison. Preserve its
reproducers and negative results without making it product policy. Unqualified
vector-contract lowering also has an executable FMA counterexample: strict
semantics cannot be assumed from pass success or absent initial fast-math flags.

## Validation and claims

Independent source fixtures exercise full SIMD-size outputs and tails,
noncommuting rectangular math, strict FMA and K-order discriminators,
non-finite/subnormal values, old-value preservation, late alias reads, owning
observations and late failure prefixes. A separate fixture passes one immutable
Value as both inputs. They run through the actual generated-strict source driver.
Stage mutations challenge indexing, iterator roles, operands, accumulator,
numerical permissions, extra calls and destination identity. Generated-code
sanitizer negative controls distinguish actual instrumentation from linked
sanitizer libraries.

Pre-integration exact-artifact experiments and their limitations are retained in
[the independent report](agent-reports/mlir-hpc-legality-v1.md).

Timing evidence is bounded to the recorded host, shapes, numerical profiles,
warm-buffer regime and artifact identities. A win over the previous scalar
generated candidate is not BLAS parity, especially when strict and provider
numerical profiles differ. No new target or universal performance claim follows.

## Exact integration and subsequent experiment disposition

Pre-merge head: `58193878394c774c3365dbf746d1622e951b0813`, clean.
Production delta originated at `9769dbafc18d4b8efd05c2e3519543eb57531604`;
later focused commits add independent wide-read ASan control and review evidence.
Final-head local Release (OpenBLAS required) passed **138/138**, 331.69 seconds;
affected Debug ASan/UBSan (OpenBLAS OFF) passed **83/83**, 286.39 seconds.
Focused generated suites passed **11/11** and **10/10**, respectively;
the candidate unit test passed **45 checks, zero failures**. These are correctness
test durations, not benchmarks. Installed/consumer/ABI/storage/provider ownership
regressions are included in the stated suites.

All **21 hosted checks** succeeded before normal merge, including push/PR
instances of native, Windows and hygiene plus legacy CI. Relevant PR runs:
[native 34289281056](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34289281056),
[Windows 34289281055](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34289281055),
[hygiene 34289281039](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34289281039),
[legacy 34289281057](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34289281057).
Merge parents preserve starting main `0e8c040809ae1fe6b0f1c01e41fc509b07212be7`
and accepted head `58193878394c774c3365dbf746d1622e951b0813`.

Durable, unmerged experiments remain separate:

- [Exact-artifact six-shape comparison](https://github.com/onlyxItachi/MatcoreDSL/blob/3650fd4c31db6c8bf7d5eb2d4432799ca425e7fa/docs/mdslc/agent-reports/mlir-hpc-execution-evidence-v1.md):
  row source medians beat the previous generated scalar route in those samples;
  provider numerics and timing layers remain distinct.
- [Cache follow-up decision](https://github.com/onlyxItachi/MatcoreDSL/blob/93e340a3121f81e5d365942d72257ca7aaae4151/docs/mdslc/agent-reports/cache-measurement-and-numerical-next-v1.md):
  the connected 4x64x32 cache specimen passes broad arithmetic/effect controls
  but loses all six comparisons to row. Preserve it as research, not a new default.
- [Coherent 22.1.8 control](https://github.com/onlyxItachi/MatcoreDSL/blob/e7e67583e3fbc9123a7611360b5eee2a4aebfc01/docs/mdslc/MLIR_HPC_22_CONTROL_V1.md):
  actual research execution succeeds; no product migration follows.
- [Source-library feasibility](https://github.com/onlyxItachi/MatcoreDSL/blob/3669d4d92178ea330227294e2f489c73f3f80906/docs/mdslc/agent-reports/semantic-modules-feasibility-v1.md):
  source-visible libraries need explicit multi-file source identity, not opaque
  function types masquerading as semantic authority.

The next leaf boundary preserves the already-proven private output-data alias
fact through LLVM; source-library work proceeds independently. This first schedule
does not complete the full campaign objective or authorize whole-region reuse.
