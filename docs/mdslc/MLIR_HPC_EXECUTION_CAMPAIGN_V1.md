# MLIR HPC execution campaign

Status: **implementation/reconciliation complete locally; final canonical
multi-source integration remains pending PR #67**. The initial investigation
below is historical; the bounded results and remaining merge gate follow it.

Canonical start: `0e8c040809ae1fe6b0f1c01e41fc509b07212be7` (PR #59),
engineering merge `8e0cf0f39833966227dee44d4367b99927a7ae03` (PR #58).
Local and live remote main matched; the canonical worktree was clean and no
open PR existed at campaign entry. Existing worktrees are preserved.

## Objective and competing hypotheses

Turn authenticated mathematical programs into useful, correctly optimized
machine execution. Test upstream MLIR mechanisms on actual workloads rather
than treating pass success as execution, correctness or performance evidence.
When this path is established, continue reusable library/module and multi-file
semantic compilation; a successful single-kernel experiment is not the campaign
terminal condition.

- **H1:** Upstream Tensor/Linalg/Transform/Bufferization/Vector machinery supplies
  the required structured scheduling and lowering mechanisms. Matcore needs
  semantic admission, legality-preserving derivations and implementation policy,
  not replacement transformation machinery.
- **H2:** Representative programs expose a missing semantic or resource planning
  abstraction. Introduce it only at the owning layer after a concrete
  counterexample; do not patch LLVM/backend architecture to hide missing source
  or runtime contracts.
- **H3:** Successful lowering still produces unsuitable implementations. Measure
  competing generated/native/provider candidates, record negative results and
  distinguish kernel cost from whole-region storage/effect cost.

## Independent lanes

| Lane | Branch | Responsibility |
| --- | --- | --- |
| Integration | `mdslc/mlir-hpc-campaign-v1` | Closed candidate derivation, implementation gates and reconciliation |
| Upstream advocate | `research/mlir-structured-hpc-v1` | Executable pinned upstream scheduling experiments |
| Execution evidence | `research/mlir-hpc-execution-evidence-v1` | Independent correctness and controlled timing harness |
| Adversary | `review/mlir-hpc-legality-v1` | Attempt to falsify numerical, alias, effect and authority contracts |

Agents own separate worktrees and reports. Experiments do not automatically
become product behavior. Normal focused commits, independent review and affected
local/hosted CI precede integration. Update CURRENT_STATE after each successful
engineering merge, retaining exactly identified canonical checkpoints.

## Initial execution experiment and invariants

The first controlled extension is a transformed realization of the already
issued strict GEMM primitive. Test tiling/vectorization of independent output
dimensions while preserving increasing K and separate f32 multiply/add. Keep
the source admission, exact untransformed region witness, static orchestration,
private candidate DSO and canonical provider adapter unchanged.

This first experiment does not prove transformed whole-region execution, fusion
or storage reuse. Those remain dependent investigations requiring their own
authenticated derivation and effect/lifetime evidence.

Acceptance includes actual object execution, rectangular/dynamic/tail/zero
cases, independent strict arithmetic and FMA/reassociation discriminators,
non-finite values and output sentinels, real sanitizer negative controls,
forged/mutated lowering refusal, and existing source/host failure-prefix tests.
No successful transform may silently grant FMA, noalias, new target authority,
provider equivalence or permission to erase a required check.

## Environment and evidence categories

Observed host: Linux x86-64 AMD Ryzen AI 9 HX 370, 24 logical CPUs. Hardware flags
alone are not an ISA execution gate. Use the installed coherent LLVM/Clang
21.1.8 tuple and extracted MLIR 21.1.8 package. The separately installed coherent
22.1.8 tuple is compatibility research only. No LLVM source rebuild or system
package modification is planned.

Keep OBSERVED, INFERRED, PROPOSED, PROVEN WITHIN A BOUNDED CONTRACT and UNSUPPORTED
distinct. Timings under concurrent builds or memory pressure are not controlled
performance evidence. Issue #15 remains partial/open; its existing envelope
cannot be completed by a few generated-kernel wins. Issue #20 remains a design
record, not a completed general tensor API.

## Campaign reconciliation still required

- Explain which upstream mechanisms suffice and which concrete counterexamples
  require Matcore-owned abstractions.
- Connect surviving schedules to authenticated source-generated execution and
  validate against independently implemented numerical/effect oracles.
- Measure actual implementations without extrapolating to untested hardware,
  shapes, providers or numerical profiles.
- Continue the justified multi-operation/resource and reusable semantic-module
  work rather than declaring the first executable schedule the full objective.
- Retain exact candidate, review, test, PR and canonical merge provenance; record
  losing alternatives and leave a concise current project map.

## Reconciled results

**PROVEN WITHIN A BOUNDED CONTRACT:** upstream MLIR mechanisms produce useful
source-connected CPU implementations. Matcore needed explicit operation/profile,
resource/effect and implementation-ownership contracts, not a replacement
tiler, vector language, register allocator or machine backend. This result does
not prove that MLIR discovers globally optimal schedules or that all future
planning abstractions are unnecessary.

| Surviving boundary | Reviewed/canonical identity | What it establishes |
| --- | --- | --- |
| Strict row-contiguous scheduling | `0b2c21f126288501fe27afea9320ec9fce392dcc` / PR #60 | Actual upstream Transform M/K/N realization without changing strict increasing-K separate arithmetic. |
| Fresh output-data isolation | `c04215e787832b35bbd6a85a51d9285b053181bc` / PR #62 | Only fresh output data receives noalias; input/input aliasing stays legal. This was not a blanket performance win. |
| Source-visible mathematical libraries | `917b5ecf1d325e525bc24e3c94764d2958980a93` / PR #63 | Sema-selected pure Value/Shape helpers and bounded templates in headers retain actual source/inclusion identity. |
| Explicitly permitted generated FMA | `c3b7b0b9a05e11d1b8d8cd3eff1c7ef3551a8280` / PR #66 | Separate authenticated per-GEMM permission, upstream register retention, isolated AVX2/FMA object, hardware/OS gate, preserved ordered effects and candidate ownership. |
| Authenticated multi-source program | Reviewed `e56e5a648295f7b5ec60bff31f30bf37c031c854` / PR #67, canonical merge pending | Genuine 2–8 source TUs, original host compilation, foreign interface/effect witnesses and protected symbol ownership before linking. Both candidate policies compose across source files. |

The distinction between immutable values, host resources and possible physical
aliasing survived. Required read checks, publication/owning observation and
sticky failure prefixes survived real overlapping-storage and later-failure
counterexamples. Ordinary host effects remain outside regions. Multi-source
linking is not opaque mathematical import or cross-region optimization.

The [source-to-generated contract](GENERATED_REASSOCIATE_CPU_V1.md),
[multi-source contract](MULTI_SOURCE_PROGRAMS_V1.md),
[composed 184/129 local regression record](agent-reports/authenticated-multi-source-v1.md),
[independent cross-file policy execution](agent-reports/program-reassociate-composition-v1.md)
and [independent campaign audit](agent-reports/mlir-hpc-campaign-terminal-independent-v1.md)
preserve exact source heads, artifact identities, actual failures and validation.
Neither a green branch nor this report substitutes for final exact-head hosted
CI, normal dependency-ordered merges and the operator checkpoint.

### What the measurements actually say

The [independently audited same-source study](agent-reports/generated-reassociate-source-measurement-independent-v1.md)
binds PR #66's frozen implementation `73cdeb9`, not an inferred post-link
multi-source performance result. One warm single-thread Ryzen AI 9 HX 370,
six shapes, 540 samples: the new generated candidate used 27.28–57.62% less
time than the previous generated candidate under the **same reassociated source
permission**. OpenBLAS won five of six. Schedule, target features and compiler
optimization level changed together; this is not an isolated FMA causal study.
Primitive and full-region costs have different allocation/address/cache histories
and cannot be subtracted to claim copy overhead or zero-copy.

No automatic selection threshold was introduced. Existing-native region policy
means the forced reference adapter, not the packed native performance envelope.
Issue #15 remains open: this study supplies neither its authenticated complete
native/provider envelope nor scaling, full planner regret or cooperative packed-B
preparation evidence. Issue #20 remains a design record, not an implemented
general tensor/view interface.

### Losing alternatives stay visible

- Masked-vector and cache `[4,64,32]` experiments were correct but slower on the
  tested workloads. They were not merged as schedules; useful boundary oracles
  survived. See retained branches `research/mlir-structured-hpc-v1` at
  `93e340a3121f81e5d365942d72257ca7aaae4151` and
  `mdslc/cache-tiled-cpu-schedule-v1` at `295b251100ff93516d80c69565e33b162c34eef5`.
- A vector-contract path introduced FMA and falsified strict arithmetic. Its
  useful mechanism was adopted only under a distinct permitted profile, never
  disguised by `-ffp-contract=off` after FMA already existed in IR.
- Wider x86-64-v3 research timing did not authorize a narrower AVX2/FMA guard.
  The actual narrower object and connected source were separately validated.
- Coherent 22.1.8 executed bounded research controls but did not become product,
  driver or installed-package truth. Keep `research/mlir-hpc-22-control-v1` at
  `e7e67583e3fbc9123a7611360b5eee2a4aebfc01` separate from coherent 21.1.8 support.
- Opaque mathematical symbols/ABI-compatible substitutes did not provide
  source semantics. Source-visible helpers won this bounded module decision.
  The real weakref call-only effect bypass and private-prelude namespace
  pollution were corrected; failed and corrected evidence remains recorded.

### Remaining ownership and exactly one next boundary

Matcore owns mathematical meaning, source authority, numerical permissions,
resource/effect/failure contracts and legal candidate choice. MLIR owns structured
transformations, SSA/tensor/buffer/vector machinery; LLVM owns machine lowering.
Runtime owns unresolved dynamic legality and the selected adapter's storage,
completion and provider-policy contract. Providers supply admitted implementations,
not language semantics. No new public format/API/ABI is frozen.

Whole-region MLIR is still an exact untransformed witness; sealed Program
orchestration calls issued primitives. General rank-N/views, fusion, automatic
reuse, devices/residency/asynchrony, generated-region Windows, opaque mathematical
modules and a learned/autotuned cost policy remain unproven or unsupported.

**Next: one source-authenticated publication-to-read forwarding derivation.**
Keep the authoritative Program/witness unchanged and mechanically check a
separate derived plan. Reuse the retained immutable published value only for
the same sealed resource/version, with dominating successful publication and
no intervening possibly aliasing write. Preserve the original late-read guard
order, requested shape checks, frontier retirement, observations and failure
prefix. Eliminating an allocation opportunity is allowed; deleting a required
check or publishing later failures early is not. No fusion or general liveness
engine is required for this bounded transformation.

The serial structured-only precursor was not selected: current orchestration is
already serial, and the forwarding case demonstrates actual optimization freedom
while testing the separate-derivation contract. Required falsifiers include an
intervening overlapping publication, untaken-branch publication, wrong late-read
shape, a distinct overlapping descriptor with insufficient capacity, retained
old values/observations and allowed OOM/failure prefixes. This next boundary has
not been implemented by the campaign.
