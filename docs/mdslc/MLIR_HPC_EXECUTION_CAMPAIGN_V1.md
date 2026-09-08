# MLIR HPC execution campaign

Status: **active investigation; no new execution or performance claim**.

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
