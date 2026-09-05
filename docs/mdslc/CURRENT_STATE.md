# MDSLC current state

Engineering checkpoint: canonical merge
`4e9e6b4d7d8eb4cfd2c409ea02713e6c4633116a`, [PR #33](https://github.com/onlyxItachi/MatcoreDSL/pull/33).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Ordinary valid C++ with explicit Matcore mathematical intent
  -> Clang/Sema authentication and sealed source evidence
  -> Matcore IR v1 per-call capture / internal Matcore MLIR semantics
      -> existing authenticated CPU runtime/provider execution
      -> inspection-only isolated structured / buffer / vector specimens
  -> bounded two-GEMM region inspection from sealed declaration bindings
      -> committed value may feed either second GEMM input, without commutation
      -> descriptor identity separate from tensor values and physical aliasing
      -> ordered guard ledger, late reads, Linalg computation, observable commits
      -> source-paired mechanical verification; no execution authority
Matcore owns semantic legality, provenance and observable ordering.
MLIR owns structured transformations; LLVM/backends own machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

The existing lhs dependency now has a verified RHS mirror: `C=A*B; E=D*C`.
Operand order, late physical-alias reads, host barriers and ordered failures
remain checked. Existing `E=C*C` keeps its lhs carry plus late rhs read.
Details and validation: [RHS checkpoint](TWO_GEMM_RHS_V1.md),
[architecture review](agent-reports/two-gemm-rhs-architecture-review-v1.md).

## Unsupported or unproven

Guard predicates remain required, not discharged. No connected-region storage
lowering or execution, fusion, general DAGs, tiling, generated runtime replacement,
GPU/NPU, expanded tensor/view frontend or API/ABI freeze. Pointer ranges do not
prove storage capacity/lifetime; descriptor inequality does not prove noalias.
Recovered C++ recognition grants no execution authority.
Native BLAS parity ([#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15))
remains partial; tensor/view types ([#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20))
remain design-only.

## Exactly one next boundary

**Connected region storage/commit handoff design, before its implementation.**
The isolated buffer proof does not settle borrowed versus snapshot values,
cross-call aliases, destination identity, allocation ownership or failure-visible
writes. Boundary interfaces versus explicit materialization are consequential
architectural alternatives: obtain owner approval before implementing a consumer.
