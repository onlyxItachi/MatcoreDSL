# MDSLC current state

Engineering checkpoint: canonical merge
`8a825350ae8e3f24e55bc8d050375a0b2c0da3dd`, [PR #31](https://github.com/onlyxItachi/MatcoreDSL/pull/31).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Ordinary valid C++ with explicit Matcore mathematical intent
  -> Clang/Sema authentication and sealed source evidence
  -> Matcore IR v1 per-call capture / internal Matcore MLIR semantics
      -> existing authenticated CPU runtime/provider execution
      -> inspection-only structured / buffer / vector specimens
  -> bounded two-GEMM region inspection from sealed declaration bindings
      -> descriptor identity separate from tensor values and physical aliasing
      -> source-bound obligation ledger, ordered Linalg computation/commits
      -> source-paired mechanical verification; no execution authority
Matcore owns semantic legality, provenance and observable ordering.
MLIR owns structured transformation machinery; LLVM/backends own machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

Each call now distinguishes source representation, pending runtime predicates,
unproven caller storage preconditions and retained dispatch/completion duties.
Source-paired checks reject false discharge and cross-call ledger reuse. Malformed
integer metadata fails closed. Release and Debug: 74/74 each; ASan/UBSan: 24/24;
hosted checks: 19/19. Details: [ledger checkpoint](REGION_GUARD_LEDGER_V1.md),
[independent review](agent-reports/region-guard-adversarial-v1.md).

## Unsupported or unproven

Region guards remain obligations, not executed or discharged runtime predicates.
No region execution, fusion or tiling; no generated runtime replacement, GPU/NPU,
expanded semantic tensor/view frontend or API/ABI freeze. A valid byte range or
successful plan proves neither backing capacity/lifetime nor full execution
legality. Provider failure may follow writes. Recovered C++ recognition grants
no execution authority. Native BLAS
parity ([#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)) remains partial;
tensor/view types ([#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20)) remain design-only.

## Exactly one next boundary

Mirrored RHS-only region admission: `C=A*B; E=D*C`. Existing sealed bindings and
role-specific obligations can support this symmetry without changing execution.
Preserve existing `E=C*C` as forwarded lhs plus late rhs read; prove operand
order, asymmetric shapes, late aliases and failure ordering. No new dual carried
edges, general DAGs or deeper lowering follows automatically.
