# MDSLC current state

Engineering checkpoint: canonical merge
`63d642d9e4eb4183acf58a4391e1c668e734b9f0`, [PR #29](https://github.com/onlyxItachi/MatcoreDSL/pull/29).
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
      -> ordered guards, real Linalg computation, observable commits
      -> source-paired mechanical verification; no execution authority
Matcore owns semantic legality, provenance and observable ordering.
MLIR owns structured transformation machinery; LLVM/backends own machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

Two adjacent dependent authenticated GEMMs now preserve their actual value
dependency, storage bindings and ordered failure/write frontiers. Adversarial
checks reject semantic/authority forgery and survive admitted upstream rewrites.
An isolated type-registration shim reconciles the prebuilt allocator's sanitizer
protocol without disabling semantic checks. Details: [region checkpoint](TWO_GEMM_REGION_V1.md),
[independent review](agent-reports/two-gemm-region-adversarial-v1.md).

## Unsupported or unproven

Region guards are retained obligations, not discharged runtime predicates.
No region execution, fusion or tiling; no generated runtime replacement, GPU/NPU,
expanded semantic tensor/view frontend or API/ABI freeze. Pointer range checks do not prove backing capacity or
lifetime. Recovered C++ recognition grants no execution authority. Native BLAS
parity ([#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)) remains partial;
tensor/view types ([#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20)) remain design-only.

## Exactly one next boundary

An inspection-only per-call guard/discharge ledger for this same region:
distinguish source-proven representation facts, still-required runtime predicates
and unproven caller obligations, using existing validation as the oracle. This
follows because the region preserves ordered guard groups but does not yet
mechanically classify their individual proof obligations.
