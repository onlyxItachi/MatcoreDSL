# Strict fused-pair issuer: qualified checkpoint

Date: 2026-10-04. This is isolated compiler-issued CPU arithmetic, not a source
fusion/runtime-dispatch or performance claim.

## Identity and review

- Engineering merge: `f55b86e5d0fbeaa8d10d5857bc2bf2ce71168df4`,
  [PR #79](https://github.com/onlyxItachi/MatcoreDSL/pull/79).
- Prior canonical main: `3818671aadf714e6b1c76144c16190d8999ca83a`.
- Qualified premerge head: `31401bf6002e10d09213f2e613ab2f29734def4f`.
- Identical qualified/merged compiler tree:
  `8a5733af566b001d98629f233d485a848f5a7715`.
- Original implementation: `8ce0bcd349cf2339162ce423c82377d3c827d642`,
  integrated as `b0172d6`; isolated research `198ee591ab3379bc5ea7d42e3c15fb980f08373b`
  remains separate history, not wholesale product adoption.

Independent Astra review accepted the exact issuer and its nine-test CI gate;
the [durable review](https://github.com/onlyxItachi/MatcoreDSL/blob/6e25df7e85ae3419d9e688541065433ff3151151/docs/mdslc/agent-reports/publication-forwarding-independent-v1.md)
records the bounded argument and negative controls. Before merge, the reviewer
rechecked unchanged issuer bytes, the saved local XML and live exact-head hosted
checks, then recommended normal merge. Review is not a separate execution run.

## Executed qualification

Exact coherent Clang/LLVM/MLIR 21.1.8, Linux x64 Release, physical Ryzen AI 9 HX
370. Root fresh complete build passed at the qualified head. Root repeated the
nine focused tests: stage/LLVM corruption gates, normal and genuinely generated
ASan arithmetic, optimized-object checks, and all three undersized-input controls.
The [implementation record](strict-fused-pair-issuer-v1.md) preserves the exact
61/13,617-check scopes, artifact hashes and counted nonempty leaf invocations.

Full local regression, without source mutation between runs:

| Scope | Outcome |
| --- | --- |
| Ordinary suite excluding only the two separately run fresh package builds | 400/400 passed, no skips, 1033.51 s |
| Both fresh source/build-inaccessible package tests | 2/2 passed, no skips, 179.14 s |

Combined coverage is **402 distinct tests**, not 402 plus the repeated focused
tests. The separate package phase reserved both compiler slots to respect local
resource limits. Raw owner logs/XML are `issuer-candidate-focused.*`,
`issuer-candidate-full-main.*`, and `issuer-candidate-full-packages.*` under
`/home/hamza-usta/mdslc-work/region-optimization-v1/evidence/`.

All qualifying exact-head hosted jobs passed:

- [Native 37213462578](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37213462578):
  all eight Release/Debug/OpenBLAS/MLIR/schedule/ASan+UBSan/TSan lanes.
- [Windows 37213462603](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37213462603).
- [ARM64 37213462575](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37213462575):
  existing scalar and row-contiguous regressions, not ARM fused-leaf execution.
- [Hygiene 37213462576](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37213462576)
  and [legacy CI 37213462594](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37213462594).

Duplicate push runs also passed; they are not additional unique test coverage.

## Retained limits and next boundary

Only the pinned built-in strict lhs pair is issued. The complete LLVM digest is
a version-bound drift gate, not a universal equivalence prover. Caller checked
E/workspace must be private/disjoint; inputs may alias. Empty E must bypass the
leaf after original guards; K=0/N>0 must still evaluate the consumer, including
`0*Inf`. The actual optimized object may call trusted conforming `memset`.
Neither successful verification nor imported IR grants source authority.

Exactly one next engineering boundary: source-connected checked CPU execution
of this leaf while preserving both original logical guards, source/failure
frontiers and prior effects. GPU failures need a separate proof; the CPU leaf's
no-recoverable-error argument does not transfer to asynchronous device work.
No benchmark was run; any earlier timing protocol remains deferred under this
correctness-first campaign. Issues #15 and #20 remain open and unchanged.
