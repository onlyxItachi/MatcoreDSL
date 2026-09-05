# Two-GEMM RHS mirror: independent oracle/evidence review

Reviewed clean head `33423177da947bc40d43680a33a6501a31d14736`
against `8a825350ae8e3f24e55bc8d050375a0b2c0da3dd`.

Scope: CLI test source, frontend/MLIR reports, Release cache, generated CTest
registration and completed log. Read-only review: no edits, builds or reruns.

Verdict: PASS. No blocking oracle, defined-behavior or evidence defect found.

## Adversarial witnesses

Original sources compile through both existing `capture-v0` and
`matcore-mlir` execution pipelines. No derived region is compiled/executed.
These prove old-route behavior; separate paired MLIR tests prove inspection.

- Rectangular C[2x4], D[5x2], E[5x4] discriminates operand geometry.
- Independent arithmetic confirms C=[19,22;43,50], D*C=[105,122;43,50],
  whereas C*D differs.
- Late aliasing uses the same six live float objects for C[2x3] and D[3x2].
  The oracle indexes D after the first write. C=[54,60,66,126,141,156],
  E[0]=10476; no cast, lifetime violation or out-of-bounds view.
- Invalid second D columns retain every C result and every E=-9 sentinel.
  Backing storage is ample; shape mismatch is the intentional failure.
- Existing C*C admission retains the independent result
  [1307,1518;2967,3446].
- Intervening observation rejects admission while original execution must
  print exactly `rhs-observed:62`.

All arrays have positive compile-time extents and remain alive; all indices
are in bounds. Outputs are separate arrays and intentional input aliasing is
legal float-to-float access. Products and partial sums are small nonnegative
integers exactly representable in binary32 (largest final result 33642).
Exact comparisons are valid for these witnesses, not a general bitwise claim.

## Authenticated Release log

`build-rhs-release/Testing/Temporary/LastTest.log` is complete, from
2026-09-05 18:49 through 18:52 +03. It contains 74 records: 71 ordinary passes
and three required-regex passes, no failures or skips.

- Region MLIR: 426/426 checks.
- Native region: 44 extractions.
- Nonexecuting runtime/FP oracle: 219 checks, zero failures, no GEMM executed.
- CLI: 105 grouped steps, 32 original-route executables (16 cases, two pipelines).

Cache: this worktree, coherent Clang/LLVM/MLIR 21.1.8, native/bootstrap/MLIR and
vector readiness enabled, OpenBLAS disabled. Source-inaccessible registration
requires clean source at the exact reviewed head; its completed log confirms
that same commit's configure/build/run/semantic-capability PASS.
`git diff --check` passed; the worktree remained clean.

This is inspected existing-log evidence, not a reviewer rerun.
No Debug, sanitizer, hosted or provider-enabled outcome is inferred.
Unavailable-region mode exits before numerical witnesses; its pass does not
prove RHS execution on Windows or another unavailable platform.
No guard discharge, generated execution, fusion, rollback, zero-copy,
performance or target-support claim follows.
