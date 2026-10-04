# GPU pair source: hosted Release qualification budget

The Release job budget changes from 45 to 60 minutes, without changing any
matrix entry, build flag, test selection/count, command or production proof.
Compiler/runtime code and AGENTS are unchanged. This is CI reliability work,
not a performance result or permission to count incomplete jobs as passing.

At source checkpoint `9f7f9af80d96c8f38d4f66097a5c8779fb185464`, push
[run37225347750](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37225347750)
retains two deadline cancellations. Both check-run annotations explicitly state
that the job exceeded 45 minutes; neither was manually canceled.

- Required OpenBLAS/MLIR job111503741539: printed 267 registered tests, zero
  failed and 14 preexisting AVX512 skips (2170.31 seconds), then canceled at
  19:25:59 UTC. Its entire job did not pass. The production-without-tests step
  is conditionally excluded for OpenBLAS ON, so this is not a missing execution
  of that particular proof.
- Disabled OpenBLAS/MLIR job111503741646: printed 264 registered tests, zero
  failed and 14 preexisting AVX512 skips (2148.50 seconds). Its subsequent
  production-without-tests proof built 140/140 targets and performed several
  installed checks, but did not complete before the 19:29:24 UTC deadline.
- The corresponding required-OpenBLAS PR job111503750841 did finish successfully
  in 44m47s: only 13 seconds of headroom remained. Its success does not convert
  the canceled push jobs into passes.

The integration owner and independent adversarial reviewer checked annotations,
logs and the unchanged scope before accepting this timeout-only adjustment.
Fresh exact-head hosted qualification remains required. Completed local571/571
evidence stays attributed to clean9f, not relabeled as execution at a later SHA.
After this adjustment only compiler/AGENTS identity is unchanged; the workflow
identity differs from9f by the documented Release budget/comment.

External evidence under
`/home/hamza-usta/mdslc-work/region-optimization-v1/evidence`:

| File | SHA256 |
|---|---|
| pr86-9f7-push-release-required-timeout.log | `699e3d9d38d6ac3bd07ee2ca0322814cb8567e184e53e7db84d16e966453f872` |
| pr86-9f7-push-release-disabled-timeout.log | `f3daa375ebd8bd5da0efad6e613dc7296bffd4f2216c1739b883ed59178b6394` |

The matching `*-annotations.json` files retain the explicit timeout diagnostics.
No test, sanitizer, package, physical-device or numerical acceptance rule was
relaxed. Existing Debug already has its separately justified 60-minute budget.
