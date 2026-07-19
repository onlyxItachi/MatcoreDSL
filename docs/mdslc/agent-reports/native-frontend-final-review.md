# Final adversarial review: native LibTooling frontend v1

Date: 2026-07-19

## Review target and independence

- Integration branch: `mdslc/native-libtooling-v1`.
- Milestone base: `3e3fa5b2d1990e1c37870f8b2096fbda6128716b`.
- Final reviewed implementation: `f71f1800a1ba70f2b363ff68ecd6632c7ae8fad1`.
- Review lane: `/root/final_postfix_review`, with earlier independent passes by
  `/root/final_quality_check` and `/root/final_acceptance_check`.
- The review lanes did not implement production fixes. Confirmed findings were
  reproduced, handed back to separate implementation lanes, and re-reviewed
  after focused commits.

The review attempted to reject trusted-header authentication, canonical
declaration and annotation checks, source ranges, extraction/compile context
identity, input races, generated-file races, dependency publication, frontend
selection, installed relocation, symbol identity, and the existing CPU artifact
pipeline.

## Final verdict

Accept the native frontend v1 for the declared standalone CPU scope. The final
re-review found **no unresolved high- or medium-severity defect**.

This verdict does not cover CUDA, BLAS lowering, MLIR lowering, GEMV, GEVM,
ReLU-GEMM, NPU dispatch, capability planning, or broader production readiness.

## Findings and resolutions

The independent review found real time-of-check/time-of-use and depfile issues.
Each was reproduced before correction and gained a focused regression:

1. `d461407` publishes complete dependency closures. It adds `runtime_c.h` and
   system-include dependencies to generated-file closure checks rather than
   allowing `-MMD` to hide a mutated trusted/system header.
2. `5bcb7e2` preserves dependency resolution identity. It detects symlink
   retargeting, preserves lexical include paths containing `..`, and rejects a
   newly introduced higher-priority header instead of accepting an unchanged
   byte snapshot from a different file.
3. `aa41c22` freezes generated compilation inputs. It rechecks host dependency
   resolution after the host compile, snapshots the generated host and VFS
   overlay, and emits valid Make depfiles for colon-bearing paths.
4. `f71f180` rechecks generated include resolution. It compares stub and
   backend dependency resolution after their compilation and again before
   final publication, removing the component/final artifact on mismatch.

The last finding was reproduced by creating a higher-priority `stdexcept`
after the generated-stub baseline scan. At the fixed head the driver returned
nonzero, diagnosed a resolution change between the baseline and post-compile
stub phase, and left neither the requested object nor depfile. The symmetric
backend and final checks are present and tested.

## Final independent evidence

At `f71f180` the final reviewer:

- rebuilt the driver successfully;
- reran the native driver adversarial suite: 72 checks passed;
- reran the complete standalone suite: 8/8 CTest tests passed in 61.35 seconds;
- reproduced the fixed generated-stub race under
  `/tmp/mdslc-generated-header-race-fixed-review.kw7jwjyr`;
- confirmed mismatch paths remove the affected object and depfile;
- ran `git diff --check aa41c22..f71f180` successfully.

The final complete suite covers the focused native frontend, primary frontend
contract, native parity/adversarial contract, driver pipeline, frontend
selection, integration matrix, installed consumer, and CPU runtime.

## Reviewed boundaries

No unresolved high or medium issue was found in:

- direct resolved trusted-header identity and semantic ABI authentication;
- fake, copied, shadowed, or macro-mutated public headers;
- direct-callee, canonical declaration, exact annotation, overload, alias, and
  unqualified-call handling;
- macro/template/lambda/header/constexpr/unevaluated-context rejection;
- SourceManager/Lexer ranges and line mapping;
- extraction and host-compile flag parity;
- source, included-header, generated-host, VFS-overlay, sites, stubs, and
  backend mutation detection;
- public depfile escaping and complete dependency publication;
- native-default/bootstrap-explicit selection;
- relocatable install discovery and absence of embedded development paths;
- stable multi-translation-unit generated symbols;
- verifier/codegen/runtime reuse and the ordinary CPU object/link/run path;
- Python, nanobind, MLIR, or legacy JIT leakage into the default path.

## Residual limitations

The driver intentionally rejects user VFS overlays, PCH/module injection,
opaque response/linker forms, and unsupported artifact modes. Generated
publication is atomic per file rather than as one directory transaction. The
implemented operation remains synchronous host-resident rank-2 row-major f32
CPU GEMM with explicit error fallback. These are declared scope limits, not
unresolved review findings.
