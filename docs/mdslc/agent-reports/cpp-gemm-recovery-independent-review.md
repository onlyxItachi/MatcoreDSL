# Ordinary C++ GEMM recovery independent review

Date: 2026-08-11

## Verdict

Accepted for the Milestone D inspection-only boundary. No unresolved high- or
medium-severity finding remains.

This review covers the original recognizer in `a0fda2b`, the three consolidated
fail-closed fixes in `a095256`, the corrected diagnostic record in `b38a7a2`,
the redeclaration-chain fix in `20be3f3`, and its evidence update in `82b4647`.
It does not authorize recovered source replacement, Matcore IR/MLIR raising,
planner selection, or runtime execution.

## Adversarial findings and resolution

The prior high-severity findings were independently rechecked against the
Clang 21 AST and implementation:

1. Local declarations now require automatic storage duration, no TLS, and no
   explicit storage class. Cleanup and otherwise unsupported declaration,
   function, and parameter attributes reject. Non-default address spaces are
   checked recursively before canonical qualifier stripping.
2. The proof records the effective Clang FP evaluation method and accepts only
   source-type evaluation. Command-line and lexical double/extended evaluation,
   unavailable lexical scope ownership, non-IEEE denormals, forbidden
   reassociation/contraction, dynamic rounding, observable exceptions, and
   approximate/fast-math semantics fail closed.
3. OpenACC, SYCL host/device compilation, OpenMP executable directives, and
   OpenMP declare-simd/declare-target attributes reject recovery.

The review found one residual high-severity variant of item 3: Clang 21 does
not inherit `OMPDeclareSimdDeclAttr` from an annotated prototype onto the later
defining `FunctionDecl`. Inspecting only the definition therefore missed the
parallel/offload contract. Commit `20be3f3` resolves this by authenticating
attributes and parameter contracts across the complete function
redeclaration chain. The exact prototype-then-definition probe now emits
`recognized_rejected`, reason `offload_or_parallel_context`, an empty Matcore
capture module, and `rewrite=preserve_original_cpp`.

No other material issue was confirmed after that fix.

## Fail-closed and artifact review

- Recovery remains native-only and explicitly opt-in through
  `--inspect-recovered-gemm`.
- The default frontend path does not recognize or rewrite an ordinary loop.
- Inspection cannot be combined with host rewrite, sites, stubs, backend, or
  the semantic MLIR execution pipeline.
- Recognized-but-illegal and merely plausible loops preserve ordinary C++ and
  emit no recovered Matcore operation or runtime symbol.
- A not-recognized candidate has an empty semantic-contract field; the report
  no longer labels a structural near miss as authenticated GEMM semantics.
- Source bytes, SHA-256, SourceManager token ranges, compilation identity, and
  stable site identity remain inspection evidence rather than permission.
- Output-producing and opaque compiler arguments are rejected before
  publication. A joined `-MF<report-path>` alias probe left neither report nor
  IR, and direct `-fopenmp` was rejected as an unsafe mode argument.
- Duplicate standard-output ownership and output/input path aliases remain
  rejected.

## Reproduced validation

Toolchain: Ubuntu Clang/LLVM 21.1.8 (`x86_64-pc-linux-gnu`). Both build trees
linked the coherent Clang 21 package. Compilation used Ninja `-j2`; tests were
serialized.

Debug:

```text
cmake --build /home/hamza-usta/.cache/mdslc-cpp-recovery-build \
  --target matcore-extract -- -j2
ctest --test-dir /home/hamza-usta/.cache/mdslc-cpp-recovery-build \
  -R '^(frontend\.native\.focused|frontend\.native\.primary|frontend\.native\.recovered_gemm_inspection)$' \
  --output-on-failure -j1

3/3 passed
Recovered GEMM inspection: 344 checks, 0 failures
```

Release:

```text
cmake --build /home/hamza-usta/.cache/mdslc-cpp-recovery-release \
  --target matcore-extract -- -j2
ctest --test-dir /home/hamza-usta/.cache/mdslc-cpp-recovery-release \
  -R '^(frontend\.native\.focused|frontend\.native\.primary|frontend\.native\.recovered_gemm_inspection)$' \
  --output-on-failure -j1

3/3 passed
Recovered GEMM inspection: 344 checks, 0 failures
```

Additional independent probes exercised:

- a lexical `#pragma clang fp eval_method(double)` scope, which reported
  `fp.evaluation_method=double` and rejected with
  `fp_evaluation_method_mismatch`;
- SYCL host-only and device-only modes, both rejected as
  `offload_or_parallel_context`;
- the non-inherited OpenMP declare-simd prototype case fixed in `20be3f3`;
- direct OpenMP mode injection and joined dependency-output alias injection,
  both rejected before any report or IR publication.

`git diff --check` passed for the complete recognizer, focused tests, contract,
and report diff through `82b4647`.

## Review boundary

This acceptance is deliberately narrow. The lane proves conservative typed
recognition and inspectable rejection, not explicit/recovered semantic
equivalence or permission to execute an optimized replacement. Runtime guards,
exact fallback preservation, physical floating-point-environment checks, the
Matcore MLIR adapter, whole-repository tests, packaging, sanitizers, and Windows
regression remain integration gates in their owning milestones.
