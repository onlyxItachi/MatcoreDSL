# Recovered GEMM to Matcore MLIR equivalence evidence

- Lane: Milestone D authenticated recovered-source semantic bridge
- Initial implementation commits: `8039bca`, `d9756da`
- Independent-review hardening commit: `230e142`
- Scope: internal analysis/inspection only; no source rewrite, execution, or
  public/installable API
- Toolchain: Clang/LLVM/MLIR `21.1.8`

## Result

The native LibTooling frontend's `recognized_guard_required` canonical C++
GEMM candidate now enters the same `mdsl.gemm` operation class as an explicit
Matcore IR v1 capture without forging a Matcore IR v1 operation or the trusted
`matcore::mdsl::gemm` callee.

The bridge no longer accepts mutable `frontend::Result` and `Options` values.
After successful native parse/Sema, source-stability checking, diagnostics, and
Matcore IR v0 verification, the native frontend issues a non-default-
constructible `AuthenticatedNativeFrontendEvidenceV1`. Its private payload
holds immutable-by-API copies of the complete `Result` and effective `Options`.
Only the native issuer and internal MLIR bridge accessor are friends. Mutating
the published diagnostic `Result` or original `Options` after issuance cannot
alter the sealed evidence.

Before constructing recovered MLIR the bridge recomputes or validates from
that sealed payload:

- native `clang-libtooling-v1` producer identity;
- normalized source identity from the extraction input;
- stable compilation identity from the exact compiler arguments;
- SHA-256 of the parsed `source_snapshot` bytes;
- stable site ID from source identity, compilation identity, source bytes,
  loop offset, and `recovered.cpp.gemm.v1` kind;
- byte-range bounds and line/column at the outer-loop offset, including CRLF;
- the exact ordered 18-role source proof range set;
- the exact ordered seven runtime guard obligations;
- the closed canonical GEMM pattern and typed overwrite contract;
- the complete relaxed effective-C++ floating-point proof; and
- nonempty output/lhs/rhs/M/N/K/function bindings.

Only a zero-rejection `recognized_guard_required` candidate is accepted.
`not_recognized`, `recognized_rejected`, and synthetic `raised` states do not
acquire permission through this API.

The authenticated explicit/recovered comparison likewise accepts only sealed
native evidence. Its explicit side authenticates the selected native v0 site
against sealed source bytes, source ranges, line/column, compilation identity,
and stable site ID, then internally performs v0 to verified v1 to Matcore MLIR.
It returns only an equality result, normalized fingerprints, and an error; it
does not expose an authenticated MLIR wrapper that could be reused as
execution permission.

## Preserved semantic boundary

The recovered operation records:

- `origin.kind = recovered_cpp_loop`;
- `origin.permission = source_proven_guard_required`;
- no `canonical_callee` field;
- target `generic` and fallback `preserve_original_cpp`;
- the exact recovered numerical profile derived from effective C++ semantics;
- dynamic M/K/N tensor relationships, row-major strides, F32 accumulation,
  mutability, required alignment, alias preconditions, effects, and synchronous
  overwrite behavior;
- exact source digest, compilation identity, source/proof ranges, and physical
  source location; and
- module-level source identity, function identity, analysis-only marker, and
  ordered runtime guards.

Explicit capture and recovered analysis use one internal GEMM semantic-site
constructor for types, tensor contracts, destination/result tying, effects,
aliases, requirements, policy, and function shape. The source-specific origin,
numerical derivation, and provenance dictionaries remain separate and are
validated by the dialect's closed cross-product.

The recovered module has its own closed analysis envelope. Executable CPU
runtime-dispatch lowering now rejects any module carrying
`mdsl.analysis_only`, clears pending records transactionally, and additionally
requires exact `mdsl.producer = clang-libtooling-v1`. Thus neither recovered
analysis IR nor a structurally valid bootstrap-produced explicit envelope can
authorize execution. No recovered C++ is rewritten or executed by this
implementation.

## Mathematical equivalence

The normalized `matcore-mathematical-gemm-v1` fingerprint removes only:

- source/site/private-symbol identity and provenance;
- explicit-call versus recovered-loop origin;
- source-expression and dynamic-symbol spelling;
- target/fallback policy; and
- numerical profile and derivation labels.

It retains tensor and result types, static/dynamic shape relationships,
strides, layout, memory space, alignment, mutability, destination/result
identity, accumulation type, semantic requirements, alias preconditions,
effects, synchronization, and every expanded numerical field.

One test invokes the real native LibTooling frontend for both the trusted
explicit call and relaxed ordinary-C++ loop. The explicit result is upgraded
through v0/v1/MLIR internally, and both sides produce byte-equal normalized
contracts and equal SHA-256 identities. A valid static-shape explicit operation
produces a different structural fingerprint. Strict/default C++ candidates are
rejected before recovered MLIR construction and cannot participate in
authenticated equivalence.

The public structural fingerprint/equality functions are explicitly named
`fingerprintStructuralMathematicalGemmV1` and
`equivalentStructuralMathematicalGemmV1`. They accept structurally verified
modules for diagnostics and make no provenance claim. Authenticated equality
is a separate evidence-token API. Neither mechanism is execution permission.

## Validation

Configured the exact opt-in build with:

```text
cmake -S compiler \
  -B /home/hamza-usta/.cache/mdslc-semantic-mlir-build \
  -G Ninja \
  -DBUILD_TESTING=ON \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_MATCORE_MLIR=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DMLIR_DIR=/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
```

Built gently and serially with other compiler lanes:

```text
nice -n 10 cmake --build \
  /home/hamza-usta/.cache/mdslc-semantic-mlir-build \
  --target matcore_mlir_recovered_gemm_tests \
           matcore_mlir_semantics_tests \
           matcore_mlir_cpu_runtime_lowering_tests \
  -j2
```

Observed results:

- recovered bridge: `78/78` checks passed;
- existing Matcore MLIR core: `204` checks, `0` failures;
- existing CPU runtime-dispatch lowering: `18` checks, `0` failures;
- focused CTest: `3/3` passed:
  `mlir.semantic.core`, `mlir.semantic.recovered-gemm`, and
  `mlir.cpu.runtime_dispatch_lowering_v1`;
- `git diff --check` passed for the lane.

The focused test runs the real native LibTooling frontend against a trusted
explicit call plus canonical relaxed, strict, and not-recognized ordinary-C++
fixtures. It proves live explicit v0 to v1 to MLIR and recovered-loop
mathematical equivalence. It then mutates the public post-extraction result with
an in-bounds proof shift, coordinated outer-end/proof drift, parameter binding
drift, function/source-display drift, state/FP/guard drift, source bytes, and a
diagnostic. Every mutation leaves the sealed recovered MLIR byte-identical.
Changing the original effective options after issuance likewise has no effect.

The strict candidate is adversarially relabeled by copying every relaxed
candidate literal into its mutable public result; its sealed evidence retains
the original strict FP rejection and cannot construct recovered IR or enter
authenticated equivalence. The test also verifies that `mdsl.analysis_only`
and bootstrap-producer explicit envelopes fail executable CPU lowering while
clearing pending records.

## Independent-review findings addressed

The first independent review rejected the initial implementation with three
medium findings and no high findings:

1. mutable frontend diagnostics were treated as an authentication boundary;
2. structurally verified fingerprints were described as authenticated; and
3. the analysis-only marker was not an enforced lowering taint.

A separate execution-boundary review also found that the CPU lowerer accepted
the bootstrap producer supported by the inspection bridge verifier. Commit
`230e142` addresses all four findings with sealed native evidence, separate
structural/authenticated equality APIs, a live explicit frontend proof, and
hard analysis-only/native-producer lowering gates. Focused tests pass; final
independent rereview remains required before integration acceptance.

## Limitations

- This is an internal Linux validation lane. The recovered analysis target is
  not built on Windows in this commit; ordinary explicit Windows support is
  unchanged.
- The bridge does not establish or execute the recorded runtime guards.
- It emits no persistent recovered JSON schema and exposes no CLI mode.
- It recognizes only the already-reviewed canonical row-major F32 GEMM loop.
- Mathematical equivalence does not imply rewrite profitability, guard
  dominance, or backend legality.

Lane verdict: implementation and focused evidence pass; independent
adversarial review remains the integration acceptance gate.
