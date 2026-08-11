# Recovered GEMM to Matcore MLIR equivalence evidence

- Lane: Milestone D authenticated recovered-source semantic bridge
- Implementation commit: `8039bca`
- Scope: internal analysis/inspection only; no source rewrite, execution, or
  public/installable API
- Toolchain: Clang/LLVM/MLIR `21.1.8`

## Result

The native LibTooling frontend's `recognized_guard_required` canonical C++
GEMM candidate now enters the same `mdsl.gemm` operation class as an explicit
Matcore IR v1 capture without forging a Matcore IR v1 operation or the trusted
`matcore::mdsl::gemm` callee.

The bridge consumes the exact `frontend::Result`, extraction `Options`, and
candidate index together. Before constructing MLIR it recomputes or validates:

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

The recovered module has its own closed analysis envelope. The existing strict
Matcore IR v1 CPU runtime-dispatch lowering rejects it and clears any pending
records. No recovered C++ is rewritten or executed by this implementation.

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

The real relaxed native-frontend loop and the reviewed dynamic explicit v1
fixture produce byte-equal normalized contracts and equal SHA-256 identities.
A valid static-shape explicit operation produces a different fingerprint.
Strict/default C++ candidates are rejected before MLIR construction and cannot
participate in authorized equivalence.

The fingerprint is an inspection/equality mechanism, not execution permission.

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

- recovered bridge: `62/62` checks passed;
- existing Matcore MLIR core: `204` checks, `0` failures;
- existing CPU runtime-dispatch lowering: `18` checks, `0` failures;
- focused CTest: `3/3` passed:
  `mlir.semantic.core`, `mlir.semantic.recovered-gemm`, and
  `mlir.cpu.runtime_dispatch_lowering_v1`;
- `git diff --check` passed for the lane.

The focused test runs the real native LibTooling frontend against canonical,
strict, and not-recognized ordinary-C++ fixtures. It adversarially mutates the
compilation identity, source bytes/digest, site ID, line, outer range, proof
order, guard order, FP proof, semantic contract, and rejection set. Every
mutation fails closed before MLIR construction.

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
