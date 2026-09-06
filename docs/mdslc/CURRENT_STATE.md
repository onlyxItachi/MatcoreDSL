# MDSLC current state

Engineering checkpoint: canonical merge
`e3defc9b699b33cc9f5d539f32f55044fef87ff8`, [PR #51](https://github.com/onlyxItachi/MatcoreDSL/pull/51).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Existing explicit C++ GEMM -> Clang/Sema authentication
  -> Matcore IR v1 / internal Matcore MLIR semantics
  -> unchanged authenticated CPU runtime/provider execution
Existing structured/buffer/vector and ordered two-GEMM paths: inspection only
Private closed regions in real host C++ -> authenticated Clang/Sema AST
  -> frozen compiler/header/preprocessing context and historical replay
  -> frontend-neutral immutable values + separate all-MAY-alias resources
  -> symbolic shapes, checked math, publication/observation/failure ordering
  -> private registered MLIR + structural and sealed-source paired verification
  -> authenticated compiled C++ orchestration + checked host adapter:
  immutable snapshots + all-MAY-alias dense resource views
  -> strict native/generated math, ordered host publication and owning observation
  -> sticky failure prefix + full per-candidate FP environment restoration
Private strict GEMM -> verified Linalg/buffer path -> LLVM -> tested x64 object
Separate trusted registry -> native/generated strict or legal legacy/provider
  -> isolated candidate output, verified completion, then immutable value issuance
Shared provider adapter serializes Matcore-owned OpenBLAS policy lifetimes.
Matcore owns legality/effects; MLIR transformations; LLVM/backends machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
Experimental named C++ region admission + opaque owning Result: now proven.
```

## Material change

Named ordinary C++ functions can now target the closed semantic layer through
experimental canonical declarations. Hidden member attributes, redeclarations,
ownership overrides and forged completion are rejected. Opaque owning results
preserve observations across return/destruction and mixed host STL configurations.
Implementation-head Release passed 111/111, Debug 109/109, ASan/UBSan 58/58;
all 19 final-head hosted checks passed. The headers deliberately remain uninstalled.
See [frontend contract](EXPERIMENTAL_REGION_FRONTEND_V1.md) and
[independent/integration evidence](agent-reports/experimental-region-integration-v1.md#canonical-merge-checkpoint).

## Unsupported or unproven

Closed source is a private Linux 21.1.8 compiled native/generated consumer, not an
installed frontend or a whole-host-TU replacement. The generated leaf accepts
no user IR and grants no source authority. Experimental named-region syntax is
not yet a shipped execution interface. No general
tensor/view API, fusion, GPU/NPU or API/ABI stability claim. The new adapter
requires valid exclusive host storage and a conforming trusted allocator; it
does not cover arbitrary interposed host effects, exports or device transfers.
Uncoordinated external OpenBLAS calls or duplicate adapter instances are not protected.
Provider admission is bounded evidence, not proof for every provider/version/core.
Snapshots are conservative realization, not a zero-copy or performance claim.
Resource/descriptor inequality never proves noalias; recovered C++ grants no
execution authority. Existing mutating GEMM has not been reinterpreted as pure.
Native BLAS parity ([#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15))
remains partial; tensor/view types ([#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20))
remain design-only.

## Exactly one next boundary

**Make the private compiler/runtime Value boundary opaque and revision-checked.**
Public owning results are now defended, but emitted orchestration must also avoid
depending on private STL layouts or linking silently against a different Session
layout. This closes a concrete packaging prerequisite without freezing an ABI or
changing mathematical/effect semantics.
Independent research branches are not canonical capabilities until integrated.
