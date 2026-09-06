# MDSLC current state

Engineering checkpoint: canonical merge
`322af8d43799304cc289748a5b0c803cf3a0f697`, [PR #48](https://github.com/onlyxItachi/MatcoreDSL/pull/48).
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
  -> authenticated compiled C++ orchestration (private Linux x64 consumer)
  -> checked host adapter, now connected and executed:
  immutable snapshots + all-MAY-alias dense resource views
  -> strict native/generated math, ordered host publication and owning observation
  -> sticky failure prefix + full per-candidate FP environment restoration
Private strict GEMM -> verified Linalg/buffer path -> LLVM -> tested x64 object
Separate trusted registry -> native/generated strict or legal legacy/provider
  -> isolated candidate output, verified completion, then immutable value issuance
Shared provider adapter serializes Matcore-owned OpenBLAS policy lifetimes.
Matcore owns legality/effects; MLIR transformations; LLVM/backends machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

The same authenticated private source programs now execute through both strict
native and actual MLIR-generated mathematical implementations. Rectangular
dependencies, late aliased reads, ordered observations, failure prefixes and
actual candidate identity are checked end to end, including instrumentation of
the generated object. This connects existing boundaries; it adds no execution
authority to editable IR. Exact-head Release passed 102/102, ASan/UBSan 51/51,
and all 19 hosted checks were green. See the
[composition evidence and canonical record](agent-reports/closed-source-generated-connection-v1.md#canonical-merge-checkpoint).

## Unsupported or unproven

Closed source is a private Linux 21.1.8 compiled native/generated consumer, not an
installed frontend or a whole-host-TU replacement. The generated leaf accepts
no user IR and grants no source authority. No public region syntax, general
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

**Establish the experimental named C++ region frontend and owning result contract.**
The private semantic/execution composition now works. A useful product needs
ordinary named-function admission and durable host-visible results without hidden
C++ effects or private standard-library layout leaking across compiler/runtime
boundaries. This is still an experimental contract, not an API/ABI freeze.
Independent research branches are not canonical capabilities until integrated.
