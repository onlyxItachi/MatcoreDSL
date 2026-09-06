# MDSLC current state

Engineering checkpoint: canonical merge
`38e0c9368c65f399b0d8d6c9c39469e2836007c9`, [PR #42](https://github.com/onlyxItachi/MatcoreDSL/pull/42).
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
  -> strict native math, ordered host publication and owning observation
  -> sticky failure prefix + full per-candidate FP environment restoration
Private strict GEMM -> verified Linalg/buffer path -> LLVM -> tested x64 object
Shared provider adapter serializes Matcore-owned OpenBLAS policy lifetimes.
Matcore owns semantic legality, provenance and observable ordering.
MLIR owns structured transformations; LLVM/backends own machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

Authenticated real-host source now produces compiled orchestration of the strict
native adapter, preserving symbolic checks, immutable values, ordered publication,
observations and failure prefixes. No runtime AST interpreter is introduced.
Clean Release 91/91, affected ASan/UBSan 40/40, independent adversarial review and
19/19 hosted checks passed. See [source execution evidence](CLOSED_SOURCE_EXECUTION_V1.md).
The [generated mathematical leaf](agent-reports/generated-strict-cpu-candidate-v1.md)
is still independent of that source consumer; the
[shared provider-policy correction](agent-reports/openblas-shared-policy-scope-v1.md)
remains in force.

## Unsupported or unproven

Closed source is a private Linux 21.1.8 compiled native-adapter consumer, not an
installed frontend or connected source-to-generated math path. The generated leaf accepts
no user IR and grants no source authority. No public region syntax, general
tensor/view API, fusion, GPU/NPU or API/ABI stability claim. The new adapter
requires valid exclusive host storage and a conforming trusted allocator; it
does not cover arbitrary interposed host effects, exports or device transfers.
Uncoordinated external OpenBLAS calls or duplicate adapter instances are not protected.
Snapshots are conservative realization, not a zero-copy or performance claim.
Resource/descriptor inequality never proves noalias; recovered C++ grants no
execution authority. Existing mutating GEMM has not been reinterpreted as pure.
Native BLAS parity ([#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15))
remains partial; tensor/view types ([#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20))
remain design-only.

## Exactly one next boundary

**Connect authenticated closed regions to the generated CPU candidate through
the checked host adapter and trusted candidate registry.** Both the host
storage/failure boundary and generated leaf are now defended independently;
source orchestration and candidate selection must compose before claiming the
connected source-to-generated path.
Independent research branches are not canonical capabilities until integrated.
