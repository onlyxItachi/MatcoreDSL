# MDSLC current state

Engineering checkpoint: canonical merge
`5df9ac3c451539bdd9c1e577d6558802fca30535`, [PR #40](https://github.com/onlyxItachi/MatcoreDSL/pull/40).
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
  -> still no connected execution or public syntax on canonical main
Independent private Linux x64 host adapter, now executed and tested:
  immutable snapshots + all-MAY-alias dense resource views
  -> strict native math, ordered host publication and owning observation
  -> sticky failure prefix + full per-candidate FP environment restoration
Private strict GEMM -> verified Linalg/buffer path -> LLVM -> tested x64 object
Matcore owns semantic legality, provenance and observable ordering.
MLIR owns structured transformations; LLVM/backends own machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

The compiler can now issue a fixed strict GEMM implementation through upstream
MLIR/LLVM and execute its object on Linux x64. Structural checks, independent
numerical oracles and a real generated-load ASan negative control defend this
leaf. Validation: 90/90 Release and 19/19 hosted checks, including the 39-test
sanitizer scope; independent review ACCEPT. See the
[implementation evidence](agent-reports/generated-strict-cpu-candidate-v1.md),
[independent review](agent-reports/generated-strict-cpu-independent-review-v1.md)
and [host contract](FOUNDATION_RESOURCE_DECISION_V1.md).

## Unsupported or unproven

Closed source is still a private Linux 21.1.8 admission proof, not an installed
frontend or connected source-to-generated execution. The generated leaf accepts
no user IR and grants no source authority. No public region syntax, general
tensor/view API, fusion, GPU/NPU or API/ABI stability claim. The new adapter
requires valid exclusive host storage and a conforming trusted allocator; it
does not cover arbitrary interposed host effects, exports or device transfers.
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
