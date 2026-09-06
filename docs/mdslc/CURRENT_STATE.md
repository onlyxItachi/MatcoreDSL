# MDSLC current state

Engineering checkpoint: canonical merge
`f6a5d9718a8c704231ad37d9e315f7d2d6912a39`, [PR #37](https://github.com/onlyxItachi/MatcoreDSL/pull/37).
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
  -> no execution, physical storage strategy or public syntax
Matcore owns semantic legality, provenance and observable ordering.
MLIR owns structured transformations; LLVM/backends own machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

The same closed grammar now survives real host headers, preprocessing and
unrelated host IO/RAII. Source/lookup identity, immutable replay and macro/pragma
closure are checked without a second parser/evaluator. The mathematical and
value/effect contract is unchanged. Validation: 78/78 Release, 27/27 focused
ASan/UBSan and 19/19 hosted checks; independent adversarial review accepted.
See [host-context evidence](CLOSED_REGION_HOST_CONTEXT_V1.md) and
[semantic contract](CLOSED_REGION_ADMISSION_V1.md).

## Unsupported or unproven

The host proof is private Linux 21.1.8 inspection, not an installed frontend.
Header-defined mathematical helpers and arbitrary compiler contexts remain
unsupported. No public region syntax, tensor/view API, generated execution,
storage/materialization lowering, fusion, GPU/NPU or API/ABI freeze. Guards,
storage/provider adapters and FP-status/trap adaptation remain unproven.
Resource/descriptor inequality never proves noalias; recovered C++ grants no
execution authority. Existing mutating GEMM has not been reinterpreted as pure.
Native BLAS parity ([#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15))
remains partial; tensor/view types ([#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20))
remain design-only.

## Exactly one next boundary

**Design-only external-storage/provider-publication adapter contract review.**
Host admission is now demonstrated; old-value preservation across aliases,
failure-visible publication and provider/FP adaptation remain unsettled.
Resolve these counterexamples and obtain owner approval before choosing a
consequential contract or implementing a consumer. Execution remains forbidden.
