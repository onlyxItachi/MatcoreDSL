# MDSLC current state

Engineering checkpoint: canonical merge
`e53f2c6302430f574473ba1a644cc42f1169384a`, [PR #35](https://github.com/onlyxItachi/MatcoreDSL/pull/35).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Existing explicit C++ GEMM -> Clang/Sema authentication
  -> Matcore IR v1 / internal Matcore MLIR semantics
  -> unchanged authenticated CPU runtime/provider execution
Existing structured/buffer/vector and ordered two-GEMM paths: inspection only
Private closed-region feasibility instrument -> Clang/Sema admitted AST
  -> frontend-neutral immutable values + separate all-MAY-alias resources
  -> symbolic shapes, checked math, publication/observation/failure ordering
  -> private registered MLIR + structural and sealed-source paired verification
  -> no execution, physical storage strategy or public syntax
Matcore owns semantic legality, provenance and observable ordering.
MLIR owns structured transformations; LLVM/backends own machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

Closed C++ admission passed both bounded closure and useful-expression tests:
lhs/rhs GEMMs, rectangular and symbolic shapes, reused Clang-instantiated helpers,
shape branches, old values, late reads and ordered effects. No shadow evaluator
or proxy-template frontend was introduced. Validation: 77/77 Release, 26/26
focused ASan/UBSan and 19/19 hosted checks; independent adversarial review accepted.
See [contract and evidence](CLOSED_REGION_ADMISSION_V1.md).

## Unsupported or unproven

The new proof uses a hermetic source fixture, not a real application's header
context or installed frontend. No public region syntax, tensor/view API, generated
region execution, storage/materialization lowering, fusion, GPU/NPU or API/ABI
freeze. Guards, storage adapters and FP-status/trap adaptation remain unproven.
Resource/descriptor inequality never proves noalias; recovered C++ grants no
execution authority. Existing mutating GEMM has not been reinterpreted as pure.
Native BLAS parity ([#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15))
remains partial; tensor/view types ([#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20))
remain design-only.

## Exactly one next boundary

**Authenticate the same inspection-only subset in a real host translation unit.**
Preserve the grammar and value/effect contract while binding compiler context,
headers, preprocessing and immutable dependency replay. This closes the remaining
C++-host feasibility gap before choosing storage/publication realization.
