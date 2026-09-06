# MDSLC current state

Engineering checkpoint: canonical merge
`f988882710ac0b3677d908b3442841e3a8986b81`, [PR #39](https://github.com/onlyxItachi/MatcoreDSL/pull/39).
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
Matcore owns semantic legality, provenance and observable ordering.
MLIR owns structured transformations; LLVM/backends own machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

The host storage/failure contract now has an executable adapter and independent
falsification coverage. Old values survive overlapping writes; late reads see
publication; later failures preserve earlier effects. Host publication has a
strong normal-return guarantee, not whole-region rollback. Validation: 83/83
Release, 32/32 affected ASan/UBSan, 19/19 hosted checks; independent review ACCEPT.
See [decision and evidence](FOUNDATION_RESOURCE_DECISION_V1.md).

## Unsupported or unproven

Closed source is still a private Linux 21.1.8 admission proof, not an installed
frontend or connected generated execution. No public region syntax, general
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

**Connect authenticated closed regions to a validated strict generated CPU
candidate through the checked host adapter.** The host storage/failure boundary
is now defended; source orchestration, generated-kernel issuance and trusted
candidate selection must compose before claiming source-to-generated execution.
Independent research branches are not canonical capabilities until integrated.
