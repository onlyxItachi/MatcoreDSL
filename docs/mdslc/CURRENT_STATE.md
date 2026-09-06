# MDSLC current state

Engineering checkpoint: canonical merge
`3a0f995d522bb3809c907ee68bc50e2a820d7ea6`, [PR #55](https://github.com/onlyxItachi/MatcoreDSL/pull/55).
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
Opt-in production region package: opaque Result/Value and private ABI gate.
```

## Material change

Region support now has one production build owner, independent of test targets.
The default-OFF option installs matching experimental headers and private runtime
artifacts; ordinary consumers do not inherit LLVM/MLIR dependencies. Actual
tests-disabled ON/OFF builds and installed consumers passed. Implementation-head
Release passed 116/116 and affected ASan/UBSan 63/63; final-head hosted checks
passed 19/19. No mathematical or candidate-selection semantics changed.
See [package contract](EXPERIMENTAL_REGION_BUILD_INSTALL_V1.md) and
[integration evidence](agent-reports/experimental-region-build-install-integration-v1.md#canonical-merge-checkpoint).

## Unsupported or unproven

Closed source is a private Linux 21.1.8 compiled native/generated consumer, not an
installed frontend or a whole-host-TU replacement. The generated leaf accepts
no user IR and grants no source authority. Experimental named-region syntax is
not yet a shipped execution interface. The unmerged driver has a reproduced
private weak-symbol cleanup ownership gap ([#56](https://github.com/onlyxItachi/MatcoreDSL/issues/56));
installed private archives are not proof against host implementation replacement.
No general
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

**Defend compiler-private implementation ownership before admitting the installed driver.**
The package now exists, but whole-archive linkage does not protect private weak
cleanup functions on allocation failure. Isolate issued helper definitions and
runtime internals, then falsify both boundaries with actual source/execution and
sanitizer tests. Passing ordinary arithmetic alone is insufficient.
Independent research branches are not canonical capabilities until integrated.
