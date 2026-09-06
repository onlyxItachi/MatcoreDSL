# MDSLC current state

Engineering checkpoint: canonical merge
`3374ffbb2100dd68fdd34a46ee93495d1c3c4137`, [PR #52](https://github.com/onlyxItachi/MatcoreDSL/pull/52).
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
Named C++ region admission + opaque Result/Value ownership and private ABI gate.
```

## Material change

Private Value ownership no longer exposes a standard-library layout. Out-of-line
retention preserves old values, failure prefixes and observations; Session
construction remains allocation-free. Versioned private linkage rejects stale
artifacts, including the reproduced weak-constructor bypass of a marker-only
design. Exact-head Release passed 115/115, ASan/UBSan 62/62 and all 19 hosted
checks passed. No mathematical or candidate-selection semantics changed.
See [implementation and canonical evidence](agent-reports/closed-private-value-abi-v2.md#canonical-merge-checkpoint)
and [independent review](agent-reports/private-value-abi-independent-v2.md).

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

**Promote opt-in region build/install support independently of test targets.**
The ownership boundary is defended. A usable package must now build the reviewed
runtime and generated leaf with `BUILD_TESTING=OFF`, install matching private
artifacts, and validate installed consumers without leaking LLVM/MLIR dependencies.
This is a packaging prerequisite, not the new source-driver execution boundary.
Independent research branches are not canonical capabilities until integrated.
