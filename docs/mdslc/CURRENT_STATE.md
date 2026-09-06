# MDSLC current state

Engineering checkpoint: canonical merge
`2201ef9154655406c6241bd6fd2d49889f1cbdba`, [PR #45](https://github.com/onlyxItachi/MatcoreDSL/pull/45).
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
Separate trusted registry -> native/generated strict or legal legacy/provider
  -> isolated candidate output, verified completion, then immutable value issuance
Shared provider adapter serializes Matcore-owned OpenBLAS policy lifetimes.
Matcore owns legality/effects; MLIR transformations; LLVM/backends machine lowering.
Legacy Python/JIT remains a separate compatibility surface.
```

## Material change

The private adapter now selects only built-in trusted candidates: strict native,
strict MLIR-generated, or compatible existing reference/OpenBLAS implementations.
Numerical legality, isolated output, full FP restoration and failure prefixes
gate value issuance; forced incompatible/unavailable choices never fall back.
Default remains strict native. No performance threshold was introduced.
Hosted Release provider-ON/OFF passed 102/102 and 101/101, Debug 100/100,
ASan/UBSan 50/50, and all 19 checks were green. See the
[candidate evidence and merge record](agent-reports/closed-candidate-coexistence-v1.md#canonical-merge-checkpoint).
The [authenticated source consumer](CLOSED_SOURCE_EXECUTION_V1.md) still uses
strict native; connecting it to this generated-candidate registry is next.

## Unsupported or unproven

Closed source is a private Linux 21.1.8 compiled native-adapter consumer, not an
installed frontend or connected source-to-generated math path. The generated leaf accepts
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

**Prove the connected authenticated-source to generated-CPU path.** Execute the
same admitted source through the now-canonical registry, retaining a strict-native
control, exact actual-candidate reports, storage/failure checks and real generated
memory instrumentation. The individual boundaries are proven; their composition
must pass before claiming source-to-generated mathematical execution.
Independent research branches are not canonical capabilities until integrated.
