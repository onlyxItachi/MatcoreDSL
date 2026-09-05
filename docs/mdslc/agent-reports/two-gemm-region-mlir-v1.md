# Source-connected two-GEMM region: MLIR lane

Starting canonical checkpoint: `5f455bacde0959983b2b888f15fd5dabd4b1ceaa`.
Agent branch: `agent/two-gemm-region-mlir-v1`.
This record covers the internal MLIR implementation, not frontend admission,
CLI integration, or a new executable compiler route.

## Design and guarantees

The immutable native-frontend evidence owns admission and compilation identity.
The existing verified v0-to-v1-to-Matcore bridge supplies the full current GEMM
contracts; this lane does not define a competing arithmetic schema. The derived
region carries those contracts, per-call descriptor bindings/snapshot stages,
source ranges and the sealed compilation digest. Editable attributes confer no
source authenticity or execution authority.

Each admitted region is a public, observable void function with this dependency:

```text
begin -> guard0 -> read(A),read(B) -> zero-fill/matmul -> commit(C)
                                                        | value and order
                                                        v
         guard1 -> read(D) -> zero-fill/matmul(C_post,D) -> commit(E) -> end
```

`region_descriptor` identifies a source descriptor binding, not an allocation or
disjoint byte range. Different bindings may overlap; even identical A/B bindings
are admitted. Input D is read after commit C, since distinct D/C bindings may
refer to overlapping storage. No initial destination tensor is read for either
overwrite. Dynamic output extents come from the current lhs row and rhs column
dimensions and are verified along the actual tensor dataflow.

`region_order` represents only the conditional successful-continuation edge.
Guard operations retain required descriptor/layout/host, shape, alignment,
output-input overlap, floating-environment and target/provider-policy checks.
They do not execute checks or prove concrete runtime facts. Source invocations
may fail after writing output: commits retain that obligation, without claiming
atomicity, rollback, safe retry or complete provider-failure modeling. Failure
of call 2 cannot move ahead of the observable write of call 1.

The five small boundary operations have conservative unscoped memory effects.
Guards/commits are non-speculatable. They deliberately have no bufferization
interface or concrete storage lowering. Upstream Linalg owns the tensor
arithmetic; named-to-generic generalization is accepted by checking indexing,
iterators, scalar multiply/add, zero initialization and value flow. Incidental
locations and operation counts are not identity.

## Verification trust boundary and adversaries

Standalone verification checks bounded IR self-consistency. Source-paired
verification separately re-derives the authoritative contract from the native
seal. Coordinated edits to a guard and its matching declared numerical or target
contract can remain self-consistent, but cannot pass source-paired verification.
A second real native extraction with different compiler options produces an
incompatible seal even with unchanged arithmetic and source bytes.

Tests challenge stale postvalues, guard hoisting, late input snapshots, changed
destinations, erased writes, omitted guard families, rollback claims, wrong
dynamic dimensions, nonzero overwrite seeds, fast-math, generic indexing and
iterator changes, dtype drift, altered source identity and missing regions.
Generic semantic mutants are required to pass upstream IR verification before
the Matcore source-pair rejection is counted.

Independent review found that memory-effect freedom alone accepts dead integer
division by zero. That concrete bug was corrected; the final implementation
also limits incidental computations to reviewed constants, identity tensor
casts and in-rank constant dimension queries. This is the bounded verifier's
modeled-operation scope, not a claim that upstream speculatability is unsound.
Unvalidated extra structured computations are rejected without freezing the
total operation count.

Actual canonicalization, CSE, symbol DCE and named-to-generic passes test the
surviving representation. A pure-tensor control demonstrates why unused DPS
results alone cannot represent observable source mutation. A separate upstream
materialization/One-Shot control accounts for allocation, copy into the original
destination and unresolved deallocation ownership. It is neither source-paired
bufferization nor a zero-copy result. Both it and forged capability/target/retry
labels are rejected by the actual existing CPU runtime lowerer.

## Validation status

The integration owner serializes exact-21.1.8 builds with two jobs. The core
compiled after correcting owning-module attribute access. Initial test
compilation exposed a ConstantIntOp factory overload mismatch; commit
`641e7c1068071dd0a06d2841455f315eaeb58c0f` corrects it and adds the semantic
falsifiers above.

At integrated checkpoint `de204b5`, the full Release build succeeded.
`ctest --test-dir build-region-release -R '^mlir.semantic.two-gemm-region$'
--output-on-failure -V` passed: **83/83 checks**, one CTest, 0.04 seconds.
The MLIR lane read the execution log at `/tmp/mdslc-region-mlir-tests.log`;
the independent reviewer separately ran the same 83 checks successfully.
The integration owner also reported 28 native admission extractions and 62 CLI
test steps, including 16 unchanged-route executable runs, passing. The complete
Release CTest suite at `de204b5` passed **73/73 tests** in 201.33 seconds,
verified from `/tmp/mdslc-region-release-ctest.log`.

The final independent review requested fresh-context serialization/parse/source
pairing, because the newly introduced descriptor/order type parser was not yet
exercised. Commit `9ddd9e39cbf43eab95379a1d0c4988f1b6eb614a` adds that control;
at integrated checkpoint `f8f15eb3fad50af81e07e2deecb25b024975c572`, the same
focused CTest passed **85/85 checks** in 0.04 seconds, including the fresh-context
roundtrip and paired verification. The MLIR lane verified the actual log at
`/tmp/mdslc-region-roundtrip-tests.log`. Production implementation is unchanged
from the full-suite checkpoint; the subsequent change adds only these tests.

## Deliberately unresolved

No generated execution, source rewrite, new public frontend type, fusion,
tiling, vectorization, provider selection or runtime policy changes are added.
Buffer allocation/copy ownership and exceptional provider behavior remain
unconsumed execution obligations. The next step requires an explicitly reviewed
storage/failure implementation boundary; deeper buffer-looking IR alone cannot
authorize execution.
