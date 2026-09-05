# MDSLC certified structured-GEMM vector-readiness seam v1

Date: 2026-09-05

Status: internal, opt-in, inspection-only implementation checkpoint. This is
not an executable lowering route, a production vector schedule, a physical
storage proof, a performance result, or a public API/interchange commitment.

## Identity and dependency chain

The canonical campaign base is
`5983c2c1bf067bed9e69d0172b0944b7a4c14c00`, the merge checkpoint containing
the corpus reconciliation, structured GEMM proof, and reusable contraction
foundation. The vector-specific implementation commit is
`4a6711654f02dc50c4d4fcc2ae5cbafd16f61b75`.

This branch contains local commit `d29e1c3`, patch-equivalent to the shared
derived-structured identity commit
`d3d4189aedb7346f853e0d5bd8d845d892612fdc` prepared independently for draft
PR #25. The vector change is intentionally separable so it can be restacked on
the canonical merge of that shared dependency without changing its design.

The implementation uses the exact LLVM/MLIR 21.1.8 package selected by
`AGENTS.md`. Corpus identity, limits, and responsibility boundaries remain
those recorded in `CORPUS_REENTRY_RECONCILIATION_V1.md`,
`STRUCTURED_GEMM_HANDOFF_V1.md`, and `CONTRACTION_FOUNDATION_V1.md`.

## Evidence disposition

### OBSERVED

- Applying upstream MLIR 21.1.8
  `transform.structured.vectorize_children_and_apply_patterns` to an exact
  certified static `2x3 * 3x4` tensor/DPS GEMM produces whole-tensor reads,
  one `vector.contract`, and one tensor transfer write.
- The contraction uses canonical GEMM maps `(m,k)`, `(k,n)`, `(m,n)`,
  parallel/parallel/reduction iterators, add combining, and an exact positive
  f32 zero accumulator. The result is written to the original output tensor
  argument and returned. Initial C is not read.
- Applying the same upstream Transform sequence to the reviewed dynamic
  structured capture reports success but is an exact payload no-op: the fill
  and matmul remain and no Vector operation is created. Transform success is
  therefore not a readiness proof.
- Positive static unit-M, unit-N, and unit-K rank-2 specimens retain GEMM
  identity and produce the same vector topology.
- MLIR 21.1.8 retains source location on the zero vector, input reads, and
  return, while generated poison/index/contract/write glue has unknown
  location.

The static specimens above are programmatically constructed and accepted by
the authoritative Matcore IR v1 verifier. They are not authenticated static
facts captured by the current frontend. The dynamic fixture is the reviewed
native-capture golden used by the established semantic/structured tests; this
unit test parses that committed capture and does not independently
reauthenticate source bytes.

### INFERRED

- `vector.contract` is a mechanically defensible structured carrier for the
  bounded whole-problem static rank-2 f32 GEMM cases proved here.
- A target-independent production vector pipeline needs a separate bounded
  tile or shape decision before dynamic GEMM can be admitted.
- Direct tensor vectorization is a useful readiness control. It does not prove
  that whole-problem vector shapes are profitable or lowerable for any target.

### UNRESOLVED OR REJECTED

- No universal Linalg/bufferization/vector ordering is established.
  Bufferization remains an independent legality boundary.
- Dynamic, tiled, scalable, masked, transposed, batched, GEMV, DOT, GER, and
  non-f32 vector forms are not certified.
- No corpus tile, vector width, target matrix shape, unroll factor, cache
  choice, or provider threshold becomes policy here.
- The tensor transfer write proves a tensor SSA/DPS result rooted at the
  original output argument. It does not prove physical memref identity,
  in-place mutation, allocation freedom, copy freedom, or zero-copy behavior.
- The earlier experiment's GEMM-specific reconstructed structured witness and
  byte-serialization equality are rejected as duplicated provenance
  machinery. The shared derived-structured certificate is the surviving
  source-binding mechanism.

## Connected implementation

```text
exact verified certified structured GEMM
                 |
                 v
static positive rank-2 f32 admission
                 |
                 v
clone source module; source remains immutable
                 |
                 v
upstream MLIR 21.1.8 Transform interpreter
  vectorize_children_and_apply_patterns
                 |
                 v
shared structured -> derived identity attachment
  exact source profile
  per-site type + semantic fingerprint
  ordered aggregate site-set fingerprint
                 |
                 v
vector-specific ledger + exact postcondition verifier
                 |
                 v
standalone self-consistency and exact paired-source verification
                 |
                 X
          no execution authority
```

`MatcoreStructuredHandoffCertificate` owns the operation-neutral provenance
chain. It binds each vector function to the exact certified structured source
type and semantic fingerprint and binds the complete ordered site set at
module scope. Paired verification recomputes that identity from the caller's
exact verified structured module. Reordering, dropping, substituting, or
changing a source site fails closed.

`MatcoreStructuredGemmVectorReadiness` owns only GEMM-specific admission and
postconditions. It uses `verifyStructuredGemmHandoffV1`, the shared certificate
helpers, `verifyRetainedStructuredGemmContractV1`, and the canonical GEMM
topology from `MatcoreContractionModel`. It does not reconstruct an invented
structured witness or use whole-module byte equality as provenance.

Standalone verification proves that an artifact is internally
self-consistent and still satisfies the retained exact GEMM semantic contract.
Only paired verification proves that it corresponds to the particular
certified structured module supplied by the caller. Neither mode authenticates
imported source bytes or grants execution.

## Exact vector postcondition

Every accepted function contains exactly:

```text
ub.poison f32 padding
arith.constant 0 : index
arith.constant dense<+0.0> : vector<MxNxf32>
vector.transfer_read A[0,0] : tensor<MxKxf32> -> vector<MxKxf32>
vector.transfer_read B[0,0] : tensor<KxNxf32> -> vector<KxNxf32>
vector.contract canonical GEMM topology, add, zero accumulator
vector.transfer_write contract_result, original_C[0,0]
func.return transfer_write_tensor_result
```

M, N, and K must be positive compile-time values. The two reads and write are
unmasked, identity-mapped, full-rank, and fully in-bounds. The contraction
maps are checked against the reusable canonical GEMM topology rather than a
second private definition. The only result path passes through the
contraction and transfer write. Reading initial C, changing maps, adding
unreviewed hints, changing locations, or changing dataflow is rejected.

This is a logical overwrite proof at tensor SSA/DPS level only.

## Retention and consumption ledger

The exact `mdsl.vector_readiness` dictionary records:

- `authority = inspection_only`;
- the exact upstream Transform operation;
- `vectorization_scope = whole_static_problem_inspection`;
- the zero-accumulator/full-write destination encoding;
- the observed partial operation-location behavior;
- exact retention of the semantic contract;
- an exact copy of the semantic contract's numerical-permission dictionary;
- the complete list of requirements that remain unconsumed.

The unconsumed list is exactly:

```text
alias_preconditions
alignment_preconditions
effects
layout_and_strides
memory_space
numerical_permissions
provenance
target_policy
```

Shape and GEMM indexing are represented in vector types/maps, and logical
overwrite is represented by the positive-zero accumulator and full write.
Those structural facts do not authorize deleting the retained contract. In
particular, contraction/reassociation permissions remain explicit and
unconsumed: this seam neither introduces fast-math authority nor decides a
machine reduction order.

## Build and dependency firewall

The new library and test exist only when
`MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS=ON`. The option defaults OFF
and requires `MDSLC_ENABLE_MATCORE_MLIR=ON`. Transform, Tensor, UB, and Vector
MLIR targets are required only inside that opt-in branch.

No installed header, package target, CLI stage, frontend operation, planner,
runtime, provider, or C ABI is changed. The vector module retains
`mdsl.analysis_only = true` and `mdsl.execution_authority =
"inspection_only"`. The existing CPU runtime lowerer rejects it and remains
the only executable GEMM path.

## Mechanical falsification

The focused suite currently performs 280 checks, including:

- non-square exact topology and deterministic golden output;
- repeat derivation and cross-context parse/print verification;
- source immutability;
- positive two-site derivation and exact ordered pairing;
- reordered and dropped multi-site artifacts;
- substitution of another independently valid structured source with a
  different static shape;
- aggregate and per-site fingerprint forgery;
- retained source-type/shape forgery;
- unit-M, unit-N, unit-K, and all-unit rank-2 GEMM specimens;
- the direct dynamic Transform success/no-op control followed by fail-closed
  readiness rejection;
- malformed structured source rejection;
- module/function authority, version, consumption, provenance-location,
  numerical-permission, alias-precondition, semantic tile, and vector-width
  forgery;
- type-compatible canonical-map substitution on a square GEMM;
- nonzero accumulation, initial-C read, partial transfer, contraction/write
  bypass, and unreviewed operation-hint rejection;
- rejection by the existing CPU runtime lowerer with no partial record.

The exact MLIR 21.1.8 output is pinned in
`compiler/tests/mlir/gemm_capture.vector-readiness.golden.mlir`. A toolchain
change that alters it requires review; blindly refreshing the golden is not an
upgrade strategy.

## Ownership and next boundary

- Matcore owns exact source admission, semantic/numerical retention,
  consumption accounting, provenance pairing, and the vector postcondition.
- MLIR owns Transform application and standard Vector operations.
- LLVM/backends own legal target shapes, instruction selection, register
  allocation, and machine scheduling.
- Authenticated runtime/library providers remain the only execution route and
  remain valid provider candidates.

Issue #15 remains partial and open. This work adds no performance, provider,
scaling, planner-regret, or BLAS-parity evidence.

The next technically justified vector step requires explicit bounded
tile/shape evidence and a reviewed choice of tensor-versus-buffer transform
ordering. Its acceptance criteria must cover dynamic remainders/masks,
consumed semantic requirements, target-independent versus target-specific
ownership, and continued execution-authority isolation. This checkpoint stops
before that work.
