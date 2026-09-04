# MDSLC structured GEMM inspection handoff v1

Date: 2026-09-04

Status: implementation checkpoint on `mdslc/structured-gemm-inspection-v1`.
Focused Release validation and independent implementation/test review have
passed. The clean committed regression and package checkpoint remains pending.
This document never grants execution or performance authority.

## Repository and evidence identity

The exact implementation base is
`d80911e388c3883ceedff935bc7be80b9b756c57`. The focused implementation commit
is `d696cd9e2f06bb7580eff22fae6cd42ce4ab78da`. The internal checkpoint is in:

- `compiler/CMakeLists.txt`
- `compiler/lib/mlir/MatcoreStructuredGemmHandoff.{h,cpp}` and its CMake target
- `compiler/tools/matcore-extract/{CMakeLists.txt,main.cpp}`
- `compiler/tools/matcore-mlir/{CMakeLists.txt,main.cpp}`
- `compiler/tests/mlir/matcore_structured_gemm_handoff_test.cpp`
- `compiler/tests/mlir/gemm_capture.structured.golden.mlir`
- `compiler/tests/mlir/run_matcore_mlir_tests.py`
- `compiler/tests/mlir/run_semantic_cpu_pipeline_tests.py`
- this durable record, with bounded status/roadmap links

This record reports validation of the implementation above. It does not
reinterpret the documentation-only reconciliation at the base commit as
validation of these new files.

The evidence boundary inherited from
`docs/mdslc/CORPUS_REENTRY_RECONCILIATION_V1.md` is:

| Evidence | Stable identity |
| --- | --- |
| Corpus research tree | `8aab0e295eb41ca5d2e1bd52c47201b05de9636b` on `origin/research/windows-lowering-corpus-llvm20-22` |
| v1 manifest | `corpus/manifests/windows-corpus-v1.json`, SHA-256 `8d5d3e6d6af89fd78d3af5bd53b530a96e55920cbda65c38d30e7646cd01e187` |
| v2 manifest | `corpus/manifests/windows-corpus-v2-archaeology.json`, SHA-256 `68435d26abc4e498a76432290446bddfd428a64b81e59fe4640218016d3c754d` |
| Structured source example | `8aab0e2:proof/mlir_avx2/01_input.mlir` |
| Produced structured artifact | `8aab0e2:proof/mlir_avx2/02_lowered.mlir` |
| Unused vector example | `8aab0e2:proof/mlir_avx2/02_vectorized.mlir` |
| Hand-authored LLVM input | `8aab0e2:proof/mlir_avx2/03_llvm.mlir` |
| Proof driver and result | `8aab0e2:proof/mlir_avx2/run.ps1` and `8aab0e2:proof/mlir_avx2/RESULT.json` |
| Relevant analysis | `8aab0e2:corpus/findings/semantic-loss-atlas.md`, `lowering-invariants.md`, `adversarial-audit.md`, `matcore-boundary-findings.md`, and `upstream-mlir-findings.md` |

The `proof/` files are a separate, unmanifested 41-file control plane. The raw
Windows data plane, compiler binaries, hardware, and connected target artifacts
were unavailable during reconciliation. The proof's script does not form a
connected `01_input.mlir` to assembly route: it produces `02_lowered.mlir`,
does not consume `02_vectorized.mlir`, and separately translates the
hand-authored `03_llvm.mlir`. Those limitations remain in force here.

## Evidence-to-contract decision

**OBSERVED (corpus):** Direct `linalg.matmul ... outs(%C)` in the proof reads
and accumulates the initial value of `C`. That contradicts explicit Matcore
GEMM's write-only, overwrite destination semantics. The proof's later
zero-initialized LLVM input is disconnected from the structured stage and has
an incorrect row-major A/C lane mapping, so it is not a lowering template.

**OBSERVED (implementation):** The handoff constructs a fresh tensor/DPS
module containing an explicit positive-zero fill before `linalg.matmul`, and
retains the complete verified `mdsl.gemm` attribute dictionary as a function
contract. It includes both a self-contained verifier and a verifier that
compares the result with the particular source semantic module.

**INFERRED:** An inspection-only tensor/Linalg form can retain a useful
composition seam longer than immediate runtime dispatch. The corpus does not
prove that this form bufferizes without copies, produces a particular vector
form, executes correctly, or improves performance.

**ARCHITECTURAL IMPLICATION:** Matcore must preserve semantic intent and reject
lossy conversion. MLIR 21.1.8 should own the canonical `arith`, `linalg`,
tensor/DPS, and verifier machinery. LLVM and target backends should continue to
own instruction selection, register allocation, and machine scheduling. This
checkpoint deliberately stops before all of those executable lowering stages.

## Implemented v1 schema and exact contract

The internal schema is
`matcore-structured-gemm-handoff-v1`, version `1`, with execution authority
`inspection_only`.

### Input boundary

`deriveStructuredGemmHandoffV1` accepts only a nonempty module that first
passes `verifyMatcoreV1BridgeModule`. It does not mutate that input. The input
must have the exact semantic bridge envelope; a merely similar Linalg module or
unverified recovered operation is not an accepted source.

### Structured module envelope

The derived module has an exact 14-field `mdsl.*` module contract:

- `mdsl.analysis_only = true`
- `mdsl.capture_schema`
- `mdsl.capture_version`
- `mdsl.execution_authority = "inspection_only"`
- `mdsl.execution_intent`
- `mdsl.numerical_profile`
- `mdsl.producer = "matcore-structured-gemm-handoff-v1"`
- `mdsl.source_bridge_schema = "matcore-mlir-semantic-v1"`
- `mdsl.source_file`
- `mdsl.source_producer`
- `mdsl.source_semantic_version`
- `mdsl.structured_handoff_schema = "matcore-structured-gemm-handoff-v1"`
- `mdsl.structured_handoff_version = 1 : i32`
- `mdsl.translation_unit`

There is one public structured function per source semantic function, in source
order. Its type and location equal the source function's type and location. Its
exact `mdsl.*` function attributes are:

- `mdsl.capture_ordinal`
- `mdsl.semantic_contract`
- `mdsl.site_id`
- `mdsl.source_semantic_symbol`
- `mdsl.structured_handoff`

`mdsl.structured_handoff` fixes the source operation to `mdsl.gemm`, the
destination rule to `original_output_full_zero_fill`, the authority to
`inspection_only`, and the handoff version to `1 : i32`.

The retained `mdsl.semantic_contract` is the exact `mdsl.gemm` attribute
dictionary, with these 13 fields:

- `accumulation_type`
- `aliasing`
- `effects`
- `lhs_semantics`
- `numerical`
- `origin`
- `output_semantics`
- `policy`
- `provenance`
- `rhs_semantics`
- `semantic_requirements`
- `site_id`
- `synchronization`

The output is structured-only: it contains upstream `func`, `arith`, and
`linalg` operations rather than `mdsl.gemm`. Matcore semantics remain in the
versioned and verifier-checked attributes above.

### Function body and overwrite proof

Each function must contain exactly this four-operation sequence:

```text
%zero = arith.constant +0.0 : f32
%filled = linalg.fill ins(%zero) outs(%original_output)
%result = linalg.matmul ins(%lhs, %rhs) outs(%filled)
func.return %result
```

The custom verifier checks all of the following:

1. The constant is exact positive `f32` zero.
2. The fill's output is the original third function argument and its sole
   tensor result is the matmul initialization value.
3. The matmul inputs are the original LHS and RHS arguments; its result is the
   returned value.
4. Fill and matmul carry the exact site marker and respectively the roles
   `destination_overwrite_zero_fill` and `gemm_contraction`.
5. All four operations have the source function location.
6. The matmul has canonical logical `(m,k)`, `(k,n)`, `(m,n)` maps and
   parallel/parallel/reduction iterators; user-defined maps are rejected.
7. The scalar region is exactly `mulf(lhs, rhs)`, `addf(acc, product)`, and
   `linalg.yield`, with no fast-math flags.
8. Functions carry no argument/result optimizer facts, `no_inline` policy, or
   explicit visibility property that could be silently inherited or dropped.
9. Normal MLIR verification also succeeds.

This is a logical overwrite proof: no initial element value of the original
output tensor reaches the contraction accumulator. It is not a proof of
physical in-place mutation, output-buffer identity after bufferization, or
copy elimination. A nonzero-initial-C case with zero LHS or RHS is the primary
falsifier: any nonzero returned element would disprove the logical overwrite
interpretation. A basis-matrix or non-square sentinel test must separately
falsify transposed or otherwise incorrect indexing.

### Retained-contract verification

The self-contained verifier reconstructs a temporary `mdsl.gemm` witness from
each function's types, results, location, and retained semantic dictionary, and
runs the existing Matcore operation verifier. The paired
`verifyStructuredGemmHandoffMatchesV1` additionally re-verifies both modules
and compares module metadata, function count and order, types, locations,
symbols, site identities, and the exact retained contract.

Comparison uses structural textual identity for types, locations, attributes,
and contracts, so the paired verifier also works when the structured module
was reparsed in a different `MLIRContext`. These mechanisms detect semantic
loss inside this projection. They are not a cryptographic source
authentication mechanism.

## Guarantees and authentication boundary

The following are implemented invariants exercised by the focused local tests:

- derivation constructs a new module and leaves the input module read-only;
- conversion fails closed unless the exact semantic bridge verifier accepts;
- destination overwrite is represented explicitly by full zero fill;
- the complete source `mdsl.gemm` contract is retained and re-verifiable;
- no alias, noalias, alignment, or reassociation fact is synthesized;
- source order, locations, site identity, and source metadata are paired with
  the structured result;
- the primary connected route is native in-process
  `matcore-extract --semantic-pipeline=matcore-mlir --ir-version=1
  --structured-ir-out FILE`; it derives and checks the structured sibling from
  the same verified semantic module used by the unchanged CPU lowerer;
- the supplemental `matcore-mlir` JSON inspection CLI defaults to semantic
  output and requires explicit `--emit-stage structured-gemm-v1` selection.

The source boundary differs by entry point. `matcore-extract` connects the
structured derivation to the native LibTooling capture in one process, but the
current explicit-GEMM v1 contract still carries declared file/range provenance
rather than a cryptographic source snapshot. The supplemental `matcore-mlir`
CLI starts from parsed and verified Matcore IR v1 JSON and therefore does not
reauthenticate source bytes. Producer text is never execution authority. The
self-contained verifier authenticates internal schema consistency; the paired
verifier proves equality to the provided verified semantic module. Neither
turns an imported module into an authenticated executable artifact.

If serialized/imported structured MLIR is ever considered for execution, it
will require a separate sealed authority mechanism. This v1 schema must not be
reused as that authority.

## MLIR 21 ownership and executable firewall

Matcore owns the semantic dictionary, provenance linkage, exact overwrite
initialization requirement, inspection authority, and fail-closed projection
checks. Upstream MLIR 21.1.8 owns the canonical `arith.constant`,
`linalg.fill`, `linalg.matmul`, tensor/DPS behavior, dialect verification, and
any later standard structured transformations. LLVM/backends would own later
vector/machine lowering, but no such route is introduced here.

The executable firewall consists of all of these boundaries together:

- `mdsl.analysis_only = true`;
- `mdsl.execution_authority = "inspection_only"`;
- no bufferization, vector, LLVM, object, runtime, or provider lowering API;
- the CLI only serializes the selected inspection module;
- the existing runtime/provider path remains the sole executable GEMM route.

The attributes are labels checked by this internal verifier, not capabilities
that make arbitrary consumers safe. Downstream code must continue to reject
this module for execution unless a future authenticated executable contract is
designed and validated.

## Explicit non-guarantees

This checkpoint does **not** establish:

- an executable structured lowering or authorization to consume imported
  structured MLIR as executable input;
- bufferization legality, no-copy DPS lowering, physical output aliasing, or
  in-place mutation;
- propagation of retained layout, stride, memory-space, alias, alignment,
  effect, or numerical-policy facts into upstream optimizer assumptions;
- permission to reassociate, contract, introduce fast math, or change floating
  point behavior;
- vectorization, tile selection, register-pressure limits, packing policy,
  fusion, target lowering, or generated execution;
- correctness for recovered ordinary-C++ GEMM, `mdsl.map`, `sin`, malformed
  inputs, or any new public operation surface;
- planner/default/provider behavior changes, OpenBLAS parity, or any
  performance result;
- public API, public ABI, serialized interchange, or backend-contract
  stability;
- GPU, NPU, heterogeneous placement, or physical accelerator evidence.

Issue #15, Native BLAS Parity, therefore remains partial and open under its
original criteria. This inspection seam supplies none of the missing
authenticated native/OpenBLAS envelope, scaling, planner-regret, or active
cooperative packed-B evidence.

## Validation evidence

Focused Release validation used the established Clang/LLVM/MLIR 21.1.8 tuple
with Matcore MLIR enabled, default semantic pipeline `matcore-mlir`, and
OpenBLAS disabled:

```text
structured GEMM adversarial suite
  PASS: 220 checks, 0 failures

matcore-mlir CLI contract
  PASS: 13 cases, 0 failures

native semantic/structured/CPU integration
  PASS: 132 checks, 0 failures

focused CTest entries
  PASS: 3/3
```

Those tests cover dynamic shape, static non-square `2x3x4`, and multiple-site
projection; deterministic serialization and a reviewed structured textual
golden; cross-context parse/print and paired verification; hostile nonzero/NaN
initial C with zero LHS; a non-square indexing sentinel; representative
high-risk contract, provenance, envelope, dataflow, map, scalar-region,
fast-math, `no_inline`, and visibility mutations; recovered/composed/malformed
source rejection; the native in-process extractor connection; and unchanged
legacy artifacts and CPU execution when structured output is requested.

Independent implementation and test reviewers reported no remaining
correctness or execution-authority blocker. The evaluator is a test-local
interpreter of the exact checked structured operations; it is not generated
MLIR execution and does not prove bufferization.

The following clean-head checkpoint evidence remains pending at this document
revision:

- the full registered 64-test Release regression, including installed and
  source-inaccessible consumers and provenance-sensitive benchmark contracts;
- a supported Debug build/test scope;
- an MLIR-disabled/default-capture compatibility build/test scope.

No OpenBLAS-enabled, Windows, sanitizer, physical accelerator, or performance
run is claimed by this checkpoint unless added below after it is actually run.

## Unresolved evidence

- No connected, fingerprinted Matcore semantic-to-structured-to-target trace
  exists; the present checkpoint intentionally ends at structured MLIR.
- The corpus supplies no correctness-executed MLIR proof for dynamic/static
  shapes, nonzero initial C, or row-major basis/sentinel cases.
- Retained source strides and layouts are not yet consumed as Linalg/tensor
  layout facts. Whether and how to do so without overclaiming remains open.
- Alias and alignment requirements remain contract preconditions, not proven
  optimizer facts. No guard or static proof has been added.
- The numerical policy is retained while the scalar region remains maximally
  conservative with no fast math. A future mapping of explicit permissions to
  MLIR/LLVM flags requires separate evidence and falsification tests.
- Tensor DPS syntax does not prove buffer ownership, packing identity, copy
  behavior, or output storage identity after bufferization.
- Source snapshot authentication and authority for imported/serialized MLIR
  remain deliberately unresolved.
- No corpus evidence resolves fusion, tiles, register caps, packing lifetime,
  provider crossover, target routes, or Native BLAS Parity.

## Exact next technical boundary

After the clean checkpoint above, the smallest justified follow-up is an
inspection-only **bufferization legality and destination-identity proof** for
this already verified structured module. It must remain a new downstream seam,
not a change to the v1 language meaning or the current executable route.

Matcore should own admission from this exact handoff schema, a ledger showing
where each retained semantic fact remains represented or is legally consumed,
and postcondition checks that the returned GEMM value denotes the original
output storage after bufferization. Alias and alignment requirements remain
preconditions: they may not become optimizer facts without static proof or a
dominating runtime guard.

MLIR should own One-Shot Bufferize and canonical tensor/buffer transformation
machinery. LLVM/backends should continue to own instruction selection,
register allocation, target scheduling, and machine lowering. The existing
runtime and authenticated external providers remain the only executable GEMM
territory.

Entry evidence must inspect the exact MLIR 21.1.8 bufferization interfaces and
produce connected static and dynamic traces. Completion would require a
deterministic, mechanically verified buffer-level inspection artifact that
accounts for every allocation/copy, proves destination storage identity and no
read of initial C, retains or legally consumes every semantic contract, rejects
unsupported cases, and leaves all current runtime/provider behavior unchanged.
Zero-copy or in-place lowering must not be claimed unless those properties are
actually proven. Stop before vector, LLVM, generated execution, planner/provider
policy, fusion, native-BLAS work, or GPU/NPU work.
