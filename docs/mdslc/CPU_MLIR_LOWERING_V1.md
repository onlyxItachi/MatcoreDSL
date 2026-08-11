# Matcore MLIR to CPU runtime-dispatch lowering v1

- Status: design contract for Milestone E; implementation is not accepted by
  this document
- Scope: explicit F32 rank-two `mdsl.gemm` sites on the existing synchronous
  host-resident CPU route
- Architecture source of truth: ADR-0009 and
  `MATCORE_MLIR_DIALECT_V1.md`

## Purpose

The first executable consumer of the Matcore semantic dialect must prove that
the new optimizer boundary participates in the real `.mdsl` artifact pipeline
without discarding the CPU implementation work that already passes runtime,
package, and platform gates.

The initial lowering is therefore a **runtime-dispatch library lowering**:

```text
verified Matcore IR v1 capture
  -> verified mdsl.gemm semantic site
  -> CPU legality and guard obligations
  -> deterministic runtime-dispatch backend entry
  -> existing stable C descriptor ABI
  -> existing deterministic CPU planner and implementation registry
  -> clang/LLVM object and executable
```

It is not a claim that MDSLC has generated a new Linalg/Vector loop nest. A
future structured-compute lowering is another legal implementation candidate,
not a prerequisite for validating the semantic boundary.

## Ownership boundaries

The layers have distinct responsibilities:

| Layer | Responsibility |
| --- | --- |
| Clang frontend | Authenticate the source declaration and exact source range after Sema. |
| Matcore IR v1 | Preserve the deterministic explicit-operation capture and provenance DTO. |
| Matcore MLIR | Preserve the generic operation, destination identity, numerical contract, effects, aliases, requirements, and provenance. |
| CPU semantic lowering | Prove the supported semantic envelope and choose the runtime-dispatch implementation family. |
| Runtime planner | Evaluate concrete M/N/K, capabilities, workspace, provider availability, requested resources, and stable implementation variants. |
| Runtime backend | Execute the selected reference, native, parallel, or OpenBLAS implementation. |
| Clang/LLVM linker toolchain | Produce and link normal platform artifacts. |

The CPU lowering must not select a private microkernel ID at compile time when
the concrete dynamic problem or runtime capability record is unavailable.

## Accepted semantic envelope

The v1 lowering accepts a site only when all of these are true:

- the enclosing module passes normal MLIR verification and the strict
  `verifyMatcoreV1BridgeModule` envelope;
- the site has one authenticated explicit-call `mdsl.gemm` operation;
- target policy is `cpu` and fallback policy is `error`;
- operands, destination, and result are rank-two F32 tensors with the exact
  operation-local M/K/N relationships;
- accumulation is F32;
- layouts, strides, memory space, mutability, effects, synchronization, and
  destination/result identity match the dialect-v1 contract;
- numerical profile is exactly `explicit-gemm-f32-v1`;
- execution intent is `generic`; and
- the lowering can retain or dominate every required dynamic precondition.

Recovered-loop forms do not enter this envelope. A strict
`recognized_rewrite_rejected` operation is analysis-only. A
`source_proven_guard_required` operation needs an ordinary-C++ fallback and a
different guarded lowering contract before it may replace source behavior.

## Internal lowering record

The lowering may use an in-memory `CpuRuntimeDispatchRecordV1` to hand verified
site facts to backend-source generation. It is not a new serialized IR and is
not installed as a public type. Its minimum contents are:

```text
schema_version       1
site_id              canonical mc_<32 lowercase hex>
semantic_symbol      source-site MLIR symbol
operation            gemm
dtype                f32
accumulation_dtype   f32
target               cpu
fallback             error
numerical_profile    explicit-gemm-f32-v1
execution_intent     generic
runtime_symbol       matcore_runtime_gemm_f32_v0
required_guards      descriptor, alias, alignment, FP environment
source_provenance    original file/line/column/range
```

It must not contain blocking constants, packed formats, ISA-specific variant
IDs, OpenBLAS handles, or assumed runtime dimensions.

The record is valid only for the verified module instance from which it was
created. It must not outlive or silently diverge from that module.

## Generated backend boundary

For each semantic site, lowering emits the existing generated backend function
shape:

```cpp
extern "C" matcore_status_v0
matcore_generated_backend_<site>_v0(
    const matcore_tensor_desc_v0 *output,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy) noexcept;
```

The definition forwards to the stable runtime dispatch entry only after the
semantic lowering has authenticated the site contract. The runtime remains
responsible for concrete descriptor validation, planner diagnostics, workspace
legality, capability detection, and execution-thread floating-point preflight.

The generated function remains a private implementation seam. Existing C ABI
symbols and layouts do not change. No MLIR type, C++ container, exception, or
template crosses the runtime boundary.

Backend text generation should be shared with the existing code generator
through one normalized internal entry representation. Copying two independent
versions of symbol naming and wrapper generation would create an avoidable
semantic drift surface.

## Dominating dynamic guards

The dialect records alias and alignment requirements as preconditions, not
facts. The runtime dispatch path must retain checks for:

- descriptor ABI and structural validity;
- dtype, rank, shape, stride, layout, memory space, and mutability;
- checked element and byte arithmetic;
- output/input overlap;
- implementation and workspace availability; and
- the `explicit-gemm-f32-v1` execution-thread floating-point environment.

Failure must happen before packing, provider invocation, workspace mutation
that is observable to the caller, or destination mutation. Lowering may remove
a runtime check only after a static proof is represented in a form that the
verifier can authenticate.

The current runtime lacks the complete FP-environment gate. That gate is a hard
Milestone E dependency; merely encoding the numerical dictionary in MLIR does
not prove runtime conformance.

## Artifact path

The supported Linux path is:

```text
foo.mdsl
  -> native matcore-extract
  -> Matcore IR v1
  -> Matcore MLIR bridge and verifier
  -> CPU runtime-dispatch lowering
  -> rewritten host, sites, stubs, semantic backend source
  -> clang++ objects
  -> ordinary relocatable object or normal final link
  -> runtime planner and selected CPU backend
```

On Windows, generated COFF objects are linked directly or archived into a
normal `.lib`; no ELF partial-link assumption is introduced.

The semantic backend source must come from the MLIR lowering when this route is
selected. Generating an MLIR file for inspection while continuing to generate
the executed backend solely from the old v0 projection would not prove an
executable semantic path.

## Frontend selection and compatibility

The native Clang frontend remains the source authority. The AST-JSON bootstrap
path remains compatibility-only and must not silently acquire the semantic
lowering route.

During migration, a build may expose an explicit semantic-pipeline selector.
If Matcore MLIR support was not built, selecting that route fails clearly; it
never silently falls back to the v0-only backend. Once the semantic route
passes all E and beta gates, it may become the normal native pipeline.

The v0/v1 serializer, verifier, rewrite ranges, sites header, descriptor stubs,
runtime ABI, and installed consumer remain compatible.

## Multi-operation boundary

Milestone C proves `mdsl.gemm -> mdsl.map(mdsl.sin)` composition in the
semantic optimizer. This v1 CPU lowering executes only the GEMM envelope.
It must reject an unsupported map/domain pipeline rather than silently drop the
map or execute only GEMM.

A later map lowering may choose a generated scalar/vector loop or a fused
implementation only after numerical, destination, outside-domain, alias, and
effect legality is proven. In particular, mapping `sin` over the GEMM result
does not authorize overwriting the original explicit GEMM destination: the
functional map result is a separate semantic value unless another explicit
commit/destination contract says otherwise.

## Required tests

### Positive

1. One explicit dynamic GEMM bridges, lowers, compiles, links, and executes.
2. Two sites produce distinct stable backend symbols.
3. Namespace aliases remain source-equivalent through canonical declaration
   identity.
4. The backend source is deterministic for the same verified semantic module.
5. Runtime planner diagnostics name the selected legal implementation.
6. The generated backend object has the expected C symbol boundary.
7. Release, Debug, supported sanitizers, install, relocated package, and
   external consumer pass.
8. Linux ELF and Windows COFF/PE artifact conventions remain native to each
   platform.

### Negative

1. Unverified or malformed semantic module.
2. Recovered analysis-only or guard-required origin.
3. Unsupported execution intent.
4. Non-CPU target or non-error explicit fallback.
5. Incomplete or altered numerical profile.
6. Missing destination write effect or broken result/destination identity.
7. Unsupported map/domain operation in the v1 lowering envelope.
8. Alias, alignment, shape, workspace, or FP-environment failure before output
   mutation.
9. Matcore MLIR support unavailable when explicitly requested.
10. Source changes between extraction and semantic backend generation.

Every rejection is deterministic, actionable, and fail-closed. No rejected
module may produce an apparently valid backend artifact.

## Future structured implementations

Linalg/Tensor/MemRef/Vector lowering remains a HOW-level implementation option.
Adding it requires a separate legality and performance proof and must not
remove the runtime/library route. Target-specific LLVM or vendor dialects begin
the MACHINE layer only after planning has consumed the relevant alternatives.

No GPU, accelerator, GEMV, GEVM, fusion, or public API work is authorized by
this contract.
