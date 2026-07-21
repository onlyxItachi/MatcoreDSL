# ADR 0003: Typed Matcore IR v1 and deterministic CPU planner

- Status: Accepted for Milestone 2 implementation
- Date: 2026-07-21
- Scope: Additive standalone `compiler/` IR, planner, and CPU runtime

## Context

The native LibTooling milestone authenticates explicit `matcore::mdsl::gemm`
calls and proves the complete `.mdsl` to ordinary executable path. Its
bootstrap Matcore IR v0 is intentionally narrow: several semantic properties
are fixed by its verifier and serializer rather than represented by the
in-memory capture record, and the CPU runtime always executes one reference
loop.

Milestone 2 needs a typed target-independent contract and an inspectable,
deterministic choice among implemented CPU lowerings without weakening the
working frontend, generated C ABI, installation, or consumer path.

## Decision

Matcore IR v1 is a separate typed schema under `matcore::mdslc::ir::v1`.
It represents operation kind, element and accumulation dtype, structured
static/dynamic dimensions and strides, layout, memory space, minimum
alignment, mutability, effects, alias preconditions, synchronization, target
policy, capability requirements, source provenance, and exact rewrite ranges.
Detected host capabilities and selected implementations are downstream plan
records and are not embedded in the target-independent IR.

The live migration boundary is explicit:

```text
authenticated native/bootstrap capture
  -> verified Matcore IR v0
  -> checked v0-to-v1 expansion
  -> Matcore IR v1 verifier
  -> deterministic v1 serialization
  -> loss-checked v1-to-v0 CPU ABI projection
  -> existing rewrite/generated C ABI path
```

JSON v0 remains byte-for-byte compatible. Unknown schema versions fail rather
than being retried as another version. The stable site-ID domain and runtime
ABI v0 suffix remain unchanged because they version naming and ABI contracts,
not the new semantic IR schema.

CPU capability discovery produces a versioned fixed-bit record containing
only facts consumed by implemented variants. Incomplete discovery is
conservative. Tests inject synthetic capability records; the planner never
depends on enumeration or registration order.

The GEMM registry has a fixed canonical order and stable identifiers for:

- a portable reference implementation;
- a cache-tiled implementation with complete tail handling;
- a compiler-vectorized candidate guarded by its exact CPU requirements.

Every variant has a pure legality decision and an explicit rejection reason.
The initial cost model uses saturating integer arithmetic and a documented
stable tie-break key. Forced illegal selections fail and never fall back.
Automatic selection returns a structured plan containing every candidate
decision, the chosen stable identifier, and a human-readable explanation.

The existing `matcore_runtime_gemm_f32_v0` entry point validates descriptors,
constructs a concrete GEMM problem, plans, and dispatches the selected
implementation. It retains synchronous execution, host-only f32 semantics,
no allocation or copy, status returns, and the prohibition on C++ exceptions
crossing the ABI. Additive versioned inspection APIs may expose plan metadata;
no C++ type crosses the public C boundary.

External BLAS is optional. It may enter the registry only when a coherent
development package, deterministic adapter, legality contract, and independent
tests are available. A runtime library alone is not sufficient. BLAS is not a
Milestone 2 completion dependency.

## Validation contract

- v0 golden serialization and generated artifacts remain stable;
- v1 JSON round-trips deterministically and malformed semantics fail;
- v0-to-v1-to-v0 conversion is loss-checked;
- all three CPU implementations are independently correctness-tested;
- automatic and forced plans are deterministic and explain every decision;
- representative small, medium, rectangular, tail, and alignment-sensitive
  shapes compare with an independent oracle;
- performance evidence is bounded and descriptive, never a global optimum
  claim;
- Release, Debug, supported sanitizers, install, external consumer, artifact,
  and independent adversarial gates remain required.

## Consequences

This milestone provides a real semantic boundary and capability-aware CPU
planner while deliberately retaining the proven frontend and ABI. It does not
claim CUDA, HIP, Metal, NPU, MLIR GPU lowering, GEMV, GEVM, ReLU-GEMM,
heterogeneous placement, fusion, autotuning, or production-wide optimality.
