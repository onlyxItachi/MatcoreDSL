# ADR 0007: MDSLC CPU performance foundation v1

- Status: Accepted for the validated Linux host scope
- Date: 2026-07-22
- Scope: F32 CPU GEMM implementations, resources, and deterministic planning

## Context

The native compiler and typed IR milestones proved correct CPU GEMM and a
three-variant deterministic planner. They did not provide a defensible external
baseline, an explicit packed-kernel resource contract, or measured crossover
evidence. A serious packed implementation cannot hide its allocations, and a
library candidate cannot be legal merely because a shared object exists.

## Decision

Planner v2 has a fixed five-entry registry:

```text
cpu.reference.f32.v1
cpu.tiled.f32.v1
cpu.compiler-vectorized.avx2-fma.f32.v1
cpu.external.openblas.f32.v1
cpu.native-packed.avx2-fma.f32.v1
```

The v1 registry and execution ABI remain available for compatibility. Planner
v2 adds implementation resources, workspace/alignment, provider availability,
requested/actual threads, deterministic integer cost, and complete rejection
reasons. Automatic selection is a reviewed static rule; no runtime autotuning
or cache of machine-specific winners exists.

OpenBLAS is optional. Configure authenticates a coherent pkg-config header and
library through a compile/link CBLAS probe, verifies LP64 `blasint`, records
provider version/configuration, and requires process-local thread control. The
runtime rejects requests above the provider-reported ceiling before selection
or output mutation. A build without OpenBLAS retains every native candidate and
reports the unavailable provider explicitly.

The native packed implementation is separately compiled and function-targeted
for AVX2/FMA. It uses MC=128, NC=256, KC=256, MR=4, NR=16 and a 64-byte-aligned
caller workspace. The microkernel is an isolated 4x16 symbol whose Release and
supported Debug objects must contain YMM operations and packed FMA. Tails never
speculatively read outside input matrices. CPU/OS capability and compiled-body
availability are separate legality facts.

The stable C ABI grows additively with workspace query/execute and prepacked-B
query/prepare/execute functions. The caller owns every buffer. Required size
and alignment are queryable; insufficient, misaligned, stale, aliased, or
overlapping storage fails before C changes. The old one-shot function may use
only zero-workspace planner-v2 candidates and never hides packed allocation.

Benchmarking follows ADR 0006. Complete implementation results and the
diagnostic-only packed microkernel interval are explicitly distinct. Static
cost changes require sanitized host/provider evidence and regret analysis.

## Consequences

On the declared AMD Ryzen AI 9 HX 370 validation host with Clang 21.1.8 and
OpenBLAS 0.3.32 pthread, the 30-shape calibration met the median, p95, and
catastrophic-regret limits. Native packed materially improved the previous
compiler-vectorized implementation, while OpenBLAS remained the stronger
general single-thread baseline. Neither result is a universal claim.

Reference, tiled, compiler-vectorized, OpenBLAS, and packed implementations are
independently forceable. An illegal forced request is an error, never a
fallback. Seven public C symbols form the runtime DSO boundary; internal C++
backend symbols remain hidden.

This decision does not claim parallel native execution, AVX-512, BF16, INT8,
AMX, real multi-node NUMA, Windows runtime support, BLAS parity, GPU execution,
MLIR lowering, or autotuning. Those require separate capability, artifact,
correctness, performance, package, and hardware-validation gates.
