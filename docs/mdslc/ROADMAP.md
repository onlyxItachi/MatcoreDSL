# MDSLC roadmap

The roadmap is gate-driven and additive to the legacy Python/JIT path. A
backend is supported only after it compiles, links, executes, validates
correctness, and passes its declared acceptance suite.

## Completed architecture gates

1. **Standalone build — complete.** `compiler/` builds independently of root
   CMake, Python, nanobind, MLIR, and the legacy JIT.
2. **Valid-C++ driver — complete.** `mdslc++` forces `.mdsl` through Clang as
   C++, preserves ordinary host C++, and produces executables or relocatable
   objects.
3. **Native LibTooling frontend v1 — complete.** In-process Clang 21
   PPCallbacks, parse/Sema, ASTMatcher, canonical declaration and exact
   annotation authentication, trusted-header identity/ABI checks, and
   SourceManager/Lexer ranges are the validated default.
4. **Typed Matcore IR v1 — complete.** Deterministic JSON v0 remains the
   compatibility boundary. Every captured operation is upgraded to verified,
   target-independent IR v1 with typed shape, dtype, accumulation, layout,
   stride, alignment, memory, mutability, effects, alias, policy,
   synchronization, requirement, provenance, and source-range contracts. The
   v0-to-v1 upgrade and lossless v1-to-v0 projection are tested explicitly.
5. **Deterministic CPU planning — complete for GEMM.** Versioned CPU
   capability discovery, a fixed reference/tiled/compiler-vectorized registry,
   fail-closed legality, saturating integer costs, deterministic tie-breaking,
   selected-plan metadata, and rejection diagnostics are implemented.
6. **Rewrite, C ABI, and selected CPU GEMM — complete.** Generated
   sites/stubs/backend retain the stable execution ABI, while runtime dispatch
   executes the selected legal plan. The additive plan-report ABI and installed
   `matcore-plan` tool expose decisions without executing or mutating output.
7. **Installation and consumer — complete.** Tools, headers, runtime, CMake
   package/helper, fresh-prefix external consumer, dependency regeneration,
   relocation, and no-op rebuild pass.
8. **Adversarial validation — complete for the declared slice.** Release,
   Debug, supported sanitizer scope, native/bootstrap parity, fake-header and
   annotation attacks, malformed IR/capability records, source/dependency
   races, artifact inspection, package, and legacy smoke gates have
   reproducible evidence.
9. **Mainline integrity and history sanitation — complete.** The legacy and
   standalone compiler lineages are consolidated on canonical `main`;
   generated artifacts are rejected by repository hygiene and removed from
   user-controlled historical refs without deleting source history.
10. **CPU performance foundation — complete for the validated Linux host.**
    The optional OpenBLAS baseline, explicit workspace/prepacked-B ABI, native
    packed AVX2/FMA engine, five-variant planner v2, reproducible benchmark
    contract, host-specific static calibration, packaging, sanitizers, and
    independent review pass.
11. **Advanced CPU backend — complete for the validated Linux host.**
    Capability record v2, guarded packed AVX-512 F32, persistent parallel
    AVX2/AVX-512 execution, topology/affinity policy, synthetic multi-node
    NUMA planning, typed BF16/F32 and I8/I32 reference semantics, packaging,
    sanitizers, calibrated planner v3, hosted checks, normal merge, and the
    immutable `mdslc-cpu-backend-v2` checkpoint pass.
12. **Windows x64 compiler/runtime distribution — validated on the hosted
    scope.** clang-cl/LLVM 21.1.8, the MSVC ABI, COFF/PE generation, runtime
    DLL/import library, native LibTooling frontend, deterministic planner,
    runtime-validated AVX2 variants, install/relocation/external consumer,
    focused runtime/generated-code ASan, exact ISA artifact checks, and the CI
    ZIP candidate pass. AVX-512 runtime remains unavailable on the hosted CPU,
    Windows OpenBLAS is omitted, multi-node NUMA is synthetic-only, and the VC
    Redistributable remains an external prerequisite.
13. **CPU performance deep audit — complete and published.**
    Schema-v6 forward/reverse evidence is fail-closed and reproducible;
    packing/data movement, microkernel scheduling, cache/roofline limits,
    persistent parallel execution, benchmark fairness, BLAS architecture, and
    future GEMV/GEVM backend design have independent audit records. Fresh
    Release, Debug, OpenBLAS-disabled, supported sanitizer, package,
    source-inaccessible consumer, native artifact, legacy, and hygiene gates
    pass. PR #14 passed hosted checks, merged normally, and is anchored by
    `mdslc-cpu-performance-audit-v1`. Hardware counters were blocked,
    controlled multi-thread OpenBLAS placement was not established, and native
    BLAS parity is not claimed.

The architecture verdict is **passed for the standalone native CPU
frontend/runtime vertical slice**. The AST-JSON producer remains an explicitly
selected compatibility oracle only; it is never a fallback from native.

## Milestone 6 published acceptance

The bounded CPU Performance Deep Audit passed local integration, independent
review, hosted checks, normal merge, immutable tagging, and tracker closure.
Its authenticated evidence explains weak native regions without changing
production planner behavior:

- medium/large square and tail-heavy native throughput remains below the
  matched single-thread OpenBLAS baseline;
- tall-skinny is the weakest declared regular family;
- one static microkernel/blocking family constrains the native implementation
  space;
- transient packing dominates vector-like one-shot work;
- parallel work is decomposed only into 128-row M bands, constraining
  short-wide shapes and creating row-boundary imbalance;
- full B preparation remains serial, and comparable multi-thread provider
  placement was not established.

The audit proposes measured experiments for Milestone 7; it does not itself
promote a kernel, alter planner thresholds, expose GEMV/GEVM, or freeze the
public API.

## Milestone 5 published Linux gate and focused Windows validation

The advanced Linux CPU implementation passed its local, independent-review,
hosted, normal-merge, and immutable-tag gates for the declared validation
host. The accepted implementation includes:

- capability record v2 with separate hardware, OS state, compiler,
  implementation, and physical runtime-validation domains;
- planner v3 with eight stable F32 candidates, adding packed AVX-512 and
  persistent parallel AVX2/AVX-512 to the Milestone 4 registry;
- exact-context legality, deterministic topology/affinity policy, explicit
  shared/per-worker workspace, and a persistent opaque C execution context;
- physically exercised AVX2/FMA and AVX-512F/FMA F32 paths on the declared
  one-node Linux validation host;
- typed BF16-to-F32 and I8-to-I32 reference semantics and additive C entry
  points; and
- synthetic multi-node NUMA planning without hidden page placement.

The implementation does **not** include accelerated BF16/VNNI/AMX or physical
multi-node NUMA validation. The subsequent focused Windows x64 phase also
validated the native frontend, COFF/PE artifact model, runtime DLL/import
library, deterministic planner, AVX2 native variants, package relocation,
external consumer, and ZIP distribution candidate. Windows AVX-512 remains
compile/disassembly-only, OpenBLAS is omitted, and no Windows performance claim
is inferred from Linux evidence.

## Next three exact engineering tasks

1. **Complete Milestone 7 evidence and review.** Evaluate the retained
   full-tile/microkernel, shared packed-B, and two-dimensional tasking changes
   against the complete declared matrix without a blind-holdout claim.
2. **Calibrate and validate the resulting native parity envelope.** Compare
   native variants and OpenBLAS under equal single-/multi-thread placement,
   keep allocation and packing visible, harden planner regret, run all
   correctness/sanitizer/package/Windows gates, and report partial status
   rather than changing the envelope if the declared targets are not met.
3. **Prepare the separate API/backend-contract freeze only after Milestone 7
   is dispositioned.** Resolve transformed-operand lifetime, forced variant
   identity, report growth, and device-neutral execution-context questions;
   do not freeze automatically from performance evidence.

## Later operation and lowering milestones

- Add `gemv`, `gevm`, then `relu_gemm` individually, with explicit legality,
  effects, ABI, runtime, diagnostics, and negative tests.
- Introduce a Matcore MLIR dialect/bridge only as the documented consumer of
  the high-level IR. Lower through structured dialects where appropriate; do
  not make Clang the matrix optimizer.
- Add target-specific scheduling, tiling, vectorization, memory mapping,
  legalization, runtime loading, and validation for every claimed target.
- Consider generated NVIDIA kernels only after the library-call backend passes.
  WGMMA requires a detected legal architecture and actual runtime validation.
- AMD/HIP, Metal, NPU, and other targets each require a separate capability
  contract, legalizer, lowering, runtime, artifact, correctness, and
  performance gate.
- Add operations and optimizations incrementally; do not introduce arbitrary
  C++ capture, implicit host/device copies, silent fallback, a general tensor
  framework, or overlapping undocumented IRs.

The long-term performance promise remains deliberately bounded: choose and
validate the best implementation available within the supported device
capability model and implemented search space.
