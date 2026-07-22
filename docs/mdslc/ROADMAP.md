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
    independent review pass. Windows remains a portability seed, not a
    runtime-validated platform.

The architecture verdict is **passed for the standalone native CPU
frontend/runtime vertical slice**. The AST-JSON producer remains an explicitly
selected compatibility oracle only; it is never a fallback from native.

## Next three exact engineering tasks

1. **Add CPU capability record v2 and ISA multiversioning.** Distinguish
   hardware, OS-enabled state, compiler support, compiled implementation, and
   runtime validation. Add safely gated AVX-512 F32 without globally raising
   the runtime ISA and retain AVX2 when measured cost favors it.
2. **Add the persistent parallel CPU execution context.** Reuse workers,
   expose requested/actual threads and per-worker workspace, prevent nested
   provider oversubscription, and plan deterministic macro-tile distribution
   from topology and problem size. Discover Linux topology/NUMA with synthetic
   multi-node tests and no hidden migration.
3. **Define BF16/INT8 semantics and finish deferred Windows validation.** Add
   typed reference/oracle/C-ABI behavior before any hardware-gated VNNI/BF16
   or AMX candidate. After the Linux advanced-CPU gate passes, validate the
   existing compiler/runtime through clang-cl, COFF/PE, DLL/import-library,
   install/consumer, and a Windows CI ZIP artifact. Do not begin GPU work.

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
