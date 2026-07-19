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
4. **Verified bootstrap IR — complete for v0.** Deterministic Matcore JSON IR
   v0 has a shared serializer, verifier, source locations, conversion boundary,
   and native/bootstrap differential tests.
5. **Rewrite, C ABI, and CPU GEMM — complete for v0.** Generated
   sites/stubs/backend, versioned C runtime, synchronous checked f32 GEMM,
   combined `.o`, ordinary final link, and independent-oracle execution pass.
6. **Installation and consumer — complete.** Tools, headers, runtime, CMake
   package/helper, fresh-prefix external consumer, dependency regeneration,
   relocation, and no-op rebuild pass.
7. **Adversarial validation — complete for the declared slice.** Release,
   Debug, supported sanitizer scope, native/bootstrap parity, fake-header and
   annotation attacks, source/dependency races, artifact inspection, package,
   and legacy smoke gates have reproducible evidence.

The architecture verdict is **passed for the standalone native CPU
frontend/runtime vertical slice**. The AST-JSON producer remains an explicitly
selected compatibility oracle only; it is never a fallback from native.

## Next three exact engineering tasks

1. **Freeze the post-JSON high-level IR boundary.** Define one versioned
   conversion from verified JSON v0 into a typed target-independent Matcore IR,
   including dynamic dimensions, dtypes, layouts/strides, memory space,
   effects, alias requirements, policy, synchronization, source locations, and
   capability requirements. Add verifier and round-trip tests before adding an
   operation.
2. **Introduce a CPU capability/planning contract.** Model detected scalar and
   vector capabilities and make the planner choose explicitly among the
   validated reference loop, a structured/vector implementation, and an
   external optimized-library implementation. Each choice needs legality,
   correctness, artifact, and bounded performance evidence; no silent fallback
   or global-performance claim.
3. **Add an explicit NVIDIA library-call milestone only after task 2.** Require
   device-resident descriptors, target/capability validation, synchronous
   cuBLAS execution, no hidden migration, no mixed residency, and CPU-oracle
   correctness. Treat unavailable architectures as errors or clearly labeled
   compile-only results, never runtime support.

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
