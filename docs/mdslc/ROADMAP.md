# MDSLC roadmap

The roadmap remains gate-driven and additive to the legacy Python/JIT path.

## Completed bootstrap gates

1. **Standalone build skeleton — complete.** `compiler/` configures and builds
   independently of Python, nanobind, MLIR, and root CMake.
2. **Valid-C++ driver proof — complete.** `mdslc++` forces `.mdsl` through
   Clang as C++, supports host executable/object modes, is shell-free, and
   preserves compiler diagnostics/status.
3. **Public API and bootstrap capture — partial by dependency.** The public
   `matcore::mdsl` header, exact trusted-declaration recognition, Clang
   parsing/Sema, deterministic JSON, verifier, and negative context diagnostics
   pass. The AST-JSON implementation must still be replaced by LibTooling to
   authenticate the annotation payload and use native Clang AST APIs.
4. **C ABI, rewrite, and CPU GEMM — complete.** Exact source rewrite, stable
   multi-TU site symbols, generated stubs/backend, synchronous checked f32 GEMM,
   combined `.o`, ordinary external link, and independent-oracle execution pass.
5. **Installation and external consumer — complete.** Tools, headers, runtime,
   exported CMake targets, helper, external `find_package`, incremental rebuild,
   and clean installed-path checks pass.
6. **Adversarial validation — complete for implemented scope.** Fresh Release,
   Debug, generated-host/runtime ASan/UBSan, artifact, runtime,
   negative-source, IR-mutation, install, and independent review gates pass.
   Full extractor instrumentation separately exposes Ubuntu RapidJSON 1.1.0's
   null-pointer arithmetic and is not claimed clean.
7. **CUDA/cuBLAS — not attempted.** It remains optional after the exact
   LibTooling frontend gate is closed.

## Native LibTooling v1 in progress

The approved matching Clang 21.1.8 development surface is installed and
verified. The active gate replaces AST-JSON capture with native PPCallbacks,
post-Sema canonical declaration and `AnnotateAttr` authentication, and
SourceManager ranges while preserving JSON IR v0 and the validated CPU
artifact pipeline. Native becomes the default only after parity, install,
consumer, sanitizer, and adversarial review gates pass. Bootstrap remains an
explicit comparison mode and never a silent fallback.

## Next three exact engineering tasks

1. Implement the existing
   frontend interface with `getDirectCallee()`, canonical `FunctionDecl`,
   `AnnotateAttr` payload checks, and `SourceManager` ranges. Run the current 44
   frontend checks unchanged against it.
2. Make the LibTooling frontend the default and retain AST JSON only as an
   explicitly selected diagnostic fallback until parity is proven. Add
   compile-database support, include/source-manager tests, and remove the
   trusted-path inference workaround once annotation identity is native.
3. Define and test the single versioned JSON-v0-to-high-level-Matcore-IR bridge,
   including capability requirements and a CPU planner choice between the
   reference loop, structured/vector lowering, and an external optimized
   library. Do not begin custom GPU kernels in this step.

## Subsequent lowering milestones

- Expand the typed high-level IR only through versioned fields and verifier
  tests; do not introduce an overlapping undocumented IR.
- Add `gemv`, `gevm`, and then `relu_gemm` one operation at a time with explicit
  legality, effects, ABI, runtime, and negative tests.
- Add a CPU capability record and vector/library selection with correctness and
  bounded performance evidence.
- Add NVIDIA capability discovery and a synchronous, explicit
  device-resident cuBLAS library-call backend. No hidden migration or CPU
  fallback is allowed.
- Only after the library backend passes may scheduled GPU/Vector/NVGPU/NVVM
  experiments begin. WGMMA requires a detected legal architecture and actual
  runtime validation.
- AMD, HIP, and other accelerators require their own capability contract,
  legalizer, lowering, runtime, artifact, correctness, and performance gates.

A backend is supported only after it compiles, links, executes, validates
correctness, and passes its declared acceptance suite. The performance promise
remains: choose and validate the best implementation available within the
supported device capability model and implemented search space.
