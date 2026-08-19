# Modern CPU Provider & Execution Taxonomy

**Investigation Scope**: Comprehensive categorization across 7 distinct numerical compute execution models  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. The Provider Taxonomy Matrix

| Provider Category | Representative Implementations | Specialization Timing | Packing Ownership | Scheduling & Microkernel Ownership | Post-Op Fusion Support | Primary Advantages |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Static Hand-Tuned BLAS** | OpenBLAS | Compile-Time (Hand-written assembly) | Hand-crafted assembly routines (`ICOPY`, `OCOPY`) | Explicit 5-loop cache nest + assembly kernels | **None** (Fixed CBLAS C ABI) | Maximum throughput on large standard GEMM ($N \ge 128$). |
| **2. Microkernel Framework** | BLIS | Compile-Time (Configurable C / Asm) | Object-based packing engine (`bli_pack.c`) | Formal control tree (`bli_gemm_cntl.c`) | Experimental control tree chaining | Modular, portable, cleanly decoupled macro/micro-kernels. |
| **3. Compiler-Cooperative** | Eigen | Compile-Time (C++ Templates) | Inlined C++ packing loops (`gebp_kernel`) | Expression templates + LLVM vectorizer | Inlined expression fusion | Zero external library dependencies; seamless C++ integration. |
| **4. JIT Microkernel Provider** | LIBXSMM | **Runtime (Dynamic Machine Code Gen)** | **Zero-Pack (Direct Streaming)** | JIT-emitted x86/ARM microkernel loops | **Native In-Register Fusion** | **Zero copy overhead; optimal for small/medium GEMMs ($N \le 128$).** |
| **5. Graph / Primitive Engine** | oneDNN | **Runtime (JIT + Graph)** | Managed tensor memory layout buffers | Graph scheduler + JIT microkernel dispatch | **Full Graph & Post-Op Fusion** | Deep learning operator chains; fused GEMM+Bias+ReLU. |
| **6. Opaque Vendor BLAS** | Intel oneMKL, AMD AOCL | Compile-Time + Dynamic Dispatch | Internal proprietary packing | Internal proprietary multi-core scheduler | Proprietary extensions | Highly optimized for specific vendor microarchitectures. |
| **7. Generic Compiler / DSL** | MLIR / LLVM / MatcoreDSL | Compile-Time (Polyhedral / Vector) | Structured MLIR / Caller Workspace | Structured `scf.forall` + `vector.contract` | **Native In-Register Writeback** | Portable; zero library coupling; custom DSL semantics. |
