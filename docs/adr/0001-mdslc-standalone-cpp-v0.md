# ADR 0001: Standalone valid-C++ MDSLC bootstrap

- Status: Accepted for bootstrap v0
- Date: 2026-07-19
- Scope: New additive `compiler/` project

## Context

MatcoreDSL currently exposes a Python frontend, nanobind native extension,
runtime JIT/cache, KernelGraphIR and RegionIR, and CPU/NVIDIA/AMD lowering
components. Those components are useful implementation evidence and possible
future backend substrate, but they do not define the standalone compiler's
source model or packaging.

MDSLC needs to compile ordinary host C++ and explicit matrix operations from
the same translation unit into normal native artifacts. It must preserve C++
language semantics through Clang while creating an inspectable,
target-independent matrix representation for later planning and lowering.

This project is not a Triton reimplementation. Triton may be studied as prior
art, but its Python blocked-programming language and runtime are neither the
MDSLC source model nor a v0 dependency.

## Decision

### Source language and public API

Every `.mdsl` file is valid C++ source. `mdslc++` always tells Clang to parse
the extension as C++ with `-x c++`. Ordinary functions, classes, non-Matcore
templates, host control flow, standard-library calls, and I/O retain ordinary
C++ behavior.

The only public operation namespace is `matcore::mdsl`, reached through:

```cpp
#include <matcore/mdsl.h>

namespace md = matcore::mdsl;

md::gemm(
    md::out(C),
    A,
    B,
    md::policy{
        .target = md::target::cpu,
        .fallback = md::fallback::error,
    });
```

The output wrapper makes mutation, aliasing, initialization, synchronization,
and memory effects explicit. User declarations under `std` or `std::mdsl` are
forbidden.

The opt-in boundary has four layers:

1. `.mdsl` extension;
2. invocation through `mdslc++`;
3. inclusion of `<matcore/mdsl.h>` and use of canonical `matcore::mdsl`
   declarations;
4. Clang annotations such as `clang::annotate("matcore.op.gemm")`.

### Frontend recognition

The first frontend uses Clang LibTooling, AST matchers, and Rewriter. It runs
normal parsing and Sema, visits `CallExpr` nodes, requires `getDirectCallee()`,
resolves the canonical `FunctionDecl`, and reads the canonical declaration's
`matcore.op.*` annotation. It never recognizes an operation from source text.

Direct qualified calls and namespace aliases resolve to the same canonical
declaration and are supported. Calls in non-template free functions and
non-template class methods are in v0 scope. Ordinary host templates containing
no Matcore calls remain normal C++.

The frontend initially rejects Matcore calls involving unqualified/ADL lookup,
indirect calls, dependent templates, lambdas, macro expansions, header-origin
inline sites, constant evaluation, unsafe side effects, unsupported
rank/dtype/layout, invalid output mutability or lifetime, alias violations,
mixed residency, implicit migration, or unavailable targets. Rejection occurs
with an original-source diagnostic before source rewriting or object emission.

### Responsibility split

The compiler responsibilities are deliberately separated:

1. Clang parses valid C++ and performs C++ semantic analysis.
2. The MDSLC LibTooling frontend recognizes explicit matrix operations.
3. MDSLC emits and verifies Matcore IR.
4. MDSLC transformations, planning, legalization, scheduling, and runtime
   layers consume Matcore IR.
5. LLVM, NVVM, ROCDL, and host toolchains may be used downstream.
6. Target-specific code generation and runtime integration produce host/device
   artifacts.

Clang remains the C++ frontend and host compiler; it is not the entire
target-independent matrix optimizer.

### Bootstrap IR

Matcore JSON IR v0 is the first explicit frontend/compiler boundary. It is a
bootstrap contract, not a substitute for the future Matcore MLIR dialect.

Every IR document has a schema name, version, deterministic serialization,
translation-unit identity, verifier, source locations, and tests. Each captured
site records at least operation kind, canonical callee, stable site ID,
operands/output, dtype, rank, static or dynamic dimensions, layout/strides,
mutability, alias constraints, memory space, target policy, fallback policy,
effects, and original source location.

No second overlapping bootstrap IR may be introduced without a documented,
versioned conversion boundary.

### First artifact pipeline

The first complete vertical slice is:

```text
valid C++ .mdsl
  -> Clang parsing and Sema
  -> canonical annotated operation capture
  -> verified deterministic Matcore JSON IR v0
  -> exact CallExpr rewrite
  -> generated C ABI site stubs and CPU backend source
  -> ordinary clang++ compilation
  -> Linux relocatable partial link
  -> normal .o and executable/shared-library integration
```

With saved temporaries, one source produces inspectable `host.cpp`,
`matcore.json`, `sites.h`, `stubs.cpp`, `backend.cpp`, intermediate objects, and
the final relocatable object. Generated artifacts stay in the build tree.

### Runtime ABI and CPU backend

The compiler/runtime boundary is a versioned C ABI. Descriptors carry data
pointer, dtype, rank, dimensions, strides, memory space, mutability where
required, target/fallback policy, and status/error data. C++ templates and
exceptions do not cross the ABI.

The first backend is a synchronous reference GEMM for rank-2, row-major,
contiguous `f32` host matrices. It validates shapes and descriptors, requires
that output not alias either input, performs no hidden allocation or copy, and
returns explicit errors for unsupported inputs. A documented synchronous host
wrapper may translate returned failure status into a C++ exception.

CPU correctness and normal artifact integration precede performance work.
CUDA/cuBLAS may follow only after the CPU gates pass and must require explicit
device-resident descriptors and target policy. WGMMA, HIP, and Metal are not
v0 implementation goals.

### Build and packaging

The standalone project lives under `compiler/` and configures independently:

```sh
cmake -S compiler -B build-mdslc -G Ninja ...
cmake --build build-mdslc -- -j2
```

It has no runtime or build dependency on Python, nanobind, the legacy JIT
cache, or root CMake. It produces normal executables, relocatable objects,
shared libraries, and an installable CMake package. A Clang plugin is not a
v0 prerequisite, and MLIR GPU lowering is not a prerequisite for the first
`.mdsl -> .o` proof.

### Long-term lowering and capability model

The intended future boundary is:

```text
Clang capture
  -> Matcore high-level IR / Matcore MLIR dialect
  -> structured transformations
  -> Linalg / Tensor / MemRef / Vector / GPU where appropriate
  -> target-specific dialects and code generation
  -> LLVM / NVVM / ROCDL / target artifacts
```

Existing MLIR/backend work is preserved and may later be adapted behind an
explicit JSON-v0-to-Matcore-IR bridge. Frontend code must not couple directly
to Python, JIT caches, CUDA path assumptions, or legacy pass ordering.

A target is supported only when its capability record, legalizer, planner,
lowering, runtime, and validator can compile, link, execute, and pass declared
acceptance tests. Capability records must eventually cover architecture,
subgroup size, scalar/matrix dtypes, accumulator types, instruction families,
tile constraints, vector width, memory hierarchy, register limits, async copy,
synchronization, alignment, address spaces, workgroup limits, binary formats,
and runtime/driver availability.

Planning may select an optimized library, generic structured lowering,
generated vector/loop code, custom accelerator code, or an explicit error. The
performance promise is limited to: choose and validate the best implementation
available within the supported device capability model and implemented search
space.

## Rejected approaches

- A new C++ parser or C++-like custom language
- Textual call recognition or LLVM-IR pattern matching as the main frontend
- Arbitrary C++ capture
- A plugin-first or MLIR-GPU-first bootstrap
- Triton as a v0 dependency or Triton's language as the MDSLC source model
- Python or nanobind in the standalone compiler's normal execution path
- Silent target fallback, implicit host/device migration, or hidden copies
- A general tensor framework, fusion optimizer, autotuner, or custom matrix
  accelerator kernel in the bootstrap milestone

## Consequences

The bootstrap deliberately implements a narrow operation and descriptor set,
but each boundary is inspectable, versioned, and replaceable. Valid C++ and
post-Sema declaration identity reduce miscompilation risk. Rewriter imposes
strict source-range limitations, so unsupported constructs must be diagnosed
rather than accepted speculatively. JSON adds an intermediate serialization
step now while preserving a clear future conversion to structured Matcore IR.
