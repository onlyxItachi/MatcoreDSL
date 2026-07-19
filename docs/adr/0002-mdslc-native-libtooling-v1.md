# ADR 0002: Native Clang LibTooling frontend v1

- Status: Accepted, implemented, and validated
- Date: 2026-07-19
- Scope: Additive standalone `compiler/` frontend

## Context

Bootstrap v0 proved `.mdsl` valid-C++ parsing, deterministic Matcore JSON IR,
source rewriting, generated C ABI code, normal object production, installation,
and CPU GEMM execution. Its explicitly labeled AST-JSON adapter could not be
the final authentication boundary because JSON AST is not the native Clang
declaration, preprocessing, annotation, and source-range API.

The audited Clang/LLVM 21.1.8 development surface is coherent and available.
The frontend gap can therefore be closed without MLIR, Python, nanobind, or a
change to the existing IR/codegen/runtime contracts.

## Decision

The supported default is an in-process Clang 21 LibTooling frontend:

```text
ClangTool + FixedCompilationDatabase
  -> PPCallbacks
  -> parse and Sema
  -> ASTMatcher / ASTConsumer
  -> canonical declaration authentication
  -> SourceManager/Lexer ranges
  -> existing Matcore JSON IR v0, verifier, rewrite, and codegen
```

The frontend requires the main `.mdsl` file to directly resolve
`matcore/mdsl.h` to the tool/package-owned public header. Trust uses the
resolved `FileEntry` unique identity, a physical content snapshot, equality
with the buffer Clang actually parsed, and semantic validation of the public
records, enums, fields, types, values, defaults, and operation signature.
External macro expansion that changes the public header ABI is rejected.
Copied, shadowed, remapped, or lookalike declarations are not trusted.

Each candidate call must have a direct callee. The frontend verifies its
canonical qualified name and signature, its declaration origin, and exactly
the expected non-inherited `AnnotateAttr("matcore.op.gemm")`. The canonical
`matcore::mdsl::out` wrapper is authenticated similarly. Namespace aliases are
accepted through canonical declaration identity; unqualified/ADL calls,
user overloads, indirect calls, templates, lambdas, macros, header-originating
sites, constexpr/unevaluated contexts, and unsafe arguments are rejected.

`SourceManager` and `Lexer::getLocForEndOfToken` provide original-file
locations and half-open token ranges. Rewriting replaces only the authenticated
main-file `CallExpr`; it does not search source text or estimate token lengths.
The driver compiles the rewritten bytes through an original-path VFS mapping so
quote includes and diagnostics retain the `.mdsl` source context.

The driver performs a dependency scan before extraction, snapshots the main
source and dependency closure, and verifies them after extraction, every
generated compilation, linking, and dependency publication. User VFS overlays,
PCH, and module injection are rejected in v1. Semantic compile context
contributes to site identity, and generated wrappers/backends use weak
definitions so equivalent deterministic sites can co-link across independent
source roots.

The AST-JSON frontend remains only as explicit
`--frontend=ast-json-bootstrap` compatibility/differential mode. `native` is
the default. A build without native support fails its default invocation; it
never silently falls back. Artifacts record `clang-libtooling-v1` or
`clang-ast-json-bootstrap-v0` as their producer.

## Toolchain contract

Native v1 requires one exact Ubuntu 21.1.8 tuple:

- `/usr/bin/clang-21` and `/usr/bin/clang++-21`;
- `/usr/bin/llvm-config-21`;
- LLVM and Clang headers and libraries below `/usr/lib/llvm-21`;
- `LLVMConfig.cmake` and `ClangConfig.cmake` for 21.1.8;
- imported `clang-cpp` and `LLVM` CMake targets.

Standalone CMake uses exact-version package discovery and fails clearly for a
missing header, target, compiler executable, or version mismatch. Native v1
does not use MLIR.

## Consequences

- Native parsing/Sema and source-manager semantics are authoritative; bootstrap
  parity does not justify preserving a bootstrap defect.
- The existing JSON IR v0 schema, verifier, rewrite/codegen implementation,
  C ABI, runtime, install layout, and CPU GEMM path remain shared.
- The validated architecture verdict is **passed for the standalone native CPU
  frontend/runtime vertical slice**.
- This decision does not claim CUDA, HIP, Metal, BLAS lowering, Matcore MLIR,
  GEMV/GEVM/ReLU-GEMM, heterogeneous placement, autotuning, or general
  production readiness.
