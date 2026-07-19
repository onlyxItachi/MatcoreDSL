# ADR 0002: Native Clang LibTooling frontend v1

- Status: Accepted; implementation in progress
- Date: 2026-07-19
- Scope: Additive standalone `compiler/` frontend

## Context

The bootstrap CPU vertical slice uses Clang parsing and Sema through an
explicit AST-JSON adapter. That proved the artifact pipeline, but AST JSON does
not expose the complete native declaration, annotation, preprocessing, and
source-range contracts required by MDSLC.

Matching Clang and LLVM 21.1.8 development packages are now present. The
standalone frontend can therefore close this boundary without involving MLIR
or the legacy Python/JIT build.

## Decision

The supported default frontend is a native Clang 21 LibTooling executable. It
uses preprocessing callbacks to authenticate a direct include of the trusted
public header, then operates after parsing and Sema to inspect direct
`CallExpr` callees, canonical `FunctionDecl` identity, the exact
`AnnotateAttr("matcore.op.gemm")` payload, declaration signature, namespace,
and trusted declaration origin.

`SourceManager` and `Lexer` provide original-file locations and exact token
ranges. Unsafe macro, template, lambda, header-originating, indirect,
unqualified, side-effectful, or otherwise unsupported sites are rejected
before rewrite. The native frontend populates the existing Matcore IR v0
model; it does not introduce a second schema, verifier, code generator, ABI, or
runtime.

The AST-JSON adapter remains available only through an explicit
`--frontend=ast-json-bootstrap` compatibility/testing selection. Native is the
default, and missing native support is an error rather than a silent fallback.
Artifacts record their producer so incompatible frontend outputs cannot be
confused.

The driver supplies the trusted public-header identity from its build-tree or
installed-prefix context. An override is allowed only through a deliberate
test interface. Arbitrary user include paths cannot redefine which header is
trusted.

## Toolchain contract

The v1 binary uses one coherent Ubuntu package tuple at version 21.1.8:

- `clang-21` and `clang++-21`;
- `llvm-21-dev`;
- `libclang-21-dev`;
- `libclang-cpp21-dev`;
- LLVM and Clang CMake packages rooted below `/usr/lib/llvm-21`.

The standalone CMake build fails clearly on version mismatch or missing native
APIs. It does not depend on MLIR.

## Consequences

Native/bootstrap semantic parity remains a migration test, not permission to
preserve a bootstrap defect. Build-tree, installed-tree, source-rewrite,
runtime, and package validation must pass before the native frontend changes
the standalone CPU architecture verdict to passed.

CUDA, BLAS lowering, Matcore MLIR, additional operations, heterogeneous
placement, and capability planning remain outside this decision.
