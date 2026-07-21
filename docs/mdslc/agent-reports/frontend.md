# Clang frontend agent report

## Ownership and scope

This agent owned only:

- `compiler/include/matcore/mdsl.h`
- `compiler/lib/ir/`
- `compiler/lib/frontend/`
- `compiler/tools/matcore-extract/`
- `compiler/tests/frontend/`
- this report

No legacy source, root build file, driver, runtime, integration branch, package,
or system configuration was modified.

## Result

The branch contains an extraction-only bootstrap frontend with a replaceable
`Frontend` interface. It executes `/usr/bin/clang++-21` without a shell, forces
`-x c++`, runs Clang parsing and Sema, captures one bounded structural JSON AST,
recognizes the resolved annotated Matcore declaration, verifies Matcore IR v0,
and emits deterministic JSON.

This is explicitly named `clang-ast-json-bootstrap-v0`. It is not presented as
LibTooling. The planned LibTooling implementation can replace the concrete
frontend without changing the IR structs or command-line tool.

## Toolchain blocker and fallback boundary

All installed Clang CMake packages (17, 18, 20, 21, and 22) are incomplete for
LibTooling: frontend headers and static libraries are absent, and
`find_package(Clang)` fails on missing `libclangBasic.a`. The coherent repair
tuple is LLVM/Clang 21.1.8 plus the not-currently-installed matching
`libclang-21-dev` package. No package was installed.

Clang 21 JSON exposes `AnnotateAttr` presence but does not serialize its string
payload. It also omits the semantic qualifier object. The bootstrap therefore:

1. maps `CallExpr` callee `DeclRefExpr.referencedDecl.id` to the complete
   `FunctionDecl`;
2. requires `AnnotateAttr` presence, canonical `matcore::mdsl` declaration
   context, controlled `matcore/mdsl.h` origin, and the exact supported
   signature;
3. determines operation kind from that semantic declaration, never from call
   source text;
4. only after semantic recognition, examines the callee source range to enforce
   qualified-call syntax and safe non-macro rewriteability.

The literal annotation payload cannot be authenticated in this fallback. That
is a recorded limitation, not a fabricated capability.

Clang JSON also compresses repeated source filenames across sibling nodes. The
walker carries the last emitted source filename through the AST stream, which
was necessary for header-origin diagnostics to point at `header_call.h` rather
than the including `.mdsl` file.

## Public bootstrap API

`<matcore/mdsl.h>` provides:

- `matcore::mdsl::matrix_view`, a non-owning host-resident contiguous row-major
  f32 view with `data`, `rows`, and `columns`;
- explicit `out(matrix_view&)`, with const and rvalue overloads deleted;
- `policy`, `target::{cpu,cuda}`, and `fallback::error`;
- annotated `matcore::mdsl::gemm`.

The frontend currently accepts only `target=cpu` and `fallback=error`. The
descriptor owns no memory and performs no allocation or host/device copy. A
generated stub can map it to the runtime C descriptor as rank 2, dimensions
`{rows, columns}`, and strides `{columns, 1}`. Runtime pointer-level alias
checking remains mandatory because the frontend only catches syntactically
identical output/input descriptor expressions.

## CLI

```text
matcore-extract \
  --input FILE.mdsl \
  --ir-out FILE.json \
  [--clang /usr/bin/clang++-21] \
  [--ast-byte-limit N] \
  [--verbose] \
  -- [clang++ placeholder] COMPILE_ARGS
```

A bare `clang++` after `--` is treated as the requested command-shape
placeholder while the coherent default remains `/usr/bin/clang++-21`.
Absolute or versioned Clang paths are explicit overrides. Output-producing,
plugin, preprocessing-only, dependency-generation, and conflicting `-Xclang`
arguments are rejected. Process execution uses `fork` plus `execvp`, not a
shell.

IR output is published through a same-directory temporary file and rename so a
partial JSON document is never exposed as a successful result.

## Accepted and rejected constructs

Verified positive cases include ordinary host-only C++, direct fully qualified
GEMM, namespace alias `md::gemm`, explicit CPU/error policy, a non-template
class method, two distinct stable site IDs, and deterministic golden JSON.

Verified rejection cases include unqualified/ADL-style calls, indirect function
references, templates, lambdas, macro expansions, header-origin calls,
side-effectful input expressions, syntactic output aliasing, CUDA policy, const
output, temporary output, constexpr execution, and `std::mdsl`.

Each successful operation records schema/version, producer, translation-unit
identity, canonical callee, original file/line/column/byte offset, output and
operands, f32/rank-2 dynamic shapes, contiguous layout/strides, memory space,
mutability, alias requirements, effects, synchronous semantics, and policy.

## Validation evidence

Manual strict build:

```bash
/usr/bin/clang++-21 -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -Icompiler/lib/frontend -Icompiler/lib/ir \
  compiler/lib/ir/matcore_ir.cpp \
  compiler/lib/frontend/ast_json_frontend.cpp \
  compiler/tools/matcore-extract/main.cpp \
  -o /tmp/matcore-extract-front
```

Result: passed.

Header portability syntax proof:

```bash
g++ -x c++ -std=c++20 -Icompiler/include -fsyntax-only \
  compiler/tests/frontend/gemm_capture.mdsl
```

Result: passed.

Debug and Release standalone frontend harnesses:

```bash
cmake -S compiler/tests/frontend -B /tmp/matcore-frontend-build -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/matcore-frontend-build -- -j2
ctest --test-dir /tmp/matcore-frontend-build --output-on-failure -j1

cmake -S compiler/tests/frontend -B /tmp/matcore-frontend-release -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-frontend-release -- -j2
ctest --test-dir /tmp/matcore-frontend-release --output-on-failure -j1
```

Results: both passed, `1/1` CTest; the encapsulated runner reported 20 checks
passed (4 positive cases, 13 negative cases, golden determinism, repeat-byte
determinism, and distinct site IDs).

ASan-only validation:

```bash
cmake -S compiler/tests/frontend -B /tmp/matcore-frontend-asan -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address'
cmake --build /tmp/matcore-frontend-asan -- -j2
ASAN_OPTIONS=detect_leaks=1 \
  ctest --test-dir /tmp/matcore-frontend-asan --output-on-failure -j1
```

Result: passed, `1/1`, 1.20 seconds.

Combined ASan/UBSan is not reported as passed. Debian RapidJSON 1.1 triggers
UBSan null-pointer-arithmetic reports in
`/usr/include/rapidjson/internal/stack.h:117`; external LLVM symbolization then
stalled. The two test process trees started by this agent were explicitly
terminated. An ASan-only run completed cleanly afterward.

## Commits

1. `5812f2e` `feat(mdslc): add annotated C++ matrix API`
2. `6bf244d` `feat(frontend): emit deterministic Matcore IR v0`
3. `1b840e6` `test(frontend): cover structural extraction contracts`

## Remaining limitations and handoff

- No source rewrite or generated sites/stubs exists in this branch.
- JSON AST capture is memory-heavy. The tool retains one bounded, in-situ DOM
  and defaults to a 512 MiB AST-input cap; standard-library-heavy translation
  units remain expensive. LibTooling is the required durable replacement.
- Only direct f32 rank-2 host GEMM is represented.
- Shapes are dynamic in IR and require runtime verification.
- Site IDs use a stable path/offset/kind FNV-1a hash; a wider installed-product
  symbol scheme should be considered before a multi-TU ABI freeze.
- Only Linux/POSIX subprocess execution is implemented.
- Python is used only by the test runner, never by the extractor execution
  path.
- The integration CMake project must add the IR, frontend, and tool
  subdirectories; this agent did not edit the assigned-out top-level compiler
  CMake file.
