# Frontend rewrite and code-generation agent report

## Ownership and baseline

- Role: Clang frontend rewrite/code-generation follow-up.
- Worktree: `/home/hamza-usta/MatcoreDSL-wt-rewrite`.
- Branch: `mdslc/frontend-rewrite-v0`.
- Integrated starting SHA: `06e99b0593793b5fd83ccb441305eb72c45cddf3`.
- Write ownership was limited to `compiler/lib/frontend/**`,
  `compiler/lib/ir/**`, `compiler/lib/codegen/**`,
  `compiler/tools/matcore-extract/**`, `compiler/tests/frontend/**`, and this
  report.

No driver, runtime, public header, root or standalone top-level CMake file,
legacy source, example, install rule, or integration-branch file was changed.

## Result

The extraction boundary now records exact, half-open byte ranges for every
validated `CallExpr` and each explicit source argument. Code generation first
runs the Matcore IR verifier, validates every range against the original source
buffer, and then replaces call ranges in descending byte order. It does not
search for textual spellings.

Generated host source begins with its site declaration include and a
`#line 1 "<original.mdsl>"` directive. A validated call is rewritten to a
stable declaration in `matcore::mdsl::detail`, while the exact argument
expression slices are retained. Three-argument calls use a default
`policy execution_policy = {}` declaration; explicit policies remain explicit.

The optional generated-output CLI group is all-or-none:

```text
--rewrite-out FILE --sites-out FILE --stubs-out FILE --backend-out FILE
```

Together with `--ir-out`, it emits deterministic host, JSON IR, sites header,
stub source, and backend source files. All contents are generated and verified
before publication, and each file is published through a same-directory
temporary rename. The five destinations must be distinct real paths. A
translation unit with no Matcore operations still emits valid, nonempty C++
files.

The generated C++ stubs convert the non-owning `matrix_view` values to stack
`matcore_tensor_desc_v0` descriptors and the source policy to
`matcore_policy_v0`. Each site calls a site-specific `extern "C" noexcept`
backend entry that accepts only C ABI structures and forwards to
`matcore_runtime_gemm_f32_v0`. No C++ template or exception crosses this ABI.
The C++ site wrapper synchronously converts only a returned error status to
`std::runtime_error`; its message is prefixed with the original
`.mdsl` file, line, and column.

The tool also provides bounded standalone verification:

```text
matcore-extract --verify-ir FILE.json
```

It parses the bootstrap JSON contract and runs the same structural verifier
used after extraction. Malformed JSON and schema or version mismatches fail
with a nonzero status and a direct diagnostic.

## Validation evidence

Strict direct build:

```sh
/usr/bin/clang++-21 -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -Icompiler/lib/frontend -Icompiler/lib/ir -Icompiler/lib/codegen \
  compiler/lib/ir/matcore_ir.cpp \
  compiler/lib/frontend/ast_json_frontend.cpp \
  compiler/lib/codegen/codegen.cpp \
  compiler/tools/matcore-extract/main.cpp \
  -o /tmp/matcore-extract-rewrite
```

Result: passed with Clang 21.1.8 and no diagnostics.

Narrow frontend/code-generation suite:

```sh
python3 compiler/tests/frontend/run_frontend_tests.py \
  --extractor /tmp/matcore-extract-rewrite
```

Result: `frontend tests: 32 checks passed`.

The suite checks the existing positive and negative semantic cases plus:

- deterministic JSON and four deterministic generated-source goldens;
- exact byte determinism across repeated five-file generation;
- successful zero-operation generation;
- clean rejection of a partial output group;
- clean malformed, schema-mismatched, and version-mismatched IR rejection;
- strict independent compilation of generated host, stubs, and backend with
  `-Wall -Wextra -Wpedantic -Werror`;
- successful `clang++-21 -r` combination into a normal relocatable object.

Clean out-of-tree Debug build and full harness:

```sh
cmake -S compiler/tests/frontend \
  -B /tmp/matcore-frontend-rewrite-build.7EBSnz -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/matcore-frontend-rewrite-build.7EBSnz -- -j2
ctest --test-dir /tmp/matcore-frontend-rewrite-build.7EBSnz \
  --output-on-failure -j1
```

Result: all eight build steps completed; CTest passed `1/1` in 1.44 seconds.

Manual artifact inspection before the committed test harness also identified
the combined output as an x86-64 ELF relocatable object. `nm -C` showed the
generated C++ call-site symbol, the site-specific C backend symbol, and the
expected unresolved `matcore_runtime_gemm_f32_v0` reference for later runtime
linkage. Default-policy and zero-operation generated sources compiled
warning-clean.

## Commits

1. `b290ca05c744a5fd762861c5a43f374b89c91061` —
   `feat(frontend): generate rewrite and CPU call-site sources`
2. `f7ea8b6a72a4441c24230c78adebf5d12b241a45` —
   `test(frontend): verify IR and generated artifacts`

## Deliberate limits and integration notes

- The underlying frontend remains the explicitly labeled bounded Clang 21
  AST-JSON fallback because matching LibTooling development headers and
  libraries are not installed. The rewrite consumes only already validated
  semantic results; this does not remove the documented LibTooling migration.
- The CLI generates all outputs in memory before writing, but publication of
  five separate filesystem destinations cannot be globally atomic.
- Exact argument expressions are preserved, while inter-argument whitespace
  and comments are normalized at the rewritten call boundary.
- Generated sources use the sites header basename. The integrating driver must
  place the generated sources together or provide the matching include path.
- This branch proves generated-source compilation and relocatable linking. The
  integration owner remains responsible for wiring these outputs through
  `mdslc++`, linking the already implemented runtime, and executing the final
  CPU GEMM example.
