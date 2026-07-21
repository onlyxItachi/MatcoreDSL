# Matcore IR v1 core implementation report

Date: 2026-07-21

Lane branch: `mdslc/ir-v1-core`

Ownership was limited to `compiler/lib/ir/**`, `compiler/tests/ir/**`, and this
report. No frontend, driver, codegen, runtime, package, public-header, or legacy
file was changed.

## Result

The lane adds a separate `matcore::mdslc::ir::v1` semantic IR without changing
the existing v0 DTO or its serializer/parser. The public integration surface is:

```cpp
bool v1::verify(const v1::Module &, std::string &error);
bool v1::fromV0(const ir::Module &, v1::Module &, std::string &error);
bool v1::projectToV0(const v1::Module &, ir::Module &, std::string &error);
std::string v1::serializeDeterministicJson(const v1::Module &);
bool v1::parseAndVerifyJson(std::string_view, v1::Module &,
                            std::string &error);
bool v1::probeJsonVersion(std::string_view, std::uint32_t &,
                          std::string &error);
```

`probeJsonVersion` recognizes only schema `matcore.ir` versions 0 and 1.
Exact v1 parsing accepts only version 1; it never retries a v0 or unknown
document through another parser.

## Exact JSON v1 schema

Serialization is deterministic, uses the following key order, two-space
indentation, and exactly one trailing newline:

```text
root
  schema: "matcore.ir"
  version: 1
  producer: "clang-libtooling-v1" | "clang-ast-json-bootstrap-v0"
  translation_unit
    identity: string
    source_file: *.mdsl
  operations: operation[]

operation
  site_id: "mc_" + 32 lowercase hex digits
  kind: "gemm"
  canonical_callee: "matcore::mdsl::gemm"
  source
    file, line, column, byte_offset
    byte_range { begin, end }       // half-open
  source_argument_ranges: { begin, end }[3 | 4]
  output: tensor
  operands: tensor[2]               // lhs, rhs
  accumulation_dtype: dtype
  requirements:
    ["rank2_gemm", "f32_arithmetic", "host_addressable",
     "synchronous_execution"]
  alias_requirements:
    [{ relation: "no_alias", between: ["output", "lhs"] },
     { relation: "no_alias", between: ["output", "rhs"] }]
  effects
    reads: ["lhs", "rhs"]
    writes: ["output"]
    synchronization: "synchronous"
  policy { target: "cpu", fallback: "error" }

tensor
  role: "output" | "lhs" | "rhs"
  expression: nonempty source expression
  dtype: "f16" | "bf16" | "f32" | "f64" | "i8" | "i32"
  rank: 2
  shape: scalar_expr[2]
  strides: scalar_expr[2]
  layout: "row_major_contiguous" | "column_major_contiguous" | "strided"
  mutability: "read" | "write" | "read_write"
  memory_space: "host" | "device"
  required_alignment_bytes: positive power of two

scalar_expr
  { kind: "static", value: positive uint64 }
  | { kind: "dynamic", symbol: C-identifier-like semantic name }
```

The enum surface deliberately reserves typed values useful to future IR
versions. The verified v1 GEMM contract implemented in this milestone is
strictly host-addressable f32 with f32 accumulation, synchronous effects, and
CPU/error policy. Detected CPU features and selected plans are intentionally
not stored in this target-independent IR.

## GEMM verifier

The verifier checks:

- module producer, `.mdsl` source identity, canonical unique site IDs, sorted
  non-overlapping calls, and exact source/argument ranges;
- direct canonical GEMM identity and ordered output/lhs/rhs roles;
- rank, shape/stride cardinality, static/dynamic scalar well-formedness,
  natural alignment, and contiguous-layout stride equations;
- exact symbolic `M`, `K`, and `N` equality rather than inferred unification;
- equal element dtypes and legal accumulation pairs, with the current semantic
  requirement set constraining the executable contract to f32/f32;
- write-only output, read-only inputs, explicit output/input non-aliasing,
  ordered read/write effects, synchronous execution, and CPU/error policy;
- the exact ordered semantic-capability requirement set, which is separate
  from downstream detected CPU capability bits.

Every JSON object is exact-member parsed. Unknown fields, enum values, schema
versions, malformed numbers, malformed source ranges, and conflicting semantic
relationships fail with a diagnostic and no partially returned module.

`compiler/tests/ir/gemm_capture.v1.golden.json` is the external reviewed byte
contract for the canonical native GEMM fixture. Tests require direct serializer
equality with that file and parse/re-serialize the golden byte-for-byte.

## Strict v0/v1 boundary

`fromV0` first calls the unchanged v0 verifier. It upgrades only verified v0
and reconstructs the semantics that v0 serialization guarantees:

```text
lhs: [m, k], strides [k, 1]
rhs: [k, n], strides [n, 1]
out: [m, n], strides [n, 1]
host f32, 4-byte minimum alignment, row-major contiguous
```

It preserves producer, translation unit, source file, site ID, exact source
location/ranges, source expressions, policy, and operation order.

`projectToV0` first verifies v1 and then requires precisely that canonical
dynamic host/f32/row-major/4-byte subset. Static shapes, stronger alignment,
column-major or general strided layout, or any other non-v0-representable
property fail instead of being erased. The existing v0 golden completes a
byte-identical v0 -> v1 -> v0 roundtrip.

## Validation evidence

Toolchain: `/usr/bin/clang-21` and `/usr/bin/clang++-21`, version 21.1.8,
with coherent LLVM/Clang CMake packages under `/usr/lib/llvm-21`.

Focused commands:

```sh
cmake -S compiler -B build-ir-v1 -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-ir-v1 --target matcore_ir_v1_tests -- -j2
ctest --test-dir build-ir-v1 -R '^ir\.v1\.core$' --output-on-failure -V
```

Focused result: `Matcore IR v1 PASS: 170 checks`; 1/1 CTest passed.

The checks cover deterministic JSON, exact-version dispatch, source provenance,
dynamic and static shapes/strides, row/column/general layouts, alignment,
dtype/accumulation, semantic requirements, effects, aliases, mutability,
invalid v0 rejection, unknown fields/versions/requirements, and every tested
lossy projection boundary.

During final review, the integration owner supplied an ownership-compatible
expansion of verifier/parser negative tests in this lane. It was inspected,
preserved, committed with the external golden, and rerun rather than discarded.

Full standalone commands:

```sh
cmake --build build-ir-v1 -- -j2
ctest --test-dir build-ir-v1 --output-on-failure -j1
```

Initial full build result: 17 build steps completed. Final no-op rebuild plus
regression result: 9/9 CTests passed in 63.01 seconds,
including the native frontend suites, driver pipeline, 32-second integration
matrix, relocated installed consumer, v1 core, and existing CPU GEMM runtime.

An additional Clang 21 `-Wall -Wextra -Wpedantic -fsyntax-only` pass over all
three v1 implementation translation units completed without diagnostics.

## Commits

1. `be9d14f` — `feat(ir): add typed Matcore IR v1 semantics`
2. `be43239` — `feat(ir): add deterministic Matcore IR v1 JSON`
3. `46b0dfe` — `test(ir): prove strict Matcore IR v0 and v1 boundary`
4. `540c547` — `test(ir): strengthen v1 provenance and version dispatch`
5. `bf541e8` — `test(ir): freeze reviewed Matcore IR v1 golden`

## Limitations

- This lane does not integrate v1 into the frontend, driver, planner, runtime,
  package, or installed API; those are separate owned lanes.
- The verified executable semantic contract remains host f32 rank-2 GEMM.
  Other enum values are typed for explicit future evolution but currently
  fail the semantic requirement contract.
- Dynamic scalar symbols express equality, not general affine expressions.
- v0 projection intentionally supports only the original dynamic contiguous
  ABI subset. This is a safety property, not an implicit lowering service.
- No CPU feature discovery, variant selection, performance measurement, BLAS,
  GPU, MLIR, fusion, or autotuning work was performed here.
