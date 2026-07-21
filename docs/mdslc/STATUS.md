# MDSLC native frontend status

Status date: 2026-07-21

- Native milestone base: `mdslc/bootstrap-v0` at
  `3e3fa5b2d1990e1c37870f8b2096fbda6128716b`
- Milestone 1 integration branch: `mdslc/native-libtooling-v1`
- Milestone 2 branch: `mdslc/matcore-ir-v1-cpu-planner`
- Milestone 2 pull request: `#5`

## Milestone 2 verdict

**Typed Matcore IR v1 and deterministic CPU GEMM planning pass local release
gates and independent review. PR #5 enforces the hosted gates before merge.**

The native driver now routes every authenticated `matcore::mdsl::gemm` through
a verified v0-to-v1 boundary. IR v1 carries typed shape, dtype, accumulation,
layout, stride, alignment, memory, mutability, effects, alias, synchronization,
policy, requirement, provenance, and exact source-range contracts. Only a
lossless canonical subset projects into the existing rewrite/codegen and v0
execution ABI.

The CPU runtime validates descriptors before discovering versioned host
capabilities. It evaluates a fixed reference/tiled/compiler-vectorized registry,
rejects illegal variants, applies saturating deterministic integer costs, emits
complete candidate and selected-plan diagnostics, and executes exactly the
selected lowering. The additive `matcore_runtime_plan_gemm_f32_v1` query and
installed `matcore-plan` tool expose the same decision without executing or
modifying output.

Current validation host capability record:

```text
x86_64; discovery complete; portable scalar f32, AVX2, FMA; 256 vector bits
```

Fresh Release with the Ubuntu 24.04 RapidJSON header passed 13/13 CTest tests
in 60.48 s before the additive object-artifact test. Fresh Debug passed the
final 14/14 suite in 63.76 s. The ASan/UBSan focused set
passed 7/7, and a separately instrumented generated GEMM printed
`MDSLC CPU GEMM PASS`. Five representative benchmark shapes passed independent
correctness and generous absolute regression guards. Results and exact
contracts are recorded in
`docs/mdslc/agent-reports/matcore-ir-v1-cpu-planner-final.md`.

The independent reviewer returned PASS after finding one P1: Debug/default
could select a vector-named implementation whose `-O0` body was scalar. The
fix preserves Release `-O3`, enables `-O2` only for Debug/default, and requires
function-local YMM packed-FMA instructions in an artifact regression test.

No coherent BLAS development package was installed, so the optional BLAS
adapter was not added and is not required. No accelerator, fusion, or
autotuning capability is claimed.

## Milestone 1 foundation

**Architecture proof passed for the standalone native CPU frontend/runtime
vertical slice.**

The default `.mdsl` operation path now uses an in-process Clang 21.1.8
LibTooling frontend, not AST JSON. It authenticates the resolved public header,
canonical declarations, exact annotations, public ABI, and SourceManager token
ranges, then reuses the verified JSON IR/codegen/C ABI/runtime pipeline.

No Python, nanobind, MLIR, legacy JIT, or bootstrap subprocess participates in
the default compiler/runtime path. The AST-JSON frontend remains explicit
compatibility/differential mode only and is never a silent fallback.

## Implemented pipeline

```text
valid C++ foo.mdsl
  -> mdslc++ (native frontend by default; forces -x c++)
  -> pre-extraction dependency scan and immutable input closure
  -> in-process ClangTool + PPCallbacks + parse/Sema
  -> direct resolved trusted-header identity and content/ABI authentication
  -> ASTMatcher + getDirectCallee() + canonical FunctionDecl
  -> exact AnnotateAttr("matcore.op.gemm") authentication
  -> SourceManager/Lexer source locations and call/argument token ranges
  -> deterministic, verified Matcore JSON IR v0
  -> exact call rewrite + generated sites/stubs/backend
  -> three ordinary Clang C++ objects
  -> clang++ -r combined relocatable object
  -> ordinary clang++ link against versioned libmatcore_runtime
  -> synchronous checked row-major f32 CPU GEMM
```

The main source and every non-system dependency are snapshotted before
extraction and checked after extraction, each generated compilation, linking,
and dependency publication. The rewritten host is compiled through a VFS
mapping at the original `.mdsl` path, preserving quote includes and diagnostic
context. User VFS/PCH/module injection is rejected. Semantic compile arguments
contribute to stable site identity; weak generated wrapper/backend definitions
allow equivalent deterministic sites to co-link across independent roots.

## Authentication and rewrite contract

- The main translation unit must directly resolve `matcore/mdsl.h` to the
  build/install package's trusted `FileEntry` unique identity.
- Clang's parsed header buffer must match the physical snapshot; the expected
  record/enum/field/type/value/default/signature ABI is checked after Sema.
- External macros may not alter public Matcore declarations.
- A GEMM call must resolve directly to the trusted canonical
  `matcore::mdsl::gemm`, have the supported signature, and contain exactly the
  expected non-inherited annotation. `out` is authenticated similarly.
- Namespace aliases work through canonical identity. Fake/copied/shadowed
  headers, user overloads, unqualified/ADL calls, indirect calls, templates,
  lambdas, macro-generated or header-originating sites, constexpr/unevaluated
  contexts, side-effectful arguments, and unsupported ABI/layout/policy cases
  fail with source diagnostics.
- Rewriter ranges are half-open SourceManager/Lexer token ranges from the same
  parsed source snapshot; no regular-expression call matching or manual token
  boundary estimation is used.

## Validation results

| Validation | Result |
| --- | --- |
| Final fresh Release, `/tmp/matcore-native-v1-final5-release.1R1Ecm` | 19/19 build steps; 8/8 CTest tests passed in 61.55 s |
| Final fresh Debug, `/tmp/matcore-native-v1-final5-debug.YZDSYz` | 19/19 build steps; 8/8 CTest tests passed in 62.94 s |
| Native focused frontend | 16 checks passed |
| Native primary frontend | 44 checks passed |
| Native/parity/adversarial core after final hardening | 536 checks passed |
| Native driver pipeline | 72 checks passed |
| Integration matrix | 63 passed, 0 failed, 6 future capabilities not counted |
| Strict `-Wall -Wextra -Wpedantic -Werror` build | build passed; full standalone CTest passed 8/8 |
| Native-disabled/bootstrap-enabled | 3/3; default failed, explicit bootstrap passed |
| Native-enabled/bootstrap-disabled | 4/4 tests passed |
| Installed native frontend checks | 7 checks passed |
| Runtime unit test | all CPU GEMM v0 cases passed |

Native/bootstrap parity compares every semantic IR field and generated
host/sites/stubs/backend output after excluding only the intentional producer
field. The native producer is `clang-libtooling-v1`; compatibility output is
`clang-ast-json-bootstrap-v0`. A deliberate mismatch self-test proves that the
comparator detects semantic drift.

### Sanitizers

`/tmp/matcore-native-v1-final5-sanitize.WdaSNP` used:

```text
-fsanitize=address,undefined -fno-sanitize=pointer-overflow
-fno-omit-frame-pointer
```

The supported focused set passed 4/4: native focused, native primary, native
core, and runtime. A separately sanitizer-instrumented generated GEMM under
`/tmp/matcore-native-v1-final5-artifacts.NM2iTj` executed
`MDSLC CPU GEMM PASS` with leak detection and no report. Its object contains
ASan and UBSan instrumentation symbols.

The full sanitizer CTest driver/package set is deliberately not counted:
configure-only sanitizer flags are not automatically forwarded into the child
consumer/final links those tests launch. This is a test-orchestration limitation,
not a claimed sanitizer pass. Pointer-overflow instrumentation is disabled only
for the known RapidJSON 1.1 compatibility expression in the explicit bootstrap
implementation.

### Install and consumer

Final fresh installation to
`/tmp/matcore native v1 final5 direct.wK4ayb/install prefix` produced:

- `bin/mdslc++` and native `bin/matcore-extract`;
- `include/matcore/{mdsl.h,runtime_c.h}`;
- versioned `libmatcore_runtime.so`;
- relocatable `MatcoreDSLConfig.cmake`, targets, version, and compile helper.

The final external consumer at
`/tmp/matcore native v1 final5 direct.wK4ayb/consumer build`
configured through `find_package(MatcoreDSL REQUIRED)`, built, ran, and returned
to a no-op Ninja build. The final Release installed-consumer test separately
verified regeneration after `.mdsl` and included-header edits. The prefix
deliberately contains spaces. Binary/text scans found no source or build-tree
absolute-path leak and no Python/nanobind dependency.

### CPU and artifact proof

The final saved artifact proof is
`/tmp/matcore-native-v1-final5-artifacts.3aSn01`.
`hello_host.mdsl` prints `5`. `gemm_v0.mdsl` prints:

```text
host-before
MDSLC CPU GEMM PASS
```

`file`, `readelf`, `nm -C`, and `ldd` confirm an ordinary ELF64 relocatable
combined object, ordinary generated C++ site/backend symbols, an unresolved
`matcore_runtime_gemm_f32_v0` boundary before final link, a normal ELF PIE after
ordinary Clang linking, and resolution through `libmatcore_runtime.so.0`.
`--save-temps` retains the host source, VFS overlay, verified JSON, sites header,
stubs, backend, three component objects, and combined object. A repeat build
produced byte-identical IR and generated sources; the JSON SHA-256 is
`afd693d72e2574d27aae53a8ccd50e975404bee6cf19425cc05f64739016d480`.
Verbose driver evidence invoked the native extractor and ordinary Clang
compilations; no AST-dump invocation or bootstrap marker appeared.

## Primary commands

```sh
cmake -S compiler -B /tmp/matcore-native-v1-final5-release.1R1Ecm -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-native-v1-final5-release.1R1Ecm -- -j2
ctest --test-dir /tmp/matcore-native-v1-final5-release.1R1Ecm \
  --output-on-failure -j1

/tmp/matcore-native-v1-final5-release.1R1Ecm/bin/mdslc++ \
  -std=c++20 --matcore-target=cpu --save-temps -c \
  compiler/examples/gemm_v0.mdsl \
  -o /tmp/matcore-native-v1-final5-artifacts.3aSn01/gemm_v0.o
/usr/bin/clang++-21 \
  /tmp/matcore-native-v1-final5-artifacts.3aSn01/gemm_v0.o \
  -L/tmp/matcore-native-v1-final5-release.1R1Ecm/lib \
  -lmatcore_runtime \
  -Wl,-rpath,/tmp/matcore-native-v1-final5-release.1R1Ecm/lib \
  -o /tmp/matcore-native-v1-final5-artifacts.3aSn01/gemm_v0
/tmp/matcore-native-v1-final5-artifacts.3aSn01/gemm_v0
```

## Legacy regression and devices

Selected legacy Python tests: 26 passed; 3 failed only because the legacy
`_matcore_native` extension is not built in this standalone worktree. A fresh
root CMake probe still fails at the known MLIR contract: root asks for 18.1.3,
while available MLIR configuration is 22.1.2. No legacy root build is claimed
green.

CPU execution is runtime-validated. NVIDIA RTX 4060/CUDA 13.3, AMD
`gfx1150`/ROCm 7.1, and `aie2p` NPU are detected only. CUDA, cuBLAS, HIP, NPU,
and all accelerator compiler paths were not attempted.

## Known limitations

- Linux, Ninja, Clang/LLVM 21.1.8, one `.mdsl` input, synchronous host-resident
  rank-2 row-major contiguous f32 GEMM only.
- `matrix_view` is a minimal host f32 view, not a general tensor framework.
- Driver-managed shared/static/PIE modes and opaque response/linker option
  forms remain rejected; emit `-c` and perform an ordinary explicit final link.
- Bootstrap remains compatibility-only and is not the semantic authority.
- GEMV, GEVM, ReLU-GEMM, BLAS selection, MLIR lowering, CUDA/cuBLAS, HIP,
  Metal, NPU placement, fusion, and autotuning are not implemented.

There is no remaining blocker to the declared standalone native CPU
architecture proof. Broader target, operation, optimization, and production
portability work remains future scope.
