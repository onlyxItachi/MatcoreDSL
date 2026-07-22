# MDSLC native frontend status

Status date: 2026-07-22

- Rewritten canonical `main` baseline:
  `6a5b931baa6ec1136fb3c0471f515bd666a23981`
- Milestone 4 integration branch: `mdslc/cpu-performance-v1`
- Milestone 4 GitHub milestone: `#2`
- Milestone 4 umbrella issue: `#8`
- Milestone 1 rewritten checkpoint tag: `mdslc-native-cpu-proof-v1`
- Milestone 2 is preserved in rewritten mainline checkpoint tag:
  `mdslc-mainline-cpu-proof-v2`
- Milestone 3 mainline pull request: `#6`, merged normally into `main`; the
  later controlled history sanitation preserved its source tree while
  remapping commit IDs.
- Milestone 3 tracker: GitHub milestone `#1`, completed

## Milestone 4 CPU performance foundation

**The implementation, local acceptance matrix, calibration, and independent
review pass for the declared single-thread Linux host scope. Publication still
requires the normal GitHub PR/check/merge/tag gate.**

Planner v2 evaluates five stable implementations:

```text
cpu.reference.f32.v1
cpu.tiled.f32.v1
cpu.compiler-vectorized.avx2-fma.f32.v1
cpu.external.openblas.f32.v1
cpu.native-packed.avx2-fma.f32.v1
```

OpenBLAS 0.3.32 is optional, authenticated through LP64 CBLAS, controlled with
the provider's process-local thread API, and bounded by the provider-reported
thread ceiling before planning. The native packed engine uses caller-owned
64-byte-aligned workspace, MC=128, NC=256, KC=256, MR=4, and NR=16. Its exact
microkernel object contains YMM packed-FMA instructions; the rest of the
runtime remains generic and capability-gated.

The existing one-shot C ABI remains compatible. Additive v1 APIs query and
execute with explicit caller workspace and support caller-owned prepacked B.
Insufficient/misaligned/overlapping storage and forced illegal providers or
ISAs fail before output mutation. No hidden allocation, host/device copy, or
silent fallback was added.

`matcore-bench` freezes the JSON benchmark contract and distinguishes complete
one-shot, reused workspace, prepacked B, and diagnostic-only packed-compute
intervals. Raw results remain ignored outside Git. On the pinned validation
host, the 30-shape deterministic calibration produced median regret 1.000,
p95 1.124, maximum 1.132, and no choice above 2.0. Native packed beat the prior
compiler-vectorized candidate on 27/30 shapes; OpenBLAS was fastest on 26/30.
These are host/provider-specific observations, not universal or BLAS-parity
claims.

Independent exact-tip validation passed fresh Release 27/27, Debug 27/27,
focused ASan/UBSan 8/8 with repeated benchmark smoke, OpenBLAS-disabled 5/5,
package/install checks, seven-symbol C ABI inspection, exact AVX2/FMA
disassembly, repository hygiene, and a fresh `.mdsl -> ELF .o -> executable`
run printing `MDSLC CPU GEMM PASS`. The review resolved four medium findings:
provider-thread overcommit, a misleading compute-only mode, double-live
one-shot allocation/memory accounting, and flaky timing-smoke acceptance.

Evidence is in ADRs 0006/0007,
`docs/performance/cpu/milestone-4-single-thread-calibration-2026-07-22.md`, and
`docs/mdslc/agent-reports/m4-final-adversarial-review.md`.

Windows has only the versioned portability seed in this milestone. No Windows
compiler/runtime/package validation, parallel runtime, AVX-512, BF16, INT8,
AMX, real multi-node NUMA, GPU, or autotuning support is claimed.

## Milestone 3 mainline checkpoint

**Local history-preserving consolidation passed; hosted PR checks and final
merge remain the publication gate.**

The integration branch starts from `origin/main`, contains the historical
`feature/device-resident-tensors` line and the complete reviewed MDSLC lineage,
and uses ordinary merge commits. It does not rebase, squash, force-update, or
delete either history. The `compiler/` tree is byte-identical to the accepted
Milestone 2 commit.

Fresh committed-tree Release and Debug standalone suites each passed 14/14.
A coherent temporary MLIR 18 legacy build linked successfully; the meaningful
legacy regression set passed 68 pytest-compatible cases, 4 CUDA graph cases,
the CPU dtype/shape matrix, 24 elementwise GPU cases, and 7 softmax GPU cases.
The machine's prior MLIR 22 package surface was restored after that proof.

The integration decision and full evidence are recorded in
`docs/adr/0004-mdslc-mainline-history-consolidation.md` and
`docs/mdslc/agent-reports/mainline-consolidation-validation.md`.

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

The final fresh Release suite passed 14/14 CTest tests in 62.11 s, and the
final fresh Debug suite passed 14/14 in 63.14 s. The ASan/UBSan focused set
passed 9/9, and a generated executable built through the same instrumented
pipeline printed `MDSLC CPU GEMM PASS`. Runtime tests passed an independent
double-precision oracle; five representative benchmark shapes compared every
legal implementation with the forced reference implementation and passed
generous absolute regression guards. Results and exact contracts are recorded
in `docs/mdslc/agent-reports/matcore-ir-v1-cpu-planner-final.md`.

Independent review found and reproduced implementation and evidence defects,
including a vector-named Debug implementation whose `-O0` body was scalar.
The fixes preserve Release `-O3`, enable `-O2` only for Debug/default, require
function-local YMM packed-FMA instructions, and suppress vector capability
when sanitizer instrumentation scalarizes that body. All confirmed
high/medium findings were revalidated before final sign-off.

At the Milestone 2 checkpoint no coherent BLAS development package was
available, so no adapter was part of that historical gate. Milestone 4 later
added the optional authenticated OpenBLAS implementation described above.

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
  -> deterministic, verified Matcore JSON IR v0 compatibility capture
  -> explicit upgrade to typed, verified Matcore JSON IR v1
  -> loss-checked projection into existing rewrite/codegen
  -> exact call rewrite + generated sites/stubs/backend
  -> three ordinary Clang C++ objects
  -> clang++ -r combined relocatable object
  -> ordinary clang++ link against versioned libmatcore_runtime
  -> checked CPU capability discovery and deterministic plan selection
  -> deterministic planner-v2 selection with explicit resource diagnostics
  -> selected reference/tiled/compiler-vectorized/OpenBLAS/native-packed GEMM
  -> optional caller-owned workspace or prepacked-B execution
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
- GEMV, GEVM, ReLU-GEMM, AVX-512, parallel-native execution, BF16/INT8, AMX,
  Windows runtime validation, MLIR lowering, CUDA/cuBLAS, HIP, Metal, NPU
  placement, fusion, and autotuning are not implemented.

There is no remaining blocker to the declared standalone native CPU
architecture proof. Broader target, operation, optimization, and production
portability work remains future scope.
