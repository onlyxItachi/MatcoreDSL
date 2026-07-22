# Milestone 5 public capability/context C ABI lane

- Date: 2026-07-22
- Base: `1f5f22983cd4146cd6befcb0aef84f3e32e664f2`
- Branch: `mdslc/m5-public-context-abi`
- Ownership: `compiler/lib/runtime/cpu_runtime.cpp`, one new focused public
  C-ABI test, and this report only

## Implemented boundary

The previously declared advanced CPU C ABI is now executable:

- `matcore_runtime_query_cpu_capabilities_v2`;
- `matcore_runtime_cpu_execution_context_create_v1`;
- `matcore_runtime_cpu_execution_context_query_v1` (additive integration
  declaration required; see below);
- `matcore_runtime_cpu_execution_context_destroy_v1`;
- `matcore_runtime_gemm_f32_context_workspace_size_v2`; and
- `matcore_runtime_gemm_f32_execute_context_v2`.

The opaque context owns a persistent `CpuExecutionContextV1` and immutable
capability/topology metadata. It owns no tensor, packing buffer, or GEMM
workspace. Planning consumes the existing capability-v2, topology-v1,
planner-v3, implementation-resource, and parallel-GEMM records directly. It
does not project back through planner v2 or duplicate planner legality.

All public input and output structures enforce exact ABI version, exact
structure size, zero flags, and zero reserved/output fields. Context affinity
and NUMA policy are fixed at creation and each execution must match them.
Descriptor, policy, shape, layout, alignment, alias, variant, context, and
workspace preflight happens before a selected implementation can mutate C.
Forced illegal candidates return their planner-v3 rejection reason and never
fall back.

The v3 report preserves all eight candidate decisions in registry order. The
public ABI's required-domain bit masks are a fixed lossless derivation from the
stable variant registry because the internal candidate record does not carry
four duplicate identical masks. OpenBLAS is deliberately reported as not
process-runtime-validated: its adapter can be legal and executed, but CPU
feature validation is not provider numerical validation.

## Runtime-validation evidence

Capability discovery freezes one deterministic, fail-closed process record.
It does not advertise AVX2/FMA or AVX-512F runtime validation from CPUID alone.
The first query performs guarded numerical tests of the exact packed AVX2 and
packed AVX-512 microkernel symbols after their existing CPU/OS-state guards.
Each test executes a 4x16x3 packed product and verifies every result. The
result is process-local static evidence used both by the public capability
record and planner-v3 resources. Unsupported hardware never enters the target
function.

The current physical host validated both packed AVX2/FMA and packed AVX-512
F32. The public test forced both single-thread variants. It then executed the
parallel AVX2 candidate twice through one opaque handle and observed
`execution_generation` advance from zero to two while
`persistent_worker_count` remained unchanged. A forced OpenBLAS execution did
not advance the native generation, proving that the provider was not submitted
inside the native worker pool.

## Validation

Toolchain: Clang/LLVM 21.1.8, C++20, Ninja `-j2`.

- Release public test with OpenBLAS 0.3.32: passed.
- Debug runtime compiled with `-Wall -Wextra -Wpedantic -Werror`: passed.
- ASan+UBSan, OpenBLAS disabled, leak detection and halt-on-error: passed.
- TSan, OpenBLAS disabled, halt-on-error: passed.
- Existing Release suite: 36/37 passed. The sole failure is the expected stale
  exact-export whitelist in
  `compiler/tests/integration/run_validation_matrix.py`; every runtime,
  planner, capability, topology, ISA-artifact, frontend, and installed
  consumer test passed.
- `git diff --check`: passed.
- Dynamic symbol inspection found only the intended 15 public C symbols and
  no C++ export.

Representative commands:

```sh
cmake -S compiler -B /tmp/matcore-m5-public-context-build -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-m5-public-context-build --parallel 2
ctest --test-dir /tmp/matcore-m5-public-context-build \
  --output-on-failure -j1

/usr/bin/clang++-21 -std=c++20 -O2 \
  -Wall -Wextra -Wpedantic -Werror -Icompiler/include \
  compiler/tests/runtime/public_context_c_api_test.cpp \
  -L/tmp/matcore-m5-public-context-build/lib \
  -Wl,-rpath,/tmp/matcore-m5-public-context-build/lib \
  -lmatcore_runtime -o /tmp/matcore-public-context-c-api-test
/tmp/matcore-public-context-c-api-test
```

## Integration requirements

The integration owner must add this declaration next to the existing public
create/destroy declarations in `compiler/include/matcore/runtime_c.h`:

```c
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_cpu_execution_context_query_v1(
    matcore_cpu_execution_context_v1 *context,
    matcore_cpu_execution_context_report_v1 *report)
    MATCORE_RUNTIME_NOEXCEPT;
```

Register `compiler/tests/runtime/public_context_c_api_test.cpp` as a C++20
executable linked to `MatcoreDSL::Runtime` and a CTest such as
`runtime.c_abi.public_context_v2`. Update the integration validation matrix's
exact dynamic-symbol list to all intentional public functions, including the
new query symbol. The test's temporary identical forward declaration may then
be removed, although retaining an identical redeclaration is valid C++.

This lane's base predates the integration branch's strict OS-affinity backend.
After cherry-pick, context creation must pass the complete deterministic
`worker_cpu_ids` selected from topology into `CpuExecutionContextConfigV1` and
preserve the affinity-application report. Do not drop that newer evidence or
revert to count-only placement. The direct planner-v3 candidate decisions,
guarded ISA self-tests, and caller-owned workspace behavior in this lane must
remain intact during that adaptation.

The current public report describes the fixed requested affinity/NUMA policy;
physical multi-node NUMA behavior remains unvalidated on this one-node host.
No AMX, BF16/VNNI acceleration, Windows runtime, or GPU claim is made.
