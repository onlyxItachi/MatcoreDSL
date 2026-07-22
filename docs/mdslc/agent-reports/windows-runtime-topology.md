# Windows runtime and topology lane

Date: 2026-07-22

Base: `091d74072a710389b4a8e9d51f696ad9773021e6`

Branch: `agent/windows-runtime-topology`

## Ownership

This lane changed only `compiler/lib/platform/**`,
`compiler/tests/platform/**`, and this report. It did not change the driver,
runtime ABI, package, root build, workflow, or legacy implementation.

## Implemented boundary

- clang-cl is normalized as the Clang compiler-capability family while its
  MSVC ABI identity remains `CompilerFrontendV1::clang_cl`. Under
  `_MSC_VER && _M_X64`, CPUID and XCR0 use `__cpuidex` and `_xgetbv`; the GNU
  `<cpuid.h>` branch cannot win accidentally.
- AVX2/FMA still requires XMM+YMM XSTATE. AVX-512 still requires opmask,
  ZMM-hi256, and hi16-ZMM XSTATE. Windows AMX remains fail-closed because this
  milestone has no validated Windows per-process tile-state permission
  mechanism.
- `WindowsCpuTopologySnapshotV1` is a portable, injectable normalization
  boundary. Linux tests cover deterministic Windows core, SMT, package, NUMA,
  and cache relationships without pretending that those tests are physical
  Windows validation.
- The physical Windows collector uses
  `GetLogicalProcessorInformationEx` for processor-core, package, cache, and
  `RelationNumaNodeEx` records. It returns a valid but incomplete topology on
  any malformed or incomplete relationship.
- Topology/affinity v1 deliberately supports one Windows processor group. A
  machine with more than one active group (normally more than 64 logical
  processors) is rejected as incomplete/unavailable. It is never flattened
  into colliding logical CPU IDs.
- Windows affinity authenticates the allowed set by intersecting
  `GetProcessAffinityMask` with `GetThreadGroupAffinity`, applies a one-CPU
  `SetThreadGroupAffinity` mask, and obtains the current CPU from
  `GetCurrentProcessorNumberEx`.
- `discover_host_cpu_topology_v1()` isolates native Linux/Windows collector
  selection for runtime consumers.
- Platform warning flags are compiler-family portable: MSVC-style builds use
  `/W4 /WX /permissive-`; GNU-style builds use
  `-Wall -Wextra -Wpedantic -Werror`.
- The synthetic sysfs fixture now includes a build-directory-derived suffix so
  simultaneous worktrees cannot race on one fixed `/tmp` directory.

The Win32 API choices follow the documented contracts for
[GetLogicalProcessorInformationEx](https://learn.microsoft.com/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex),
[NUMA_NODE_RELATIONSHIP](https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-numa_node_relationship),
[GetProcessAffinityMask](https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-getprocessaffinitymask),
[SetThreadGroupAffinity](https://learn.microsoft.com/windows/win32/api/processtopologyapi/nf-processtopologyapi-setthreadgroupaffinity),
and
[GetCurrentProcessorNumberEx](https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocessornumberex).

## Validation

Configured a fresh Release tree with Clang/LLVM 21.1.8, native frontend on,
bootstrap comparison on, and OpenBLAS deliberately off:

```text
cmake -S compiler \
  -B /home/hamza-usta/matcore-win-platform-build-agent-ujAS56 \
  -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=OFF \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release

cmake --build /home/hamza-usta/matcore-win-platform-build-agent-ujAS56 \
  --target matcore_platform_v1_test \
           matcore_cpu_capability_v2_test \
           matcore_cpu_topology_v1_test \
           matcore_thread_affinity_v1_test -- -j2

ctest --test-dir \
  /home/hamza-usta/matcore-win-platform-build-agent-ujAS56 \
  --output-on-failure -R '^platform\\.' -j1
```

Result: **4/4 passed**.

The existing persistent-executor consumers were also rebuilt and exercised:

```text
cmake --build /home/hamza-usta/matcore-win-platform-build-agent-ujAS56 \
  --target matcore_execution_context_test \
           matcore_parallel_packed_gemm_test \
           matcore_planner_v3_resources_test -- -j2

ctest --test-dir \
  /home/hamza-usta/matcore-win-platform-build-agent-ujAS56 \
  --output-on-failure \
  -R 'runtime\.cpu\.(execution_context|parallel_packed|planner_v3_resources)' \
  -j1
```

Result: **3/3 passed**. Repository hygiene and `git diff --check` passed.

A non-authoritative `_WIN32` syntax pass using a temporary WinSDK-shaped shim
also compiled `windows_cpu_topology_v1.cpp` and `thread_affinity_v1.cpp` with
Clang 21 and strict warnings. The shim is external and untracked. It is useful
only for branch coverage; it is not a Windows SDK, MSVC ABI, link, or runtime
validation result.

## Required hosted follow-up

No Windows SDK, MSVC libraries, or Windows runtime is installed on the Linux
validation host. Consequently, the `_WIN32` physical tests are **authored but
not yet compiled or executed**. The Windows CI lane must compile with the real
Clang 21/Windows SDK tuple and execute:

- physical capability discovery and XCR0 legality;
- single-group topology discovery and `RelationNumaNodeEx` parsing;
- child-thread affinity application and current-processor authentication;
- explicit multi-group fail-closed synthetic tests.

Physical Windows NUMA performance and machines with multiple processor groups
remain unvalidated. No such support is claimed.
