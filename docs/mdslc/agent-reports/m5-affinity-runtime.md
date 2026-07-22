# Milestone 5 worker-affinity lane

## Ownership and result

This lane owned only the thread-affinity platform backend, the internal
`CpuExecutionContextV1` configuration/implementation, focused platform and
execution-context tests, and this report. It did not edit the public C ABI,
planner, tools, benchmark, packaging, or CMake files.

Commit `7972a7f` adds strict, caller-selected worker affinity to the persistent
CPU pool:

- callers provide one unique logical CPU ID per actual worker;
- each worker applies its assignment once, before context creation returns;
- the Linux backend uses `pthread_setaffinity_np` in an isolated platform
  translation unit;
- unsupported platforms report `unavailable` and never claim a no-op success;
- any incomplete application tears down the entire new context;
- creation and live-context reports distinguish requested, applied, complete,
  unavailable, failed, and partially applied states;
- diagnostics preserve the first failed worker, CPU ID, and platform error;
- the report explicitly states that no NUMA allocation, first-touch, page
  migration, binding, or interleaving was performed;
- existing unbound contexts, deterministic cyclic task ownership, persistent
  worker reuse, error propagation, reentrant-operation rejection, provider
  nesting protection, and idempotent shutdown remain intact.

The validation process was allowed on logical CPUs `0-23`. A successful
two-worker context retained exact one-CPU masks. A controlled request with one
allowed CPU followed by `UINT32_MAX` produced one applied assignment, one
deterministic rejection, complete pool teardown, and a partial-application
report without submitting work.

## Validation evidence

All commands used Clang 21.1.8, `-std=c++20`, `-pthread`, and
`-Wall -Wextra -Wpedantic -Werror`.

| Configuration | Platform test | Execution-context test |
|---|---:|---:|
| Release (`-O3 -DNDEBUG`) | PASS | PASS |
| Debug (`-O0 -g`) | PASS | PASS |
| ASan + UBSan (`-O1 -g`) | PASS | PASS |
| TSan (`-O1 -g`) | PASS | PASS |
| simulated unsupported platform (`-U__linux__`) | PASS | PASS |

Representative direct build commands:

```sh
/usr/bin/clang++-21 -std=c++20 -O3 -DNDEBUG \
  -Wall -Wextra -Wpedantic -Werror -pthread \
  -Icompiler/lib/platform \
  compiler/lib/platform/thread_affinity_v1.cpp \
  compiler/tests/platform/thread_affinity_v1_test.cpp \
  -o /tmp/matcore-thread-affinity-release

/usr/bin/clang++-21 -std=c++20 -O3 -DNDEBUG \
  -Wall -Wextra -Wpedantic -Werror -pthread \
  -Icompiler/lib/platform -Icompiler/lib/runtime \
  compiler/lib/platform/thread_affinity_v1.cpp \
  compiler/lib/runtime/cpu_execution_context.cpp \
  compiler/tests/runtime/parallel/execution_context_test.cpp \
  -o /tmp/matcore-execution-context-affinity-release
```

TSan used `-fsanitize=thread -fno-omit-frame-pointer` and
`TSAN_OPTIONS=halt_on_error=1`. ASan/UBSan used
`-fsanitize=address,undefined -fno-omit-frame-pointer`,
`ASAN_OPTIONS=detect_leaks=1`, and
`UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`.

## Required integration CMake changes

The lead must make these shared-file changes after cherry-picking:

1. In `compiler/lib/platform/CMakeLists.txt`, add
   `thread_affinity_v1.cpp` to `matcore_platform_v1`.
2. In that same testing block, add executable
   `matcore_thread_affinity_v1_test` from
   `compiler/tests/platform/thread_affinity_v1_test.cpp`, link it to
   `MatcoreDSL::PlatformV1`, apply the standard strict warnings, and register
   CTest name `platform.thread_affinity.v1`.
3. In `compiler/lib/runtime/CMakeLists.txt`, add the direct private dependency
   `MatcoreDSL::PlatformV1` to `matcore_cpu_backends_v1` alongside
   `Threads::Threads`; do not rely only on the planner's transitive dependency.
4. Link `matcore_execution_context_test` directly to
   `MatcoreDSL::PlatformV1` as well as `MatcoreDSL::CpuBackendsV1`, because the
   focused test directly consumes the platform inventory API.

No installed header or public C ABI change is required by this internal lane.

## Limits

- The backend supports Linux scheduler affinity only. Windows remains
  explicitly unavailable until its documented platform backend is implemented
  and executed there.
- CPU IDs must fit the fixed Linux `cpu_set_t` represented by `CPU_SETSIZE`.
  Larger systems fail closed rather than truncating a mask.
- The context does not perform or claim NUMA memory placement. The caller still
  owns workspace allocation and first-touch policy.
- Affinity is immutable for the lifetime of a context; changing a placement
  plan requires creating a new context.
