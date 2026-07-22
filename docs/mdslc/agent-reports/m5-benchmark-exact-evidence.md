# Milestone 5 exact-context benchmark evidence

Date: 2026-07-22

## Scope and ownership

This lane changed only `compiler/tools/matcore-bench/`,
`compiler/tests/benchmark/`, and this report.  The benchmark now authenticates
runtime support against the exact execution context used for the measurement,
passes that evidence to the planner, and reports whether worker affinity was
actually established.  It does not infer provider or parallel support from a
different process-global context.

## Exact-context behavior

- Each cached execution-context record owns its
  `CpuRuntimeValidationEvidenceV1`.
- Compact, scatter, local-first, and physical-core-only measurements validate
  their variants on the exact bound context.
- The lazy default context is likewise validated on the context it executes.
- Capability-level AVX2/AVX-512 evidence remains a direct tiny packed-kernel
  numerical self-test; planner variant evidence is context-specific.
- A forced variant is rejected when that exact context has not runtime
  validated it.  The benchmark never silently substitutes another variant.
- A one-CPU `taskset` context rejects parallel variants, while serial variants
  remain available when their exact numerical validation passes.
- OpenBLAS is allowed only after the exact single-thread provider validation;
  the benchmark never authenticates a multi-thread provider path that the
  runtime validation helper did not execute.

## Adversarial finding and resolution

Repeated tiny benchmark processes exposed a high-severity executor lifetime
race: inactive workers could retain the pointer to a stack-backed submission
after `run_tasks` returned.  The benchmark change was held from final handoff
until the runtime owner fixed participation selection under the executor mutex
and added a 4,096-iteration alternating-active-worker regression.

Integrated runtime commits used for final evidence:

- `55943dc` — inactive workers no longer retain borrowed submissions.
- `1c40346` — alternating active-worker-count lifetime stress regression.

The final validation below was run from clean integration HEAD `1c40346`.

## Validation evidence

Toolchain and provider:

- Clang/LLVM 21.1.8.
- Release and RelWithDebInfo ASan/UBSan builds, Ninja `-j2`.
- OpenBLAS 0.3.32, pthread, LP64, discovered through `pkg-config`.

OpenBLAS-enabled Release on exact integration HEAD:

```text
ctest --test-dir /tmp/matcore-m5-root559-release --output-on-failure -j1
100% tests passed, 0 tests failed out of 41
```

The complete suite includes frontend, native driver, integration, installed
consumer, installed C17 ABI, IR, platform/topology/affinity, all CPU runtimes,
planner, benchmark contract, and benchmark CLI JSON tests.

OpenBLAS-disabled Release:

```text
ctest --test-dir /tmp/matcore-m5-root1c-noblas --output-on-failure -j1 \
  -R 'benchmark.cpu.contract|benchmark.cpu.cli_json|planner.cpu.cli.registry_v3'
100% tests passed, 0 tests failed out of 3
```

Forcing `cpu.external.openblas.f32.v1` in that build returned exit status 1 and
the exact diagnostic `OpenBLAS CBLAS adapter is not linked`.

Sanitizer validation:

```text
ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1:abort_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir /tmp/matcore-m5-root559-asan --output-on-failure -j1 \
  -R 'runtime.cpu.execution_context.v1|benchmark.cpu.contract|benchmark.cpu.cli_json'
100% tests passed, 0 tests failed out of 3
```

Repeated-process lifetime stress at exact HEAD:

- Release: 100/100 invocations of a 2x2x2 reference GEMM with two requested
  threads passed; stderr stayed empty.
- ASan+UBSan with leak and stack-use-after-return detection: 100/100 identical
  invocations passed; stderr stayed empty.
- An earlier independent replay immediately after the core fix also passed
  200/200 Release and 100/100 ASan+UBSan invocations.

## Result

The benchmark lane is ready for integration.  Planning evidence, affinity
metadata, and measured execution now refer to the same concrete execution
context.  The discovered executor race has a central fix and a dedicated
runtime regression; it is not hidden or worked around in benchmark code.
