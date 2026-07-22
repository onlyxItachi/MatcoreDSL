# Milestone 5 capability v2 and topology v1 lane

Date: 2026-07-22

Base: `e4dc0affff6c540a65435ba25c5cefa4d69cb562`

Branch: `mdslc/m5-capability-topology`

## Ownership

This lane owned `compiler/lib/platform/**`, `compiler/tests/platform/**`, and
this report. It did not edit the shared runtime, planner, public C ABI, or
top-level standalone CMake files.

## Implemented contracts

CPU capability record v2 separates five independently known feature domains:

- physical hardware support;
- OS-enabled architectural state;
- compiler support;
- compiled implementation availability;
- runtime-validation status.

Each domain uses separate `known` and `available` masks. Consequently an
incomplete probe is not converted into a false unsupported claim. The record
has a fixed canonical order for portable scalar F32, AVX2, FMA, AVX-512F/DQ/
BW/VL/VNNI/BF16, and AMX-TILE/BF16/INT8. An old v2 reader rejects unknown
record versions and feature bits rather than executing optimistically.

The x86 detector reads CPUID and XCR0 in process. AVX2/FMA requires XMM/YMM
state, AVX-512 requires opmask and complete ZMM state, and AMX requires tile
XSTATE plus the Linux `ARCH_GET_XCOMP_PERM` result. It never requests AMX
permission as a side effect. AArch64 receives a portable, injection-friendly
record without any x86 feature claim. Compiler support and implementation
availability remain distinct; the latter is deliberately injected by the
runtime/build integration layer.

CPU topology record v1 discovers Linux state directly from sysfs without a
subprocess. It represents sorted logical-CPU to core/socket/NUMA mappings,
SMT thread indices, NUMA membership, and deduplicated cache-sharing groups.
The same parser accepts an isolated sysfs root for deterministic tests, while
the validator and placement engine accept fully synthetic records.

Placement v1 provides compact, scatter, and local-first policies with
physical-core-only, prefer-physical-core, and allow-SMT choices. Cross-NUMA
placement requires an explicit request. A selected plan is metadata only:
it identifies CPUs/nodes and reports that affinity application and multiworker
first-touch remain caller-owned. It never changes affinity, allocates on a
NUMA node, migrates pages, or silently reduces the requested worker count.

## Validation-host evidence

The direct host probe reported:

- architecture: x86-64;
- 24 online logical CPUs;
- 12 physical cores;
- one socket;
- one NUMA node;
- 38 deduplicated sysfs cache-sharing records;
- XCR0: `0x2e7`;
- hardware and OS-enabled AVX2/FMA;
- hardware and OS-enabled AVX-512F/DQ/BW/VL/VNNI/BF16;
- no AMX hardware and no granted AMX permission.

For the lane test, compiled implementation availability was injected only for
portable scalar F32 and AVX2/FMA. Therefore AVX-512 remained explicitly
unimplemented and not runtime-validated even though hardware, OS, and compiler
domains were affirmative. This lane makes no AVX-512 execution claim and no
multi-node NUMA performance claim.

## Commits

1. `37e7fa4` — `feat(cpu): add fail-closed capability model v2`
2. `ef6785e` — `feat(cpu): add deterministic topology and placement model`

## Commands and results

Focused Debug/OpenBLAS-disabled build:

```sh
cmake -S compiler -B /tmp/matcore-m5-capability-build -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/matcore-m5-capability-build \
  --target matcore_platform_v1_test \
           matcore_cpu_capability_v2_test \
           matcore_cpu_topology_v1_test -- -j2
ctest --test-dir /tmp/matcore-m5-capability-build \
  --output-on-failure -R '^platform\\.' -j1
```

Result: platform tests 3/3 passed.

Fresh full Release build used native Clang 21 LibTooling, the bootstrap parity
path, and required OpenBLAS 0.3.32. The complete standalone suite passed
29/29 in 71.67 seconds. This includes native frontend, driver, integration,
installed consumer, typed IR, existing runtime/planner/benchmark gates, exact
AVX2 artifact checks, and all three platform tests.

Focused ASan/UBSan build used Clang 21 with
`-fsanitize=address,undefined -fno-omit-frame-pointer` and leak detection.
Platform tests passed 3/3. Direct strict compilation of both new production
translation units with `-Wall -Wextra -Wpedantic -Werror` passed and produced
ordinary ELF64 x86-64 relocatable objects.

`git diff --check` passed and the lane worktree was clean after this report's
commit. TSan is not a lane-local gate because these detection and planning
functions contain no mutable global state, worker threads, or shared executor;
the parallel-runtime lane owns the race-detection gate.

## Limitations

- Topology discovery is implemented for Linux sysfs. Synthetic records make
  the planner portable; a documented Windows API discovery backend is still
  required during the deferred Windows validation phase.
- The validation host has one NUMA node. Two-node behavior passed synthetic
  record and injected-sysfs tests only.
- Placement is a deterministic recommendation. The persistent runtime must
  apply affinity and first-touch deliberately and report the outcome.
- Capability v2 reports AMX permission but never requests it. Any future AMX
  execution context must perform and validate the documented permission flow.
- Runtime-validation bits are injected evidence. Discovery never turns a
  hardware flag or compiled function into a runtime-validation claim.

## Lane verdict

Capability model v2 and topology/placement v1 are ready for integration into
the Milestone 5 planner and persistent runtime. No hardware-gated execution or
multi-node performance claim is made by this lane.
