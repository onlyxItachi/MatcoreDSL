# Bounded pre-freeze contract resolution report

Date: 2026-08-11

Status: implementation evidence for Milestone G review. This report does not
freeze the public API, ABI, backend contract, operation set, or deprecation
schedule.

## Scope

This bounded change resolves three ambiguities in already exported interfaces:

1. ownership, invalidation, and supported reuse of packed-B v1;
2. returned C-string ownership and machine-readable diagnostic boundaries; and
3. additive operation, capture-schema, and C ABI evolution before a later
   public freeze.

No C record layout, field offset, enum value, symbol signature, implementation
selection, planner cost, frontend behavior, or serialized Matcore IR v1 byte is
changed.

## Implementation audit

### Packed-B v1

`compiler/lib/runtime/cpu_packed_b_format.h` constructs provenance from the
source address, packed address, storage extent, packed element count, shape,
and packing constants. It does not read source or packed bytes into the token.
Metadata validation recomputes that token and checks the same address/layout
facts. `compiler/lib/runtime/cpu_runtime.cpp` additionally requires exact RHS
address identity, validates storage/descriptor/workspace non-overlap, and
routes only the forced native packed AVX2/FMA request. The v2 planner rejects
that candidate unless `requested_threads == 1`.

Therefore the implementation already matches the newly explicit bounded
contract:

- caller-owned source and packed storage;
- borrowed address/metadata snapshot;
- no content authentication;
- explicit repack after mutation, relocation, or deallocation;
- synchronous serial reuse only; and
- no cross-context concurrent-sharing surface.

The implementation was not changed to detect same-address source mutation,
because doing so would falsely claim a content or immutable-owner mechanism
that v1 does not have.

### Returned diagnostic strings

`compiler/lib/runtime/cpu_runtime.cpp` forms status values from string literals,
platform rejection functions returning static literals, planner `string_view`
values backed by static registry/literal storage, and OpenBLAS provider identity
pointers. `compiler/lib/runtime/cpu_openblas.cpp` snapshots the latter in a
function-local static provider record; OpenBLAS documents its configuration and
core-name results as provider-owned strings.

No path in the exported report/status population code returns a pointer into a
temporary `std::string`, stack buffer, or per-call plan object. The implementation
therefore satisfies the documented NUL-terminated runtime-static or
provider-lifetime rule without a runtime source change.

Exact human wording remains deliberately outside the machine ABI. Stable IDs,
status codes, enums, versions, and structured report fields retain their
documented machine meaning.

## Contract changes

- `compiler/include/matcore/runtime_c.h` now states the common returned-string
  lifetime and parsing rules and the complete packed-B v1 ownership contract.
- [Pre-freeze decisions](PRE_FREEZE_DECISIONS.md) distinguishes resolved v1
  behavior from the still-deferred general transformed-operand problem.
- [Interface evolution policy v1](INTERFACE_EVOLUTION_POLICY_V1.md) forbids
  in-place source-operation, capture-schema, and C ABI reinterpretation while
  leaving the final freeze and support windows for later approval.

## Focused tests

`workspace_runtime_test.cpp` covers:

- two successful sequential executions using one packed descriptor;
- fail-closed rejection of a two-thread packed-B v1 request without output or
  requirements mutation;
- rejection of copied packed bytes relocated to a different address while
  retaining stale provenance;
- explicit repacking after same-address source mutation and execution of the
  new contents; and
- continued readability of borrowed runtime/provider strings after later
  runtime calls.

`runtime_cpu_test.cpp` no longer treats exact status wording as compatibility
ABI and verifies that borrowed status, selection, stable-ID, and reason pointers
remain readable and NUL-terminated across later runtime calls.

`runtime_abi_compat_test.cpp` pins all existing candidate-count/version constants
in this surface, the pointer field types, and the existing LP64 layouts and
offsets. It adds no symbol or record.

## Validation

A fresh out-of-tree Release build used Clang/LLVM 21.1.8, the coherent native
Clang 21 frontend configuration, optional OpenBLAS 0.3.32 discovered through
pkg-config, and Matcore MLIR disabled because this bounded test surface does not
link the semantic dialect. The serialized build ran at low priority and `-j2`:

```sh
nice -n 10 cmake -S compiler -B "$MDSLC_G_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_MATCORE_MLIR=OFF \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang

nice -n 10 cmake --build "$MDSLC_G_BUILD" --target \
  matcore_runtime_cpu_test \
  matcore_workspace_runtime_test \
  matcore_runtime_abi_compat_test -- -j2

ctest --test-dir "$MDSLC_G_BUILD" --output-on-failure -j1 -R \
  '^(runtime\.cpu\.workspace_v1|runtime\.c_abi\.compatibility_v1|runtime\.cpu\.gemm_v0)$'
```

Result: all three requested targets built successfully; focused CTest passed
3/3 with zero failures. This report does not claim the full Release, Debug,
sanitizer, installed-consumer, or Windows gates; those remain root milestone
integration gates.

## Deferred decisions

The following remain explicitly unresolved and require additive design:

- content-authenticated or generation-authenticated transformed operands;
- immutable ownership and cross-context concurrent sharing;
- context-backed prepacked-B execution;
- caller-sized structured planner/diagnostic iteration;
- a device-neutral execution context and public execution intent;
- public dynamic-shape constraint encoding; and
- final compatibility and deprecation support durations.

These deferrals do not weaken the bounded v1 rules and are not permission for a
global mutable packed-weight cache.
