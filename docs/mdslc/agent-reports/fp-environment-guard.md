# Floating-point environment guard

Date: 2026-08-11

## Scope and result

This lane implements the fail-closed `explicit-gemm-f32-v1` floating-point
environment boundary required by ADR-0009. The implementation is internal and
read-only: it never normalizes caller or worker state. The public C ABI gains
only the additive status code
`MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0 = 26`.

The focused implementation is in commits:

- `0e66cc1` — versioned platform inspector and pure decoder;
- `2426cf3` — execution gates, persistent-worker barrier, public status, and
  mutation-preservation tests.

## Authenticated contract

Linux x86-64 reads MXCSR with `_mm_getcsr` and the x87 control word with
`fnstcw`. The pure decoder requires:

- round-to-nearest-even in MXCSR and the x87 control word;
- every MXCSR and x87 exception mask set;
- MXCSR FTZ and DAZ both clear.

MXCSR exception status flags are deliberately ignored. The x87 precision mode
is reported for diagnostics but does not affect legality because the currently
consumed F32 backends do not require one x87 precision setting.

The Windows x64 backend uses `_controlfp_s` plus `_mm_getcsr`. In addition to
the common contract, it requires `(control & _MCW_DN) == _DN_SAVE` so the CRT
control state preserves denormal operands and results. The inspector source is
compiled with `/fp:strict` under the MSVC-ABI frontend. GNU-mode Clang/GCC use
`-frounding-math -ftrapping-math -ffp-contract=off` for the inspector
translation unit. Unknown platforms and architectures fail closed.

## Execution boundary

The guard runs after existing descriptor, policy, plan, alias, alignment, and
workspace checks so established validation errors retain priority. It runs
immediately before the first operation that can pack, mutate caller workspace,
or write output for:

- the legacy v0 F32 entry point;
- workspace-aware v1 direct execution;
- packed-v1 execution;
- B prepacking and prepacked-B execution;
- bound and unbound execution-context single-worker routes;
- persistent parallel AVX2 and AVX-512 routes.

Persistent submissions use an all-active-worker preflight barrier. No task
callback runs until every active worker has reported a compatible environment.
If one worker fails, every task is suppressed and the context remains reusable.
Serial shared-B preparation was moved from the submitting thread to worker zero
after this barrier; therefore neither serial nor dormant cooperative packing
can mutate shared workspace before all actual execution threads pass.

The runtime cannot inspect provider-private OpenBLAS workers. Consequently the
validated OpenBLAS candidate ceiling is one thread and direct multithread
requests fail closed. This is a deliberate limitation, not an assertion that
opaque provider workers inherit conforming state.

## Focused validation

Toolchain: Ubuntu Clang 21.1.8, target `x86_64-pc-linux-gnu`.

The normal Debug build used native LibTooling and OpenBLAS. The focused aggregate
passed 8/8:

```text
platform.fp_environment.v1          PASS
runtime.cpu.execution_context.v1    PASS
runtime.cpu.parallel_packed.v1      PASS
runtime.cpu.workspace_v1            PASS
runtime.cpu.openblas_adapter        PASS
runtime.c_abi.compatibility_v1      PASS
runtime.c_abi.fp_environment_v1     PASS
runtime.c_abi.public_context_v1     PASS
```

The focused ASan+UBSan Debug build used
`-fsanitize=address,undefined -fno-omit-frame-pointer`. Five guard-sensitive
tests passed 5/5 with leak detection and UBSan stack traces enabled:

```text
platform.fp_environment.v1          PASS
runtime.cpu.execution_context.v1    PASS
runtime.cpu.parallel_packed.v1      PASS
runtime.cpu.openblas_adapter        PASS
runtime.c_abi.fp_environment_v1     PASS
```

The tests include pure decoding, physical RAII violations for MXCSR/x87
rounding and mask families, physical FTZ/DAZ, legality-neutral status flags,
direct and prepacked public ABI rejection, byte-for-byte output/workspace/
packed-storage preservation, a poisoned persistent worker, suppression before
shared-B packing, deadlock absence, and reuse after a rejected submission.

Representative commands:

```sh
cmake --build /home/hamza-usta/.cache/mdslc-fp-env-build \
  --target matcore_fp_environment_v1_test \
           matcore_fp_environment_runtime_test \
           matcore_execution_context_test \
           matcore_parallel_packed_gemm_test \
           matcore_openblas_adapter_test \
           matcore_workspace_runtime_test \
           matcore_runtime_abi_compat_test \
           matcore_public_context_c_api_test -- -j2

ctest --test-dir /home/hamza-usta/.cache/mdslc-fp-env-build \
  -R '^(platform.fp_environment.v1|runtime.cpu.execution_context.v1|runtime.cpu.parallel_packed.v1|runtime.cpu.openblas_adapter|runtime.c_abi.fp_environment_v1|runtime.cpu.workspace_v1|runtime.c_abi.compatibility_v1|runtime.c_abi.public_context_v1)$' \
  --output-on-failure -j1

ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
ctest --test-dir /home/hamza-usta/.cache/mdslc-fp-env-sanitize \
  -R '^(platform.fp_environment.v1|runtime.cpu.execution_context.v1|runtime.cpu.parallel_packed.v1|runtime.cpu.openblas_adapter|runtime.c_abi.fp_environment_v1)$' \
  --output-on-failure -j1
```

No whole-repository CTest run was started by this lane; integration owns that
serialized gate.

## Validation boundary and limitations

- Linux x86-64 behavior was physically executed and sanitizer-tested.
- The Windows x64 backend and strict build flags are source-complete but were
  not compiled or executed on Windows by this Linux lane; hosted Windows
  validation remains an integration gate.
- Unknown architectures are unsupported rather than silently assumed safe.
- Multithread OpenBLAS is unavailable under this proof contract until provider
  worker state can be authenticated.
- This guard applies to the explicit F32 GEMM semantic profile. BF16-to-F32 and
  I8-to-I32 reference entry points have separate typed numerical contracts and
  are not claimed by this report.
- The environment report remains internal; no mutable or C++ platform type was
  added to the public C ABI.
