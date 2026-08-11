# Floating-point environment guard

Date: 2026-08-11

## Scope and result

This lane implements the fail-closed `explicit-gemm-f32-v1` floating-point
environment boundary required by ADR-0009. The public C ABI gains only the
additive status code
`MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0 = 26`.

The focused implementation is in commits:

- `0e66cc1` — versioned platform inspector and pure decoder;
- `2426cf3` — execution gates, persistent-worker barrier, public status, and
  mutation-preservation tests;
- `51c399e` — configure-time unsafe-flag rejection and target-local precise
  compilation profile;
- `715936e` — native and OpenBLAS numerical/control-state conformance;
- `847eff8` — complete storage/span validation before every FP guard;
- `6e58c29` — whole-object byte-preservation assertions for rejected public
  calls.

## Authenticated runtime contract

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
control state preserves denormal operands and results. Unknown platforms and
architectures fail closed.

The runtime does not normalize a rejected caller or worker environment. The
only restoration operation is after an opaque provider call: it restores the
caller's authenticated control snapshot, rather than choosing a new state.
Control comparison ignores status flags, and Linux x87 restoration clears
pending flags before restoring the control word, so this is exact **control
state** preservation rather than full status-snapshot preservation.

## Authenticated compile contract

The standalone configure rejects known unsafe global flags, including fast or
unsafe math, finite-only/NaN/Inf disabling, approximate/reciprocal math,
non-IEEE denormal modes, and `/fp:fast`. It then applies a target-local precise
profile to both `matcore_cpu_backends_v1` and `matcore_runtime`:

- Clang GNU mode: `-fno-fast-math`, finite/NaN/Inf preservation,
  `-ffp-contract=fast`, no approximate/reciprocal math, and
  `-fdenormal-fp-math=ieee`;
- GCC mode: the supported equivalent subset;
- clang-cl/MSVC ABI: `/fp:precise`.

A forced internal header rejects `__FAST_MATH__`, nonzero
`__FINITE_MATH_ONLY__`, `_M_FP_FAST`, or a missing target-local profile macro.
The compile-command test authenticated all 11 runtime/backend translation-unit
commands in the focused build. A nested configure test proved that an injected
`-ffast-math` fails with the MDSLC contract diagnostic.

## Backend conformance

Each available native variant is executed on deterministic finite, gradual
subnormal, and nonfinite fixtures before it is recorded as runtime validated.
The nonfinite fixture independently checks:

- direct NaN propagation;
- positive and negative infinity classes;
- `Inf * 0 -> NaN`;
- `(+Inf) + (-Inf) -> NaN` across the K reduction.

The same report checks exact control-state preservation. Reference, tiled,
compiler-vectorized, packed AVX2/FMA, and packed AVX-512/FMA are authenticated
with one execution worker when built and legal. Parallel AVX2 and AVX-512 use
two actual persistent workers, test the same numerical classes, and verify both
workers' control state. Instrumented builds do not advertise the deliberately
disabled compiler-vectorized implementation.

OpenBLAS 0.3.32 is keyed to the exact linked package/configuration/core and
threading identity. Its immutable process-local v1 report uses fixed aligned
stack scratch, forces one provider thread, checks the same finite/subnormal/
nonfinite classes, and verifies provider-thread and caller control-state
restoration. The registry advertises OpenBLAS only after that report passes.
Provider-private multithread worker state is not observable, so OpenBLAS remains
limited to one advertised thread under this proof contract.

Every ordinary OpenBLAS call snapshots and restores provider thread count and
caller control state. Pre-call failures preserve output. A state violation
detected after `cblas_sgemm` is necessarily post-execution and may therefore
leave C modified even though the call returns provider failure; no stronger
mutation claim is made.

## Execution and validation precedence

The guard runs after all established non-FP validation and immediately before
the first pack, workspace mutation, or output write:

- v0: descriptors/policy/shape/layout/memory/mutability, pointer/alignment/
  alias, then plan/variant, then FP;
- execute v1: options/report, GEMM, plan, complete workspace null/size/
  alignment/address/overlap, then FP;
- prepack: options/forced variant, empty output descriptor, GEMM, plan and
  requirements, complete storage and descriptor spans/overlaps, then FP, pack,
  and one final descriptor publication;
- prepacked execution: options/report and basic packed-descriptor ABI, GEMM
  validation needed to construct the problem, complete packed metadata/
  provenance/source identity/spans/overlaps, then plan, complete workspace
  spans/overlaps, and FP;
- context execution: context/options/report, GEMM, plan, complete workspace,
  all actual-worker preflight, then execution.

The public adversarial test poisons FTZ while supplying insufficient,
misaligned, overflowing, or overlapping direct workspace; invalid prepack
storage/descriptor; and malformed, stale, overflowing, or overlapping
prepacked views/workspace. Every earlier error wins over status 26, and rejected
output, workspace, packed storage, descriptor, and report bytes remain
unchanged.

Persistent submissions use an all-active-worker preflight barrier. No task
callback runs until every active worker has reported a compatible environment.
If one worker fails, every task is suppressed and the context remains reusable.
Serial shared-B preparation occurs on worker zero after this barrier, so no
submitting-thread or worker mutation precedes collective acceptance.

## Focused validation

Toolchain: Ubuntu Clang 21.1.8, `x86_64-pc-linux-gnu`, OpenBLAS 0.3.32.
Physical host: AMD Ryzen AI 9 HX 370, 12 cores/24 logical CPUs, one NUMA node,
usable AVX2/FMA and AVX-512F state.

The final normal Debug focused matrix passed 12/12:

```text
platform.fp_environment.v1             PASS
runtime.fp.compile_profile             PASS
runtime.fp.reject_unsafe_global_flags  PASS
runtime.cpu.packed_avx2                PASS
runtime.cpu.execution_context.v1       PASS
runtime.cpu.parallel_packed.v1         PASS
runtime.cpu.workspace_v1               PASS
runtime.cpu.openblas_adapter           PASS
runtime.c_abi.compatibility_v1         PASS
runtime.c_abi.fp_environment_v1        PASS
runtime.cpu.variant_conformance.v1     PASS
runtime.c_abi.public_context_v1        PASS
```

The same 12/12 matrix passed in a Debug build compiled with
`-fsanitize=address,undefined -fno-omit-frame-pointer`, using
`ASAN_OPTIONS=detect_leaks=1` and `UBSAN_OPTIONS=print_stacktrace=1`.

Physical tests exercise FTZ, DAZ, all three non-RNE modes in MXCSR and x87,
every MXCSR/x87 exception-mask family (with pending flags cleared before
unmasking), and legality-neutral raised MXCSR status flags. Runtime tests also
cover a poisoned persistent worker, collective suppression before shared-B
packing, deadlock absence, and reuse after a rejected submission.

Representative commands:

```sh
cmake --build /home/hamza-usta/.cache/mdslc-fp-env-build \
  --target matcore_fp_environment_v1_test \
           matcore_fp_environment_runtime_test \
           matcore_cpu_runtime_validation_test \
           matcore_openblas_adapter_test \
           matcore_execution_context_test \
           matcore_parallel_packed_gemm_test \
           matcore_workspace_runtime_test \
           matcore_runtime_abi_compat_test \
           matcore_public_context_c_api_test \
           matcore_packed_avx2_test -- -j2

ctest --test-dir /home/hamza-usta/.cache/mdslc-fp-env-build \
  -R '^(platform\.fp_environment\.v1|runtime\.fp\.compile_profile|runtime\.fp\.reject_unsafe_global_flags|runtime\.cpu\.packed_avx2|runtime\.cpu\.execution_context\.v1|runtime\.cpu\.parallel_packed\.v1|runtime\.cpu\.workspace_v1|runtime\.cpu\.openblas_adapter|runtime\.c_abi\.compatibility_v1|runtime\.c_abi\.fp_environment_v1|runtime\.cpu\.variant_conformance\.v1|runtime\.c_abi\.public_context_v1)$' \
  --output-on-failure -j1

ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
ctest --test-dir /home/hamza-usta/.cache/mdslc-fp-env-sanitize \
  -R '^(platform\.fp_environment\.v1|runtime\.fp\.compile_profile|runtime\.fp\.reject_unsafe_global_flags|runtime\.cpu\.packed_avx2|runtime\.cpu\.execution_context\.v1|runtime\.cpu\.parallel_packed\.v1|runtime\.cpu\.workspace_v1|runtime\.cpu\.openblas_adapter|runtime\.c_abi\.compatibility_v1|runtime\.c_abi\.fp_environment_v1|runtime\.cpu\.variant_conformance\.v1|runtime\.c_abi\.public_context_v1)$' \
  --output-on-failure -j1
```

No whole-repository CTest run was started by this lane; integration owns that
serialized gate.

## Validation boundary and limitations

- Linux x86-64 behavior was physically executed and sanitizer-tested.
- The Windows x64 backend and `/fp:precise` profile are source-complete but were
  not compiled or executed on Windows by this Linux lane; hosted Windows
  validation remains an integration gate.
- Unknown architectures are unsupported rather than silently assumed safe.
- Multithread OpenBLAS is unavailable until provider workers can be
  authenticated.
- The guard applies to the explicit F32 GEMM semantic profile. BF16-to-F32 and
  I8-to-I32 reference entry points retain separate typed numerical contracts.
- The environment and conformance reports remain internal; no mutable or C++
  platform type was added to the public C ABI.
