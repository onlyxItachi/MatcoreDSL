# Independent FP environment and evaluation review

## Verdict

Accepted for the implemented Linux x86-64 explicit-GEMM-F32 scope at
`d131ba3634519df0bec001c2d616a5cf78a10b12`.

The implementation series reviewed was:

- `0e66cc1`, `2426cf3`, `51c399e`, `715936e`, `847eff8`, `6e58c29`, and
  `d964fb6` for the original FP compile/runtime guard;
- `5d3fe703d5003713982336b30999c1990cda32cd` for the independently requested
  source-evaluation correction; and
- `d131ba3634519df0bec001c2d616a5cf78a10b12` for the corrected evidence
  report.

No unresolved high- or medium-severity finding remains in this bounded review.

## Reproduced blocker and closure

The first independent pass reproduced a high-severity compile-profile bypass:
Clang 21 accepted global `-ffp-eval-method=double` and
`-ffp-eval-method=extended`, while the target-local profile did not restore
source-type evaluation. `FLT_EVAL_METHOD` was consequently 1 or 2 rather than
0, which contradicted the `explicit-gemm-f32-v1` requirement for F32
accumulation without excess intermediate precision.

Commit `5d3fe70` closes that path at three independent layers:

1. configuration rejects every explicitly supplied evaluation method other
   than `source` in all discovered C and C++ global/configuration flag
   variables;
2. Clang runtime/backend targets append `-ffp-eval-method=source` to their
   private precise profile; and
3. the forced internal compile-contract header rejects an undefined
   `FLT_EVAL_METHOD` or any value other than zero.

The safe `source` value remains accepted. The nested configure test reproduced
rejection of `double` and `extended`, acceptance of `source`, and the existing
`-ffast-math` rejection. Inspection of both final Ninja command graphs found
11 critical runtime/backend translation-unit commands per build, each with
exactly one `-ffp-eval-method=source` and the forced compile-contract header.

## Adversarial review coverage

The full reviewed series was challenged for:

- unsafe global compile-profile ordering or bypass;
- native and linked-OpenBLAS finite, gradual-subnormal, NaN, infinity,
  infinity-times-zero, opposite-infinity, and FP-control conformance;
- immutable provider identity, deferred `call_once` probing, single-thread
  provider control, and restoration of provider thread count and caller FP
  controls;
- the documented boundary that an opaque provider violation detected only
  after a call cannot promise unchanged output;
- malformed descriptor, forced variant, tensor span, workspace, transformed
  storage, provenance, and alias validation precedence;
- whole-byte preservation of caller-owned output, workspace, transformed
  descriptor/storage, options, policy, and report objects on pre-execution
  rejection paths covered by the public contracts;
- all-active-worker FP admission, collective task suppression, reuse after
  rejection, barrier/deadlock behavior, and shared packed-B publication order;
- the distinction between cached process capabilities and mandatory
  caller/worker thread-local FP-state admission;
- additive C ABI status 26 and preservation of the prior public layout; and
- the explicitly bounded Windows claim.

No additional high- or medium-severity defect was confirmed. In particular,
provider post-call mutation is described as a limitation rather than claimed
away, every active native worker is admitted before task or shared-B work, and
capability discovery is not used as a substitute for thread-local execution
state validation.

## Executed evidence

At current head, with no concurrent compiler build, the independent lane ran:

```sh
ctest --test-dir /home/hamza-usta/.cache/mdslc-fp-env-build \
  -R '^(platform\.fp_environment\.v1|runtime\.fp\.compile_profile|runtime\.fp\.reject_unsafe_global_flags|runtime\.cpu\.packed_avx2|runtime\.cpu\.execution_context\.v1|runtime\.cpu\.parallel_packed\.v1|runtime\.cpu\.workspace_v1|runtime\.cpu\.openblas_adapter|runtime\.c_abi\.compatibility_v1|runtime\.c_abi\.fp_environment_v1|runtime\.cpu\.variant_conformance\.v1|runtime\.c_abi\.public_context_v1)$' \
  --output-on-failure -j2

ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
ctest --test-dir /home/hamza-usta/.cache/mdslc-fp-env-sanitize \
  -R '^(platform\.fp_environment\.v1|runtime\.fp\.compile_profile|runtime\.fp\.reject_unsafe_global_flags|runtime\.cpu\.packed_avx2|runtime\.cpu\.execution_context\.v1|runtime\.cpu\.parallel_packed\.v1|runtime\.cpu\.workspace_v1|runtime\.cpu\.openblas_adapter|runtime\.c_abi\.compatibility_v1|runtime\.c_abi\.fp_environment_v1|runtime\.cpu\.variant_conformance\.v1|runtime\.c_abi\.public_context_v1)$' \
  --output-on-failure -j2
```

Results:

- normal Debug focused matrix: 12/12 passed, 0 failed, 5.22 seconds;
- ASan/UBSan Debug focused matrix: 12/12 passed, 0 failed, 14.14 seconds;
- normal command graph: 11/11 critical commands authenticated;
- sanitizer command graph: 11/11 critical commands authenticated; and
- `git diff 5d3fe70^..d131ba3 --check`: clean.

This lane did not run the full repository suite, performance/parity sweeps, or
a Windows build.

## Acceptance boundary

The acceptance applies to the physical Linux x86-64 runtime and compile profile
tested with Clang 21.1.8, including the linked single-thread OpenBLAS adapter
and the existing native serial/parallel variants. The Windows `/fp:precise`
path and Windows implementation of the FP environment inspector remain
source-reviewed only in this lane and still require the hosted Windows
integration gate. Multithread OpenBLAS remains deliberately unavailable until
provider worker FP state can be authenticated.
