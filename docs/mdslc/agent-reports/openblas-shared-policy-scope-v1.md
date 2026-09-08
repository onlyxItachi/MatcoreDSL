# OpenBLAS shared policy scope correction

Normally merged in [PR #43](https://github.com/onlyxItachi/MatcoreDSL/pull/43)
at **`76b024abadcb32eb9f01effe92ec19ea13d6d763`**, 2026-09-06T19:37:42Z.
Reviewed head: `3a85a14ba478129ffd4c8fca8691b978e4335c8a`;
canonical pre-merge parent: `5ab85faac305862681817a5cec7e9ecfc912bf22`.
Merge tree `76b49d9241bcfe07a404eaf8e4d8ef20a36b6597` matches the independently
composed parent/head tree. Two independent reviews accepted the exact adapter
correction and all **19/19 exact-head hosted checks succeeded**. The hosted
required-provider lane executed the new concurrent-policy oracle against the
separately built OpenBLAS 0.3.32 GENERIC provider; this is distinct from local
Cooperlake validation. See the [normal-merge gate](https://github.com/onlyxItachi/MatcoreDSL/pull/43#issuecomment-5561663407).

## Falsified hypothesis

**OBSERVED:** retaining an independently authored concurrent first-use test
exposed an intermittent Release failure that its initial ASan run had missed.
Eight separate thread-confined Sessions concurrently requested the real linked
OpenBLAS provider. A worker correctly refused to issue a value, reporting
`candidate_failure`; the independent C ABI wrapper traced this to legacy status
19, `OpenBLAS SGEMM failed or did not honor the thread policy`. Caller FP-state
checks did not fail. Three subsequent successful fresh runs were not accepted as
disproof of the failing run.

The earlier belief that `openblas_set_num_threads_local` isolates each caller's
policy was wrong for the inspected implementation. At OpenBLAS `v0.3.32`,
[openblas_set_num_threads.c lines 40–44](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.32/driver/others/openblas_set_num_threads.c#L40)
invokes the global setter, and
[blas_server.c line 107](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.32/driver/others/blas_server.c#L107)
declares `blas_omp_threads_local` as a plain global integer, not TLS.

**ARCHITECTURAL IMPLICATION:** this provider policy is a shared mutable runtime
resource. Serialization belongs in the existing provider adapter used by both
the compatibility C ABI and the new immutable-value route, not only in a new
Session wrapper. No source semantic, numerical permission, or scheduler threshold
needs to change to repair it.

## Bounded correction

One private mutex in `cpu_openblas.cpp` encloses each complete Matcore-managed
provider policy lifetime: save prior count, set requested count, inspect, compute,
restore, and validate. The conformance probe uses the same mutex. Execution
finishes the once-only conformance query **before** acquiring this mutex; there
is no mutex-held wait on the conformance once-flag. Early error paths retain the
scope until restoration/postchecks finish. Other native/generated candidates do
not acquire this provider resource.

This covers one canonical Matcore runtime instance. Direct external provider
calls/global policy mutation must not overlap without equivalent shared exclusion;
independent static embeddings or multiple runtime copies have independent locks.
No provider/interposer callback-reentry, async-signal, post-fork, crash recovery or
universal process-isolation contract is invented. The old caller-control-only FP
contract remains unchanged; the closed adapter continues to supply its separate
full-FP-state preservation contract.

## Preserved negative control

`openblas_policy_scope_test.cpp` uses ordinary linker `--wrap` interception of
the **real** provider thread-count setter in a test-only executable. It tracks
overlapping save/set/compute/restore lifetimes while holding the first competing
entry for a bounded 200 ms. It does not fake arithmetic or add a production hook.
Eight barrier-released callers execute 128 real GEMMs with correct-output and
one-actual-thread checks.

The same test linked against the preserved pre-fix backend archive failed:
`maximum=3 active=0 failed executions=2` (exit 1). Artifacts remain outside the
repository at `/tmp/mdslc-openblas-policy-race.bXAEqS/`:

- `libmatcore_runtime.before.so` SHA256
  `5521a1cce94f492577685d08be65cb0ac6fd727553f18f88848fc8dd15cb9790`;
- `libmatcore_cpu_backends.before.a` SHA256
  `a7f630be5e594c557852a6b117888d5e94799bb8ec924f79ec30b4ad087bc903`;
- `independent.before` SHA256
  `496a5d3c6b4bbf2800f75a05f58d8cf19cda203b24aedb243dc87525f21db448`;
- `policy_scope.before`, the failing ordinary-link negative-control executable.

The source baseline is canonical `fd64850`'s unchanged provider adapter;
`7813502` adds the separately reviewed registry that exposed the latent race.
The correction therefore deserves a separate focused commit and review.

## Evidence limits

The tested provider is local OpenBLAS 0.3.32 pthread/DYNAMIC_ARCH Cooperlake.
Existing hosted CI builds the pinned 0.3.32 GENERIC implementation separately.
Locking establishes mutual exclusion structurally, and the tests falsify overlaps
and incorrect result/state reports. It does not prove all provider cores or
uncoordinated external clients. This is correctness evidence, not multithread BLAS
execution, scaling, parity, or an updated performance envelope.

## Post-correction validation

- Release ON: **17/17**, including 11 candidate-registry tests and 6 affected
  runtime/provider/workspace/ABI/FP tests. The real-provider policy-scope oracle
  reports one scope and one actual thread for all 128 calls.
- Debug ASan+UBSan ON: **20/20**, the above plus three generated-leaf execution
  and deliberate-OOB controls. The external provider itself is not instrumented.
- Release OFF: **10/10** candidate/absence/hostile-provider-boundary tests.
- Independent arithmetic reviewer repeated the original failing Release oracle
  in **100 fresh processes**: all passed, 47,638 checks per process, 4.73 s total.
  **30 fresh ASan+UBSan processes** also passed the same independent oracle.
  The policy-scope test separately passed. Guard reviewer independently repeated
  the pre-fix oracle (maximum 2 simultaneous scopes, 2 failures) and fixed tests.
- Both independent source reviews accepted the lock scope and once ordering at
  `cpu_openblas.cpp` SHA256
  `e1f5cfdbf9f57b67720dd9eaf5058f5d66d38934eca1b8165549ce73013bd5a5`, header
  `172ef492ffd6ce7fc377109144f4b181ab3106b0e6f169ecaebb133a42e5bad1`.

## Standalone canonical integration

The focused integration starts from independently refreshed local `main`,
`origin/main`, and GitHub `main`, all
`f988882710ac0b3677d908b3442841e3a8986b81` (PR #39). The canonical worktree was
clean. Original fix `437bdcc1d15373634f6cacd5439566ea6cd3d689` was cherry-picked
normally as `b42942e`; the registry and generated/source experiments are **not**
included. The owning `AGENTS.md` now describes shared policy scope rather than
assuming the provider's `_local` spelling means thread-local isolation. The
independently reviewed adapter hashes above are unchanged.

Clean standalone runtime builds used Clang 21, Ninja and
`cmake -S compiler/lib/runtime`, followed by the full registered CTest suite:

- `build-runtime-release`, Release with `MDSLC_ENABLE_OPENBLAS=ON` and
  `MDSLC_REQUIRE_OPENBLAS=ON`: **29/29 passed**, 5.69 s.
- `build-runtime-off`, Release with `MDSLC_ENABLE_OPENBLAS=OFF`:
  **28/28 passed**, 4.72 s. The real-provider policy test is correctly absent.
- `build-runtime-asan`, Debug with OpenBLAS required, `-O1 -g`,
  `-fsanitize=address,undefined -fno-omit-frame-pointer`, leak detection,
  initialization-order checking and halt-on-error: **26/26 passed**, 6.32 s.
  Established sanitizer configuration excludes three object-instruction tests;
  no registered test was skipped. The linked external OpenBLAS library is not
  itself sanitizer-instrumented.

These suites cover runtime numerical profiles, native kernels, planner,
workspace/context, C ABI, FP guards, provider-enabled/disabled behavior and
platform support. Hosted checks and the final canonical merge are recorded on
the focused PR; they are not implied by these local outcomes.
