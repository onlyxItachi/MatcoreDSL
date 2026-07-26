# Milestone 7 final adversarial review

Date: 2026-07-26

Branch: `mdslc/native-blas-parity-v1`

Validated code checkpoint:
`ff483afcfc06c491deed88b8e194737940701086`

## Disposition

The local code, correctness, artifact, sanitizer, ABI, package, and evidence
contract review has no unresolved high- or medium-severity finding. The
candidate is acceptable as bounded CPU-backend and measurement-contract
hardening, subject to hosted pull-request gates.

Milestone 7 itself remains **partial**. No complete authenticated
same-checkpoint forward/reverse parity collection exists, so native/OpenBLAS
family ratios, the declared scaling envelope, and full-envelope planner regret
are not established. Issue #15 and GitHub milestone #5 remain open. No parity
completion tag or API/backend-contract freeze is authorized.

## Adversarial review targets and resolutions

1. **Under-measured cooperative packing activation — resolved.** The retained
   performance evidence covered one accepted short-wide shape and could not
   justify the former production region. `should_parallel_pack_b()` now fails
   closed, the focused runtime test requires `packed_b_threads == 0`, and the
   implementation remains private and dormant.
2. **Partial evidence producing an acceptance verdict — resolved.** Summary
   schema v3 distinguishes a bounded paired-measurement assessment from manual
   milestone disposition. Incomplete or malformed evidence emits no verdict;
   complete adverse evidence emits a bounded failure.
3. **Stale accepted output after a rejected run — resolved.** The summarizer
   invalidates existing Markdown/JSON outputs before validation and protects
   manifests, raw evidence, runner, and summarizer files from output aliasing.
4. **Selective planner-regret evidence — resolved.** The runner and summarizer
   authenticate the complete ordered eight-variant planner-v3 registry.
   Omitted, duplicated, illegal-timed, comparable-legal-untimed, and
   legal-non-comparable-timed candidates are rejected.
5. **Evidence-tool substitution or source drift — resolved.** The summary
   authenticates its own bytes and Git blob against the declared source
   commit. Runner, benchmark, plan, raw file, timing scope, result, placement,
   and forward/reverse coverage identities are also checked.
6. **Misleading performance attribution — resolved.** Historical ABBA and
   S-P-P-S measurements identify their exact pre-final-review checkpoints.
   They are not attributed to Milestone 5 or to the final code checkpoint, and
   no confidence interval, geometric mean, or sample-level win/loss claim is
   inferred.
7. **Scalar or spill-heavy bodies under ISA IDs — resolved.** Exact Release
   artifact tests require YMM/ZMM packed FMA operations, register-tile
   occupancy, and no optimized full-tile vector stack spill.
8. **Hidden allocation, transformed-operand cache, ABI break, or fallback —
   not found.** Workspace remains caller-owned, packed-B state is not cached
   across calls, the 15-function public C ABI is unchanged, and unavailable
   OpenBLAS requests fail without fallback.
9. **Windows scheduler-quantum-sensitive benchmark fixture — resolved.** The
   original 200 microsecond synthetic steady-state sleeps could be rounded
   above the 2 millisecond calibration target on Windows. Test-only follow-up
   `ff483af` uses a comfortably larger synthetic gap; the focused Release and
   Debug tests pass, including 20/20 repeated Release executions. Production
   calibration logic is unchanged.

## Exact local validation

At `ff483af`:

- Release with OpenBLAS 0.3.32: 50/50 CTest passed.
- Debug with OpenBLAS 0.3.32: 50/50 CTest passed independently.
- Release with OpenBLAS disabled: 50/50 CTest passed independently.
- ASan+UBSan supported runtime scope: 19/19 passed.
- TSan shared-state scope: 4/4 passed.
- Installed package/consumer/C17/source-inaccessible scope: 4/4 passed.
- Exact AVX2/compiler-vectorized/AVX-512 artifacts: 3/3 passed.
- Native `.mdsl -> ELF relocatable -> executable`: printed
  `MDSLC CPU GEMM PASS`.
- Repository hygiene and `git diff --check`: passed.
- Legacy frontend contract: 14/14 passed.

The blanket sanitizer CTest set is not a supported gate because two package
probes launch uninstrumented child linkers against the instrumented shared
runtime. Those expected unresolved-runtime failures were excluded; the
declared sanitizer scopes above passed without a sanitizer diagnostic.

## Evidence identity

The reviewed summary-v3 implementation at the validated code checkpoint is:

- SHA-256:
  `e92a5c7198cb7c56988a2b2ab54c5a79da1609fcf105256bfddf91ebfcde96fb`;
- Git blob: `191e68a2c6a8ea1954454e811ea147a690c48e08`.

The incomplete external receipt belongs to pre-final-review checkpoint
`2863253d1f2b06a943c2028ae298d0381d15ddf4`; it is not accepted by summary v3
and is not quantitative milestone evidence.

## Pending external gates

- Hosted Linux/OpenBLAS-required and OpenBLAS-disabled jobs.
- Hosted Windows clang-cl/COFF/PE/DLL/package regression.
- Hosted generated-artifact hygiene.
- A future exclusive-host complete forward/reverse parity collection.

The first three are pull-request merge gates. The last is a Milestone 7
completion gate and remains open after any bounded hardening merge.
