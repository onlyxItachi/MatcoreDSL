# Accelerator expansion: exact interrupted handoff

Recorded 2026-10-04. Correctness-first campaign; no performance claim.
All three specialist/reviewer agent services stopped at their usage allowance.
The root preserved their work, made no speculative merge and did not substitute
self-review for the missing independent final gates. This is an interrupted
campaign, not a completed target qualification.

## Canonical and branch checkpoints

Canonical `main` remains clean at
`af4867270bf79684f3e1c6585d59440711778ba4` (documentation PR82).
Latest engineering merge remains PR79,
`f55b86e5d0fbeaa8d10d5857bc2bf2ce71168df4` (isolated CPU pair issuer).
No new engineering merge occurred, so `CURRENT_STATE.md` remains unchanged.

All worktrees below are under
`/home/hamza-usta/mdslc-work/region-optimization-v1`.

| Lane / worktree | Exact meaningful checkpoint | Disposition |
|---|---|---|
| CPU source proof / `source-candidate` | `6e25df7e85ae3419d9e688541065433ff3151151` | Frozen local qualification, PR80 prerequisite |
| CPU CI repair / `source-ci-budget` | `9a4dfeff70db57f3522e66fe9e13e8602c5a167c` | Pushed to PR80; workflow only; new hosted checks pending |
| Isolated GPU research / `gpu-fused-pair` | `9a9c484` (physical proof `40cac19`) | Pushed research branch; independently accepted bounded feasibility |
| Production GPU issuer / `gpu-pair-issuer` | `60622b06739c98dd4640596eff54982bfb98817a` | Local WIP commit, NOT qualified |
| GPU source/runtime / `gpu-pair-runtime` | `cf5391dc7f26ccf0f1c414775402799e23f00579` | Local code checkpoint; focused mocks reviewed, not composed |
| Independent law/tests / `accelerator-review` | `c191bff9f82f19a4514a3f2a8017a59ec991d134` | 449-check real Session test already integrated unchanged |
| SPIR-V/Vulkan research / `spirv` | `97a9b1ef68b6ebd18b3e2c899d912318c424cadb` (code `706e44e`) | Static physical feasibility; final independent review missing |

The report-only commit containing this handoff follows the runtime code checkpoint
in its branch; it does not change the code validation identity.

## Evidence, kept distinct

- CPU PR80 compiler subtree `d53a28d99ec12d5204c3263accc287ca492bc40c` passed
  406 distinct local tests with zero failures/skips: 404 ordinary + 2 genuinely
  installed/source-build-inaccessible package tests. Focused 14 are a subset.
- At `6e25df7`, hosted Release/Windows/ARM64/ASan/TSan/hygiene/legacy passed;
  push Debug passed, duplicate PR Debug hit 45 minutes twice. First run stopped
  during 253/257; retry job111483156050 during 250/257 after 249 passed. No failed
  testcase, but incomplete is not passed. `9a4dfef` changes only that budget to
  60 minutes. Do not merge until exact-new-head CI and review support it.
- GPU research: each actual sm_89/gfx1150 target passed 281 launches and 264 host
  bypass cases, 17305 numerical comparisons, 2554 canaries and 79676 immutable
  input checks. CUDA memcheck passed and detected the isolated malformed-scratch
  negative. AMD has oracle/canary evidence, not device sanitizer qualification.
- GPU runtime: independent real Session/mocked device test passed 449 checks.
  Extended API mocks passed 36 CUDA + 39 HIP processes with ASan+UBSan; original
  route mocks separately passed 34 + 34. These are not physical/source tests and
  do not inject C++ host allocation/thread failures. See the [runtime evidence](gpu-fused-pair-runtime-v1.md).
- SPIR-V/Vulkan: actual AMD RADV static 2x3x4 GEMM passed 80 numerical
  comparisons; NVIDIA Vulkan and llvmpipe refused missing f32 denormal controls
  before dispatch. This is not arbitrary Vulkan, LLVM SPIR-V backend, source
  candidate or performance qualification. The separate branch holds its report.

## Accepted reasoning and outstanding falsifiers

Resource contract8 permits a separately declared combined GPU realization:
retire original f1 guards only, keep f1/f2 source/capability/full-extent/FP and
target-bound order, and assign actual shared invocation/completion failures to
f2. No counterfactual first launch is claimed or real first failure relabelled.
Earlier publications/observations and old output handles survive; uncertain
completion quarantines all potentially live resources and poisons both routes.
See the [independent law and losing objection](accelerator-fusion-authority-review-v1.md).

Production issuer unresolved gates are in its WIP report. In particular:
default-false memref attribute normalization needs final exact review; the NVVM
same-kernel CALL.REL.NOINC-to-EXIT allowance is not qualified; the newly added
real HIP physical-ASan registration must be removed/deferred absent independent
justification. Do not confuse HIP API-mock ASan with real HIP/global host-ASan.
Skinny maximum-dimension production-image cases and composed actual adapters
remain pending. Neither a manifest nor the research images grant source authority.

## Resume order

1. Reauthenticate main/PR80/checks and all local worktrees; preserve every frozen
   artifact. Complete independent final review, merge PR80 normally only if its
   exact head is green, then update the concise canonical operator checkpoint.
2. Resume independent issuer and SPIR-V reviews. Retain SPIR-V as research until
   its controls, arithmetic, descriptor/barrier and completion law are reviewed.
3. Compose issuer `60622b0` with runtime `cf5391d` in a NEW worktree. Shared hooks
   already exist in both bases; do not duplicate them. Include
   `generated_gpu_fused_pair/RuntimeTests.cmake` before issuer-only early returns.
4. Fresh exact21.1.8 Release build, relevant issuer/object/mocked-runtime tests,
   then physical adapters/source (none/fused, single/program), unavailability and
   interposition, genuine source/build-inaccessible installed GPU consumers,
   full regression/package scopes, supported sanitizers and exact hosted CI.
   New focused ASan scope expects200 tests (one added no-SDK frontier).
5. Independent final review; separate reviewable issuer/source checkpoints and
   normal merges only when supported. Update CURRENT_STATE after each engineering
   merge. No automatic target selection, residency, performance or parity claim.

Use at most two compiler jobs TOTAL, one heavy link under the shared
`builds/heavy-link.lock`, and external SSD TMPDIR
`/var/tmp/mdslc-region-correctness.bvy8fe79`. No toolchain source build or system
package change. No active local test/build process remained at interruption.
Issues15/20 stay open; accelerator campaign issue83 tracks these separate lanes.
