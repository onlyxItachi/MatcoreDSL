# Strict GPU pair issuer: interrupted engineering snapshot

Status: **WIP, not qualified, not merge-ready**. The agent service exhausted its
usage allowance on 2026-10-04 before composed testing and independent review.
No active compiler/CTest process remained when the root preserved this snapshot.
Base: CPU source head `6e25df7e85ae3419d9e688541065433ff3151151` plus shared
integration hooks `b0fe264` (equivalent to root runtime hook commit `3e168a1`).

This branch owns the closed built-in GPU pair issuer, CLI, exact outlined graph,
LLVM graph pins, build-time image gate/embedding, and issuer/object/physical
fixtures. It does NOT contain the source/runtime implementation. That separate
branch is `mdslc/gpu-strict-fused-pair-runtime-v1`.

## Evidence actually available

- The predecessor isolated research at `40cac19` passed physical strict-pair
  correctness on sm_89 and gfx1150; hygiene-only follow-up `9a9c484` preserves
  those identities. It is not evidence that this production candidate passed.
- Tiny issuer builds emitted both target artifacts under external directory
  `/home/hamza-usta/mdslc-work/region-optimization-v1/builds/gpu-pair-issuer-tiny`.
- Complete outlined graph equivalence initially failed on ten default
  `memref.load/store` nontemporal attributes: builder false versus parser absent.
  The fix normalizes only absent false defaults on comparison clones after
  ordinary verification. True remains unequal; a negative was added. Independent
  exact-commit review remains pending.
- `rocdl-object-tests/report.json` records PASS for 15 image-inspection corruption
  controls on image SHA256
  `d279b1da34ee09f05263893ee26849191e8a73466b59a1b71b8500eab0b9ffcd`.
  This is object-gate evidence, not physical execution of that image.
- The NVVM gate found a same-kernel `CALL.REL.NOINC` targeting an EXIT address.
  The current unqualified change permits only that pattern and rejects other
  calls. Its instruction/target reasoning, negative controls, and actual gate
  must be independently reviewed and rerun; do not claim machine qualification.

## Required continuation

1. Review the default-attribute normalization and NVIDIA transfer-to-EXIT gate.
   Confirm exact AMD descriptor-symbol binding and all 35 explicit ABI fields.
2. Run issuer/CLI/adversarial object tests from a clean composed build. Confirm
   full LLVM pins on the exact supported 21.1.8 tuple, never relax them merely
   because a new artifact differs.
3. The real HIP `physical.asan` registration was already removed before
   `60622b0`: HIP is normal-only; NVVM separately registers normal/ASan. Real
   HIP/global host-ASan remains unqualified; HIP API mocks under ASan do not
   prove physical HIP sanitizer compatibility. Qualify the declared narrow
   physical modes, not an unsupported HIP sanitizer configuration.
4. Execute the production images and skinny dimension-boundary fixtures on the
   actual GPUs, then normal CUDA Compute Sanitizer and its isolated negative.
5. Compose with the separately reviewed runtime/source branch. Include its
   `RuntimeTests.cmake` BEFORE the issuer-only early return in this branch's
   test CMake file, so the no-SDK frontier test exists in Debug/ASan builds.
6. Qualify actual adapters, same-source none/fused single/program execution,
   unavailable targets, symbol ownership, relocated source/build-inaccessible
   packages, complete regressions, hosted CI, and independent final review.

No source/default execution, public API, performance, residency, arbitrary GPU,
or merge authority follows from this preservation commit.

## Resumed issuer checks (2026-10-04)

These results supersede only the pending component checks above, not the
research/production distinction or the unqualified integration checkpoint.

- The NVIDIA rejection was a CMake semicolon/list-splitting defect: the captured
  instruction's semicolon disappeared before its per-call regex. The corrected
  parser excludes semicolons, retains the exact `CALL.REL.NOINC` spelling, and
  requires its destination to name an unpredicated EXIT in exactly the sole
  expected disassembled kernel. Both global and local extra executable symbols
  are rejected. This bounded internal transfer is not a blanket call-free claim.
- Fresh tiny issuer test: **147 checks, 48 valid-IR corruption rejections PASS**.
  Its first run failed because the scratch-GEP mutation anchor omitted LLVM's
  actual `inbounds nuw` spelling; fixing that fixture did not change the issuer
  graph or its pins. Private CLI unsupported-input tests also passed.
- Actual NVIDIA and AMD production-image gates each passed **16** captured
  inspector corruption controls, including local extra functions; NVIDIA also
  covers a genuine redirected non-EXIT destination and a second disassembly
  section. These are synthetic gate attacks, not mutated image execution.
- The physical harness now computes all independent strict oracle expectations
  before any device API, protecting the oracle from vendor-library FP changes.
  Production-image physical execution remains pending at this checkpoint.
