# Independent GPU strict-pair review v1

Review snapshot: 2026-10-04. Reviewer: optimization adversarial lane.

**Accept the bounded issuer component and the reviewed runtime/source design;
do not yet treat this record as complete PR #85/#86 merge qualification.**
The composed full regression, separate safety gate, genuine producer-inaccessible
packages, and final exact-head hosted results remain pending at this snapshot.
The integration owner may record their later outcomes in a focused follow-up;
they must not be inferred from the component tests below.

## Exact identities and independence

| Reviewed object | Identity and scope |
| --- | --- |
| Isolated issuer implementation | `21aad61e21fbc3e86687fd35e3290a77a7d0c401`: bounded component ACCEPT. |
| Issuer-only stacked PR #85 | `950e5cabcaf42b0f22b09160fa140a0015a85729`, report-only successor of `ada14a44fc4abb451c3127cd71bc2bacd98a0e3d`: static composition ACCEPT. Its compiler tree `dcdfdca1e7e6ed705ceaaa7b8d97caf4ba8734c1` is identical to `21aad61e`. |
| Composed runtime/source and source runner | `cb2bb730693aedd1d469e73e2fd943fb955b4f3d`: static ACCEPT plus the four actual source executions described below. Runtime, codegen, frontend and public-header bytes match the earlier `cf5391dc7f26ccf0f1c414775402799e23f00579` review. |
| Source PR #86 composition | `16bc65d64affce88f3081e29452de55d658c9bd9`: production-equivalent to `cb2bb73`; the only compiler-tree delta is the reviewed producer-inaccessible package extension. |
| SDK-free test dependency correction | `a1d900feb6c389769392fa0f79670951ddf678fa`, integrated unchanged as `9f7f9af80d96c8f38d4f66097a5c8779fb185464`: one test-local `find_package(Threads REQUIRED)`, static ACCEPT. |

The reviewer authored the contract-first real-Session/mocked-device test
`compiler/tests/generated_gpu_fused_pair/frontier_test.cpp` at
`c191bff9f82f19a4514a3f2a8017a59ec991d134` before inspecting the runtime body;
integration retained it unchanged as `24010a0`. Production issuer, adapters,
source runner and package wiring were authored by other agents. This review
inspected their code, actual artifacts, saved logs and command ledgers. The
reviewer did not execute the reported compiler, device or package runs.

The [combined realization decision and executable bounded model](accelerator-fusion-authority-review-v1.md)
remain the semantic basis. Resource contract item 8 permits implementation
failure opportunities to change with a realization. A new combined invocation
may remove a hypothetical first launch; it may not relabel an actual failure
from the old sequential adapter. Logical f1 retirement is not a claim that C
was physically produced. The earlier losing sequential-error argument remains
recorded there rather than being silently discarded.

## Issuer and actual machine boundary

The closed issuer takes a target choice and compiler context, not arbitrary
source/MLIR/LLVM input. It reuses the authenticated built-in CPU pair and fixed
upstream Transform/One-Shot row-panel schedule, then standard serial Linalg/GPU
outlining and NVVM/ROCDL lowering. A single thread owns one kernel and a caller
workspace of `min(4,M)*N` floats. There is no parallelism or performance claim.
The reference outlined operation is a verification fixture, not an executable
certificate imported from a caller.

Whole outlined-graph equivalence and whole final-LLVM fingerprints accompany
the existing structural/ABI/FP checks. The parser/builder discrepancy is
normalized only on verifier clones: absent optional `nontemporal=false` on
memref load/store is made explicit after ordinary verification. True or other
attributes remain exact, with a true-valued corruption rejection. ModuleID's
diagnostic comment is the only ignored LLVM fingerprint line.

The production image gate checks the actual ELF and disassembly, not merely
the requested compiler flags. Both targets have the expected sole kernel and
35 explicit eight-byte descriptor fields. CUDA checks all parameter records,
the exact constant-bank size, zero stack, separate arithmetic and no FTZ/FMA.
AMD binds the actual 64-byte kernel descriptor to its symbol/section, checks
explicit and hidden argument metadata, nearest rounding, preserved denormals,
IEEE mode and zero private/group segments/spills.

The one observed CUDA `@P0 CALL.REL.NOINC` is not evidence of a generally
call-free machine program. Independent `nvdisasm` JSON/control-flow inspection
bound its destination to the existing terminal, unpredicated EXIT in the sole
kernel; there is no intervening callee body or external import. Acceptance is
bounded by the pinned compiler/backend/decoder trust and the exact terminal
transfer gate, not a complete public specification of the NOINC modifier or a
general call allowlist. Non-EXIT targets and extra-function decoys are rejected.
The GPU's actual completion/failure checks remain required.

Review-driven corrections included CMake semicolon/list parsing of that valid
instruction, binding EXIT to the sole disassembled function and both T/t
symbols, and counting all CUDA parameter records rather than only size-eight
records. The final captured object controls reject 17 CUDA and 16 AMD corruptions.
The fresh composed issuer suite passed four tests, including 147 issuer checks
and 48 valid-IR corruption cases. Test passage supplements these boundaries;
it does not establish all-program or arbitrary-backend correctness.

Independently checked production image SHA256 values, equal in the owner and
fresh composed builds:

- NVVM: `6c82e0d54d351987b5b3a8cd033df8875156185067832afa6d3367128f3d04ae`.
- ROCDL: `d279b1da34ee09f05263893ee26849191e8a73466b59a1b71b8500eab0b9ffcd`.

Each owner's real normal-mode GPU run recorded 285 launches, 264 host bypasses,
148377 output checks, 2586 canary checks and 407354 immutable-input checks.
The oracle is frozen before any GPU API. Cases include noncommuting operands,
reduction/FMA discriminators, the intermediate f32 rounding boundary,
subnormals, signed zero, K=0 with Inf/NaN D, input aliases, exact work limits
and legal skinny dimensions of 65535. CUDA Compute Sanitizer reported zero
errors on the normal run; its deliberately undersized workspace control
required exit 86 and detected the generated write, followed by fail-stop with
no output exposure or live-resource reclamation. AMD has no corresponding
device-sanitizer claim.

The [issuer owner's immutable evidence report](https://github.com/onlyxItachi/MatcoreDSL/blob/950e5cabcaf42b0f22b09160fa140a0015a85729/docs/mdslc/agent-reports/gpu-fused-pair-issuer-v1.md)
records commands and the five raw-log hashes; all five were independently
rehashed. Crucially, real CUDA host-ASan failed at `cuInit`, status 2, before
module load/launch. That failed log is retained. The final physical registration
is explicitly normal-only; host-ASan API mocks and CUDA device Memcheck are
separate evidence and do not make the real vendor host-ASan path qualified.

PR #85's isolated composition changes no runtime/source admission or defaults:
CPU source fusion remains the only source-authorized pair on that branch.
Adding compiler-private images and build macros alone grants no GPU source
execution authority. Its five implementation picks and final report preserve
the accepted issuer blobs exactly.

## Runtime, source authority and failure ownership

The PR #86 production re-review found no new blocker within the declared
combined law:

- The sealed, immutable source pair derivation and unchanged original witness
  remain authoritative. No manual IR, serialized plan, manifest or image path
  issues source authority. Explicit generated target selection is required;
  automatic/default and original per-GEMM routes are not replaced.
- f1 retains A/B validity, first numeric/contraction/candidate checks, actual
  capability discovery, full logical C dimension/product/byte checks, original
  GPU limits and its own FP entry/control-check/restore. It retires guard-only,
  without a C Value or attempted arithmetic invocation. D and its profile are
  inspected only at f2.
- f2 retains D validity/contraction/candidate checks, full E extent and original
  second-GEMM limits. The tighter combined limit `M*N*(K+P) <= 2^18` is checked
  there, after f1 succeeds, not forged into the source graph. Original limits
  include each dimension <=65535, output <=`2^20` and work <=`2^26`.
- Empty E and consumer N=0 bypass the leaf only after both original guard/FP
  sequences. Producer K=0 with N>0/nonempty E still executes `+0 * D`, including
  Inf/NaN behavior. The direct private GPU entry now rejects nonempty-E/N=0;
  this review found the earlier gap between that entry and the leaf precondition.
- E is private and issued only after success, preserving an old output handle
  on failure and supporting result/input handle aliasing. A/B/D may alias;
  candidate output/workspace are separate. Earlier publications and owning
  observations are not rolled back.
- Each combined adapter reuses the original translation unit's isolated-worker
  and poison domain. All host A/B/D staging allocations precede asynchronous
  use. Uncertain upload/launch/download completion retains every potentially
  live host/device/context/stream/module resource. Known completion precedes
  release; release failure also prevents output and poisons the adapter. Both
  original and combined routes then refuse further attempts. No CPU fallback
  or partial E exposure occurs.
- Host FP state and errno remain isolated. Failure to join one's live worker
  is fail-stop, not a normal return leaving stack captures alive. The API mock
  suite injects driver/device-allocation failures, not C++ vector/new/thread
  failures; the latter argument is code inspection, not fault-injection proof.

Image ownership is the closed build issuer, actual machine gate, SHA-bound
embedding and local/Bsymbolic private DSO. `FusedPairImageAvailable`'s minimum
size check is not authentication of an untrusted image. There is no caller
image/byte/path ingress. Actual source controls reject host definitions of the
target launch symbol and `pthread_create`/`pthread_join`.

The fresh composed mocked-device suite passed 151 CTests: the independent real
Session test's 449 checks, plus 75 API scenarios in both normal and ASan/UBSan
modes. These are not physical GPU tests. Prior original-route mock regression
retained its 31 CUDA / 23 HIP good-path call counts. Source/hosted/package
qualification remains a separate obligation from those mocks.

## Connected source and package falsifiers

The work-cap fixture `7acacd5ce8b328361f1bf8e3120b793a45e1dab3` preserves an
earlier publication at f4 and observation at f5, then the pair at f6/f7.
M=64 performs exactly 262144 combined work units and succeeds. M=65 performs
266240, while each original GEMM still requires only 133120 and legal extents.
Therefore `none` succeeds but the combined realization must return
`candidate_incompatible` at f7 with completed f6/effect f5. Identical-region
copies differ only in ordinary host expectations. The fixture checks the
physical source location, unchanged output, owning observation lifetime,
input/canary contents, FP state and errno. It does not by itself count device
calls; the independent mocked Session test covers no invocation on guard refusal.

At `cb2bb73`, the actual enabled NVVM and ROCDL single-source and multi-source
tests all passed under `PYTHONOPTIMIZE=1`: four CTests, zero failures/skips.
The reviewer inspected all four saved command ledgers. Each reports 24723
checks for `none` with over-cap execution and 24724 for fusion with f2 refusal,
both zero failures. Each also retains the original 22-case parity oracle,
actual undefined pair-call symbol discrimination, three explicit interposition
refusals and unavailable-target refusal. `source_execution.py` uses explicit
`require`, not removable Python `assert` gates.

Retained correction: the original unavailable fixture omitted a return from
`main`; multi-source staging renames it to `math_main`, which does not have
main's implicit-return rule. Its SIGILL was traced to that fixture. The explicit
`return 0` in `bd92a4e129f00dee8f1b3aae1724d8e11348ff19` repairs only the host
test; it does not weaken source admission, the ABI verifier or runtime behavior.

Package extension `cf5d807e6e1fc4fbd434439bd657a086422e1001` is statically
accepted. Both cap variants and separate-program copies are staged before
deleting exact owned producer source/build directories. Every enabled target,
route and mode then uses the relocated installed driver, checks the actual
pair-call reference and requires its distinct expected cap summary. Existing
22-case parity, driver digest and producer-absence checks remain. This is
genuine source/build-inaccessible wiring, unlike the separate install-prefix-only
test, but its execution is still pending here.

The SDK-free hosted failure is also retained: PR #86 head `16bc65d`, run
`37224950148`, job `111502584458`, failed CMake generation because the frontier
test could not see `Threads::Threads`. A configure pipeline masked that exit
until the later missing `build.ninja`; this was not a numerical test failure.
The accepted one-line `9f7f9af` correction discovers Threads in the test's own
directory after the Linux/x64/driver guard. Sibling discovery is not imported
target visibility. Production, image generation and test scope are unchanged;
passing replacement hosted runs are not yet claimed.

## External receipts and remaining decision

These saved files are under
`/home/hamza-usta/mdslc-work/region-optimization-v1/evidence/`; the reviewer read
their outcomes and independently computed these SHA256 values:

| Receipt | SHA256 |
| --- | --- |
| `gpu-pair-composed-issuer.xml` | `86d945fa5f15239d16586ca2cec91c535f3e7eb0a317a77f44d0b8e1e1da559a` |
| `gpu-pair-composed-mocks.xml` | `bb941dcbfe7032b28fe64bb2cb4a86b4406c7971b255e87d264fa4d929d427bb` |
| `gpu-pair-composed-source-cap.xml` | `e49d5298c3b74e15353c2f490b6d88b083303d8bc9211f400f573d64636c5755` |
| `gpu-pair-composed-source-cap.log` | `e445a7c1f6410f621fb23d05102c18456d8e75c1ff393f2f3677fd5df3c97325` |
| `gpu-pair-composed-physical.xml` (retained failure) | `8e80c20d50d9b1ef9e63fa1c99bcdc6c3b2fa104343b44355c83dfc81de6d5c5` |

The last XML is deliberately **not** an all-pass receipt: four physical/Memcheck
tests passed and the stale previously configured CUDA host-ASan registration
failed at `cuInit(2)`. Reconfiguration to the reviewed normal-only registration
does not erase that failure or qualify vendor host-ASan.

Qualification correction: the initially running 568-test invocation inherited
`PYTHONOPTIMIZE=1` from the deliberate four-source-test experiment. Legacy Python
oracles still use `assert`, so even an all-green result from that invocation
would be auxiliary optimized-Python compatibility evidence, **not** complete
regression assertion coverage. Preserve its record; do not change legacy tests
or count disabled assertions as executed proof. The four newly inspected source
tests use explicit `require` and retain their distinct optimized-Python claim.

Pending: reconfigure/rebuild the exact clean PR head `9f7f9af`, complete the
568-test ordinary composed regression with `PYTHONOPTIMIZE` unset, its separate
safety gate and two genuine producer-inaccessible packages, fresh registration/
result accounting, and relevant exact-head hosted checks for the final PRs.
None is substituted by a partial run, optimized-away oracle or prior-head
result. CPU PR #80's independently qualified 406 local tests and all-green
hosted checks belong to the CPU boundary, not to this new GPU qualification.

No performance, GPU residency, asynchronous source execution, general fusion,
other device/toolchain, sparse/GEMV API, or universal strict-f32 proof follows.
SPIR-V/Vulkan remains a separate bounded research result, not production source
authority. Root owns integration, final evidence and any normal merge; this
reviewer has not merged either GPU PR.
