# Accelerator fused-pair authority review v1

Review date: 2026-10-04. Research base:
`3818671aadf714e6b1c76144c16190d8999ca83a`.
Branch: `research/mdslc-accelerator-fusion-review-v1`.
This review owns only this report and its adjacent executable state model.
No compiler, runtime, source gate, system package or frozen candidate changed.

## Decision

**A separately declared combined GPU realization is not forbidden merely because
the original two GPU invocations could fail independently.** Resource-contract
item 8 permits implementation failures to vary with the realization. An eligible
pure pair can potentially remove the first launch opportunity, retain every
required original guard at f1/f2, and assign its actual shared invocation and
implementation failure to f2. This is not permission to relabel an error from
an existing sequential invocation that actually ran and failed at f1.

Root selected this route for further engineering if the isolated strict kernels
pass: reuse the sealed source-pair derivation, introduce explicitly owned
target-specific combined adapters and truthful reports, and preserve defaults
and existing per-GEMM routes. **This report issues no source execution authority.**
Keep the current source pair gate closed to GPU policies until concrete issuer,
adapter, source, failure-injection, ownership and installed-artifact tests pass.
No new public mathematical operation or weaker aggregate-failure source API is
proposed. This decision elaborates a bounded realization under the existing
resource law; its applicability must still be demonstrated by implementation.

The immediate isolated prototype is one device kernel, one thread, executing
the actual upstream-derived four-row-panel schedule serially with caller-owned
global scratch. That can demonstrate arithmetic fusion, not parallel speed,
source integration or device residency. Phase-separated device-intermediate
reuse remains a different optimization and must not be labelled kernel fusion.

## Why the initial blanket objection lost

The initial review compared these injected sequential outcomes for the admitted
shape pattern A[1,1], B[1,1], D[2,1], after an earlier successful publication:

| Execution assumption | Result |
| --- | --- |
| Retain a producer invocation that fails | `candidate_failure` at f1; consumer shape guard never executes. |
| Retire f1 guards without invoking the producer; then inspect D | `shape_mismatch` at f2; no device invocation occurred. |

These are observably different status/source/completed-frontier traces. But the
comparison silently assumed that the first physical launch remained mandatory.
That is the losing assumption. It does not follow from the
[resource contract](../FOUNDATION_RESOURCE_DECISION_V1.md), whose item 8 covers
implementation failures generally, not allocation exhaustion exclusively.
There is no actual launch failure or actual pending resource in the second trace
to preserve. Counterfactual adapter poisoning is not an observable effect that
must occur after an invocation has legitimately been eliminated.

Three independent contract facts support this correction:

- Resource item 3 preserves source-required shape checks and observations, not
  every physical computation. Item 4 requires checked candidate completion
  before exposing a result. The eligible intermediate C is never exposed.
- The [source contract](../REGION_COMPILER_V1.md) explicitly does not require
  permanent intermediate materialization. Frontier IDs are diagnostic, not
  scheduling commands.
- The reviewed CPU pair already returns `completed_frontier=f1` on an f2 output
  allocation failure before producer arithmetic has run. Its documented f1 is
  logical guard retirement, not a physical-C-readiness certificate. Therefore
  that public diagnostic alone cannot prove GPU guard retirement impossible.
  See the exact [CPU source contract at 6e25df7](https://github.com/onlyxItachi/MatcoreDSL/blob/6e25df7/docs/mdslc/STRICT_FUSED_PAIR_SOURCE_V1.md).

The CPU proof itself still cannot simply be reused: it relies on a pinned
no-recoverable-error/control-preserving arithmetic leaf and conforming bounded
`memset`. A GPU adapter genuinely has recoverable errors and uncertain device
completion. The compatible route needs an explicit combined realization law
and ownership implementation; it does not acquire the CPU leaf's infallibility.

## Exact proposed realization law

1. Keep source eligibility: adjacent strict C=A*B; E=C*D, A/B/D dominating
   immutable reads, exactly one retained use of C, no live shape use or
   intervening read/publication/observation/barrier. Preserve the unchanged
   original Program/witness and both physical source records.
2. At f1, execute the original ordered begin/sticky/platform, A/B validity,
   numeric, contraction, candidate/capability, full logical C extent and GPU
   shape/work-bound checks, with any separately declared pair compatibility
   checks placed without reordering original failures. Enter, validate and
   fully restore the original FP scope. Any actual failure here remains f1.
3. Only after those succeed, retire f1 as **guard-only**. No producer invocation,
   C buffer or issued C Value is claimed. Do not inspect D or retire an f2
   failure before this point. Do not hide a failing f1 discovery/probe merely
   because no kernel has launched.
4. At f2, retain original D/numeric/contraction/candidate checks, logical E
   extent and target bounds, then checked private output/workspace preparation
   and its own complete FP scope. Extra pair-wide execution limits are explicit
   target realization predicates, not facts forged in semantic IR.
5. One new combined candidate owns the actual shared setup, transfers, launch,
   completion, cleanup and private result at f2. Its recoverable implementation
   error is reported at f2 with completed logical frontier f1 and the unchanged
   earlier completed effect prefix. No partial E escapes. An actual first
   candidate error is never cleared or reattributed to make this rule fit.
6. Confirm transfer/device quiescence and required cleanup/provider state before
   issuing E; fully restore caller controls/status on every normal return.
   Unknown completion retains every potentially live device AND host resource
   and poisons the relevant adapter. No later source access, fallback or retry
   follows sticky failure. Destruction must not free a quarantined live buffer.

This law removes a physical first launch; it does not promise that GPU failures
match the original sequential failure injection by ordinal. It retains required
semantic failures and the original order of actual guard/probe failures. Any
change to an existing sequential adapter's error handling needs a separate
justification; introducing the combined path must leave that route unchanged.

## Executable model and genuine falsifiers

Run from this research worktree:

```sh
python3 -B docs/mdslc/agent-reports/accelerator-fusion-frontier-model-v1.py
```

Observed exit 0:
`MODEL_ONLY: 147 proposed-law scenarios; 305 checks; 0 failures; no GPU/runtime/source execution`.

The [model](accelerator-fusion-frontier-model-v1.py) imports no Matcore code. It
abstracts each operation's correctly ordered internal guard bundle to its first
error; it does not itself verify that ordering, FP handling, arithmetic, resource
APIs or generated code. It starts with a healthy adapter and an earlier completed
publication. Its device outcomes are success, quiescent failure with
successful cleanup, and unknown completion requiring quarantine. Real cleanup
failure needs additional implementation tests. All cases preserve an earlier
publication. The model is executable review evidence, not a product CTest,
device experiment, plan verifier or proof over all programs.

It deliberately retains the initial sequential counterexample and verifies that
its different combined trace does not refute realization-dependent failures.
It also rejects actual violations that survive that correction:

- Hoisting a bad f2 contraction ahead of a failing **actual f1 capability guard**.
- Skipping the required full logical C extent because E is empty; e.g.
  A[INT64_MAX,0], B[0,2], D[2,0], subject to the original target guard order.
- Freeing resources after **actual** unknown completion, or issuing E on that
  failure. A missing physical launch needs no invented quarantine.
- Rewriting the first actual failure or allowing a later effect after it.

An aggregate device error alone cannot reconstruct which of two sequential
invocations failed. That is a blocker only for a route claiming to preserve that
stronger sequential attribution law. It is not a blocker for an honestly
declared single combined invocation under the law above.

The concrete production counterweight remains the existing source driver's
[forced-policy rejection fixture at 6e25df7](https://github.com/onlyxItachi/MatcoreDSL/blob/6e25df7/compiler/tests/closed_driver/strict_fused_pair.cmake#L59):
`generated-nvvm` and `generated-rocdl` with `strict-fused-pair` must reject before
publishing an executable. This review inspected that test; it did not rerun it.
Passing isolated arithmetic does not remove this gate automatically.

## Runtime and issuer implementation/review checklist

- Separate new combined registry identity, reports and generated-artifact
  ownership from the existing per-GEMM workers. No default/automatic change,
  foreign artifact substitution, manual IR/proposal authority or callback route.
  Recompute the exact original source plan and bind target/recipe identity.
- Test every original guard's priority and frontier independently of allocation
  and device fault injection. Preserve both original GPU M/N/K, output and work
  bounds, plus full C/E dimension/product/byte checks. Do not borrow the CPU
  huge-empty success expectation where GPU candidate bounds require rejection.
- Retain actual f1/f2 dynamic discovery failures in their source order; do not
  cache away the second required capability check or convert a first probe
  failure to a later shape error. Shared execution rediscovery may itself fail
  at f2 under the new law.
- Allocation-only tests must cover every new private buffer/frame/transfer and
  ensure old output-handle retention, previous publications/observations and
  whole caller FP state. Input and output handle variables may alias; inputs
  can share immutable storage. E and scratch must be disjoint from inputs and
  each other. No external descriptor inequality is a no-alias proof.
- Scratch is checked min(4,M)*N, not full M*N; output is M*P. Retain the f32
  producer store/load boundary, increasing K/N reductions, +0 seeds and separate
  mul/add. First K=0 with N>0 and nonempty E must execute 0*Inf/NaN. N=0 yields
  +0. Empty-E leaf suppression is a realization choice only after both original
  guard bundles, not permission to suppress them.
- If parallelizing later, one shared row panel cannot be written by multiple
  blocks. Give each concurrent owner proved-disjoint storage or prove a legal
  synchronized mapping. A block barrier is not a global producer-completion
  certificate. The selected one-thread prototype avoids this race but proves
  neither parallelism nor a whole-C completion checkpoint inside its row loop.
- Inject upload, launch, synchronize, download, cleanup and thread/setup errors.
  Unknown completion includes an error from an asynchronous submission that
  might have queued work. Retain pinned/ordinary host transfer storage as well
  as device memory, modules, stream/context and ownership records. Failed cleanup
  needs an explicit quarantine/poison policy even after known quiescence.
- Do not read a device phase flag after a fault as proof of completion without
  independent visibility, validity and lifetime guarantees. The chosen combined
  law needs no fictitious internal f1 phase marker.
- Report f1 as guard-only, actual none/no invocation/no C issued. Report f2's
  real combined target or truthful empty/zero path, and E only on success. Keep
  failure source location, completed logical/effect frontiers and caller FP
  state consistent on every exit, including input/result self-aliasing.
- Prove the actual binary is selected: exact issuer stage/ABI/strict-FP gates,
  artifact identity, optimized target code inspection, negative corruption
  controls, physical independent arithmetic, generated-code memory checking
  where available, and same-source none/combined tests. Then check multi-source,
  installed, relocated source/build-inaccessible and unavailable-target routes.
  Mock, sanitizer, physical, source, package and hosted evidence stay separate.

The existing GPU workers make these concrete obligations rather than hypothetical
API folklore: [CUDA worker](../../../compiler/lib/runtime/closed_cuda_candidate_v1.cpp)
isolates discovery/execution on joined threads, synchronizes before cleanup and
quarantines uncertain ownership; the [ROCDL worker](../../../compiler/lib/runtime/closed_rocdl_candidate_v1.cpp)
marks possible pending work before asynchronous submissions. Both issue output
only after their checked path succeeds. These complete files were read, not
modified or executed in this lane. CUDA 13.3 documents synchronization errors
from preceding asynchronous launches; a failing return cannot be treated as a
universal certificate that nothing was queued.
[Pinned CUDA synchronization contract](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-driver-api/group__CUDA__CTX.html)

## Accelerator inventory: lowering is not execution authority

Local read-only `/usr/lib/llvm-21/bin/llc --version` reports Ubuntu LLVM 21.1.8
and registers `nvptx64`, `amdgcn`, `spirv`, `spirv32`, `spirv64`, Hexagon, VE and
CPU targets including AArch64/RISC-V. It lists no DirectX or Metal backend.
No compiler invocation, package install or new hardware query was needed.

| Route | Actual machinery and remaining ownership |
| --- | --- |
| NVIDIA | MLIR GPU/NVVM -> LLVM NVPTX -> PTX, then vendor assembler/driver. Pinned [NVPTX conventions](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/docs/NVPTXUsage.rst) define kernel ABI and address spaces, not Matcore resource/failure semantics. Existing sm_89 single-GEMM qualification is not fused-pair qualification. |
| AMD | MLIR GPU/ROCDL -> LLVM AMDGPU -> HSA code object, then runtime. Pinned [AMDGPU guide](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/docs/AMDGPUUsage.rst) describes target/OS-specific code conventions. gfx1150 discovery and NPU enumeration are different evidence; neither authorizes a new fused recipe. |
| SPIR-V / Vulkan or OpenCL | Two seams exist: [MLIR GPU-to-SPIR-V](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/mlir/lib/Conversion/GPUToSPIRV/GPUToSPIRVPass.cpp), and the [LLVM IR-to-SPIR-V backend](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/docs/SPIRVUsage.rst). Shader versus Kernel capability, addressing, interface layout, runtime and numerical modes must match; they are not interchangeable merely because both output SPIR-V. |
| Intel GPU | SPIR-V can feed Intel device compilation through its toolchain/runtime. Intel's [compiler architecture](https://intel.github.io/llvm/design/CompilerAndRuntimeDesign.html) separates IR translation from device native compilation; [Compute Runtime](https://github.com/intel/compute-runtime) implements Level Zero/OpenCL. This is a plausible engineering route, not a tested Matcore Intel candidate or a generic LLVM machine-code backend guarantee. |
| DirectX / DXIL | [Pinned LLVM 21.1.8](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/llvm/docs/DirectXUsage.rst) labels DirectX experimental and separately enabled. The [current upstream guide](https://llvm.org/docs/DirectXUsage.html), read 2026-10-04, instead describes an official default-built target. That newer upstream status does not change the pinned local inventory, supply a reviewed MLIR recipe, or validate a D3D runtime adapter. |
| Metal | An alternate chain is SPIR-V -> [SPIRV-Cross MSL](https://github.com/KhronosGroup/SPIRV-Cross) -> Apple toolchain/runtime. Apple's separate [shader converter](https://developer.apple.com/metal/shader-converter/) accepts DXIL and produces metallib; it is not evidence that arbitrary Matcore LLVM IR is a qualified Metal target. Both bridges need numerical and runtime proofs. |
| Other backend names | Hexagon/VE registration is code-generation inventory, not a Matcore DSP/vector-engine implementation. AArch64 and RISC-V CPU targets do not establish an accelerator driver, device memory model or offload path. An enumerated `aie2p` agent likewise does not establish generic LLVM/MLIR NPU execution. |

For Vulkan strict f32, query independent RTE, signed-zero/Inf/NaN and denormal
preservation properties and encode the corresponding 32-bit execution modes.
Use NoContraction and exclude relaxed/unsafe math. The Vulkan environment makes
FP32 add/multiply correctly rounded with the declared rounding mode, but defaults
alone do not guarantee preserved denormals or operation order. A false
`shaderDenormPreserveFloat32` forbids requesting DenormPreserve32. These are
necessary arithmetic conditions, not complete source/ownership qualification.
[Float-control properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFloatControlsProperties.html),
[SPIR-V Vulkan rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).

Root's separately observed inventory reports DenormPreserveFloat32=false for
NVIDIA 615.71.09 RTX 4060 and llvmpipe, and all three relevant properties true
for RADV STRIX1 Mesa 26.0.8. This reviewer did not rerun device discovery. Refuse
the former strict native-arithmetic probes rather than treating successful
enumeration or SPIR-V validation as qualification. The latter permits a bounded
physical feasibility experiment, not automatic admission; target-specific
artifact, arithmetic and checked-completion evidence are still required.

The prior exact Metal research checkpoint `564bef047` was read: its dynamic
rank-2 outlined specimen failed static-stride SPIR-V pointer lowering; its hosted
Apple Paravirtual execution flushed positive/negative subnormals even with the
tested controlled shader translation. This disproves that tested recipe, not
all imaginable Metal implementations. Static specialization is a separate
bounded derivation; it must not quietly erase dynamic descriptor meaning.
[Exact prior Metal report](https://github.com/onlyxItachi/MatcoreDSL/blob/564bef047/docs/mdslc/agent-reports/multitarget-metal-feasibility-v1.md)

## Qualification ownership and handoff

Before this research, final independent PR #79 recommendation was MERGE at
`31401bf6002e10d09213f2e613ab2f29734def4f`: issuer bytes matched the reviewed
`8ce0bcd349cf2339162ce423c82377d3c827d642`; saved regular 400 plus two package
tests had no failures/skips; exact-head hosted checks were green. Root subsequently
reported normal merge `f55b86e`. This is CPU issuer closure, not GPU or CPU source
qualification by association.

Root owns integration/full/hosted/package gates. The GPU prototype lane owns
isolated lowering and physical arithmetic; root owns bounded SPIR-V/Vulkan
feasibility. This review contributes contract analysis, primary-source inventory
and the executed finite state model only. No new GPU math, source, package or
performance result is claimed here. The next execution-authority boundary is the
explicit combined GPU adapter/source refinement above, only after independently
qualified issued arithmetic and concrete failure/ownership/source evidence.
