# NVIDIA strict-GEMM lowering feasibility v1

Research branch `research/mdslc-nvvm-correctness-v1`, starting from canonical
`9c2149a25e12f5192c168f3e047097505263fc1a`. Evidence date: 2026-09-11.
The initial research checkpoint `f73cab9` is a bounded physical lowering
experiment, **not product NVIDIA support**. Its four files change no production
compiler/runtime source, dispatch policy, public API, or CPU execution authority.
The subsequent shared issuer and checked-adapter implementation is recorded
separately below; complete source-to-executable integration belongs to the
integration branch, not the research harness.

## Result and connected derivation

**OBSERVED:** the existing compiler-owned strict GEMM structured primitive lowers
through upstream MLIR 21.1.8 to real NVIDIA code and executes correctly on the
owner's RTX 4060 Laptop GPU (`sm_89`). It is not handwritten CUDA, a replacement
arithmetic kernel, CPU emulation, or a newly admitted source program.

```text
canonical matcore-cpu-gemm-candidate closed issuer
  -> its exact paired semantic / structured siblings
  -> tensor linalg.fill(+0) + linalg.matmul
  -> One-Shot Bufferize (identity-layout dynamic rank-2 memrefs)
  -> Linalg parallel loops (M,N parallel; K serial)
  -> GPU mapping / launch / index sinking / outlining
  -> GPU-to-NVVM conversion + LLVM dialect
  -> MLIR LLVM translation -> LLVM 21 NVPTX -> PTX 8.0
  -> CUDA ptxas sm_89 cubin
  -> checked CUDA Driver API launches and explicit transfers
  -> private completed host value, then test-local publication
```

The research utility invokes the real baseline issuer built by the integration
owner. No independent textual copy of the mathematical primitive survives.
`extract_device.cpp` parses and verifies the lowered module, clones each GPU
module's already-lowered LLVM operations into a builtin module, and verifies
again. It performs no arithmetic rewriting. Its output is research input to
the LLVM translator, **not an artifact-authentication service**.

The exact upstream parallel schedule uses one block per `(M,N)` output and one
thread per block. This is deliberately inefficient; no performance was measured
or inferred. The `linalg.fill` and `matmul` become two ordered kernel launches.
Canonicalization and `gpu-launch-sink-index-computations` before outlining keep
constant indices and the K query inside kernels, producing these private ABIs:

| Kernel | Expanded arguments |
| --- | --- |
| Positive-zero fill | One rank-2 memref, 7 fields |
| Increasing-K GEMM | A/B/C rank-2 memrefs, 21 fields |

A memref expands as allocated pointer, aligned pointer, offset, two sizes, and
two strides, all pointer/index fields 64-bit. The tested layout is contiguous
row-major, zero offset, strides `{columns, 1}`. Descriptor identities do not
prove disjoint storage; A/B were additionally tested with the same actual
device pointer. Output is privately allocated and cannot alias device inputs.

## Numerical and failure contract

**PROVEN WITHIN THE TESTED BOUNDED CONTRACT:** dynamic nonnegative M/K/N, rank-2
f32, positive-zero overwrite, sequential increasing-K reduction, separate f32
multiply/add, nearest-even, subnormal preservation on tested cases, signed-zero
preservation, NaN/Inf classification with NaN payload unspecified.

LLVM code generation explicitly uses `-fp-contract=off`; CUDA assembly explicitly
uses `--fmad=false`. PTX contains `mul.rn.f32` and `add.rn.f32`, without `.ftz`,
`fma` or `mad.f32`. Final cubin disassembly contains `FMUL` and `FADD`, not
`FFMA` or `.FTZ`. The assembler unrolls parts of the scalar loop while preserving
the reduction dependency; source/semantic IR contains no target unroll factor.

The harness does not assume GPU arithmetic updates host floating-point flags.
That aspect of adapting the existing Session contract remains production work.
It creates explicit device allocations, H2D input snapshots, private device C,
two checked launches, checked synchronization and private D2H staging. Host
publication happens only after those operations succeed. A deliberately invalid
second launch after successful fill leaves the host destination untouched.
This does not prove crash atomicity or safe recovery from arbitrary device faults.

Harness limits are intentional: each dimension <= 65,535 and each matrix <=
262,144 elements. M/N zero does not launch; K zero runs the fill and empty
reduction. Input/output physical object lifetime remains a caller/test condition.
The test environment is one CUDA device/context, synchronous operations, no
asynchronous source effects, no persistent residency, and no generated region
execution.

## Validation

The clean reproducible run executes **53/53 cases** with no skips:

- Rectangular/noncommuting GEMM and odd `39 x 37 x 29` geometry.
- Zero M, N and K; DOT/outer-product geometries through the rank-2 substrate.
- Discriminating FMA and reduction-reassociation counterexamples.
- Normal-to-subnormal and subnormal output, signed zero, NaN/Inf, finite overflow.
- Two successive GPU GEMMs with lhs- and rhs-carried host immutable results.
- All host buffers alias, without premature destination mutation.
- Actual A/B device-storage alias, not merely equal host input arrays.
- Thirty-two deterministic random dynamic shape triples, seed `0x4D44534C`.
- Negative/beyond-launch-limit shape rejection before output mutation.
- Real CUDA invalid-block second-launch rejection after first-kernel completion.

Every finite result is checked against a separate f32 stepwise oracle bitwise
(NaN payloads excepted), and finite non-overflow results also against independent
double-precision accumulation with an explicit error bound. Every successful
nonempty test checks device guard words and that read-only input bytes remain
unchanged. Host output starts nonzero to falsify missing overwrite initialization.

An earlier minimal run passed 18 cases; the final suite adds alias/guard/order/
overflow/randomized coverage rather than treating that initial sample as general
proof. CUDA Compute Sanitizer `memcheck --leak-check full --error-exitcode 99`
also passed all **53/53**, with **0 errors and 0 bytes leaked in 0 allocations**.
`--report-api-errors no` suppresses sanitizer reporting of the deliberately
invalid launch; the harness still checks every Driver API result itself.

## Reproduction and artifact identity

No system package changes, LLVM source builds, Rust builds, or benchmarks.
Only one small O0 C++ utility is compiled against installed shared MLIR/LLVM.
The script refuses `/tmp` and RAM-backed filesystems and checks free disk space.
Task-local `TMPDIR` and CUDA cache live inside the selected SSD work directory.

```sh
python3 -B compiler/experiments/multitarget_nvvm_v1/run.py \
  --issuer /home/hamza-usta/mdslc-work/multitarget-v1/builds/base/bin/matcore-cpu-gemm-candidate \
  --mlir-prefix /home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21 \
  --work-dir /home/hamza-usta/mdslc-work/multitarget-v1/builds/nvidia/reproduce \
  --execute --sanitize
```

Without `--execute`, the script records compile-only evidence and does not open
the GPU. It requires preexisting tools and never downloads dependencies.
`commands.json` records exact commands, exit statuses and tool version output;
`manifest.json` binds stage bytes with SHA-256 and labels product authority false.
`execution.txt` records the case outcomes. Local raw artifacts are not committed.

Inspected tuple: Ubuntu LLVM/Clang/MLIR **21.1.8**, CUDA ptxas **13.3.73**,
PTX target `+ptx80`, physical NVIDIA driver **615.71.09**, RTX 4060 Laptop `sm_89`.

| Artifact | SHA-256 |
| --- | --- |
| Canonical structured primitive | `eb80297723b75c3d23ea781f483eaa269ee088cd89f7da527847710056a0f804` |
| Fill cubin | `2142a5eb35b130f6775e637409c48f511fbc5fa08c88cf6a647f79e0dfed8b01` |
| GEMM cubin | `5c7dc7c3b12a7372d11b1290b83216d31fd61dd074c2da2f5b42f842042c2fe4` |

## Integration seam and negative decisions

**ARCHITECTURAL IMPLICATION:** reuse `buildStrictGemmStagesV1` or a narrowly
renamed target-neutral issuer as the common strict mathematical source. Add a
separate checked GPU derivation rather than making the CPU issuer accept imported
IR. Source authentication, semantic Program, and exact paired witness remain
above this primitive. A production candidate must bind its generated device
artifact and launch ABI, check device capability/availability, preserve the
existing numerical/failure policy, stage output privately, and register as an
explicitly forced candidate without changing the automatic CPU/provider route.

- Rejected handwritten CUDA kernel: would not test the promised MLIR route.
- Removed independent handwritten Linalg mirror after the canonical issuer
  became available: direct existing semantic derivation is stronger evidence.
- Rejected arbitrary C++/JSON/MLIR import authority: successful device compilation
  cannot authenticate the original language source.
- Do not use `mgpu` void-returning convenience wrappers as a failure contract:
  the research harness checks actual CUDA error codes instead.
- No tensor cores, TF32, FMA, parallel reduction, packing, shared-memory promotion,
  fusion, target autotuning, hardware performance, residency or multi-GPU claims.
- The experiment does not connect an admitted user region to a GPU candidate;
  that is the next bounded integration task, not retroactively proven here.

## Upstream basis

The implementation follows installed 21.1.8 interfaces and the corresponding
versioned upstream implementations, not speculative support inferred from names:

- [GPU compilation design](https://mlir.llvm.org/docs/Dialects/GPU/#gpu-compilation).
- [LLVM 21.1.8 Linalg loop lowering](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/Linalg/Transforms/Loops.cpp).
- [LLVM 21.1.8 SCF-to-GPU mapping](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/SCFToGPU/SCFToGPU.cpp).
- [LLVM 21.1.8 GPU kernel outlining](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp).
- [LLVM 21.1.8 GPU-to-NVVM conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp).

## Initial research next boundary

Connect this same compiler-owned strict primitive to an opt-in, artifact-bound,
force-only NVIDIA candidate in the existing private-output Session execution
mechanism, with checked transfer/completion failures and no default policy change.
The experiment's passing arithmetic tests are necessary but not sufficient for
that source-to-executable production contract.

## Shared issuer and checked-adapter implementation

The subsequent implementation reuses `buildStrictGemmStagesV1` directly.
`MatcoreGpuGemmCandidate.{h,cpp}` (commit `c5657a6`) is a closed issuer with no
imported source/IR argument. It emits common semantic, structured, bufferized
and outlined stages plus separate fill/GEMM LLVM modules for NVVM `sm_89` or
ROCDL `gfx1150`. Target selection leaves all four common stage bytes identical.
`closed_gpu_images_v1.h` names private embedded device images, not a public
kernel ABI or an imported-artifact authority mechanism.

**MECHANICALLY CHECKED:** the outlined witness has exactly two ordered launches,
positive-zero fill, contiguous dynamic rank-2 f32 descriptors, one block per
output, one thread per block, increasing-K separate multiply/add, correct
operand/index/destination identity, and a returned original destination. Extra
operations, fast-math permission, argument/result attributes including forged
noalias, target attributes, unexpected private/workgroup storage, and unknown
execution metadata fail closed. Device export checks 7/21-field LLVM entry
signatures and rejects nonintrinsic external calls or unexpected definitions.
IEEE denormal attributes are explicit; downstream machine tools must retain
the no-contraction/no-FTZ obligations recorded in the manifest.

**OBSERVED upstream integration requirement:** unlike `mlir-opt`, an embedded
compiler does not obtain every external dialect-conversion interface merely by
loading the dialect. Explicit upstream Arith/MemRef/ControlFlow-to-LLVM interface
registration was required for GPU-to-NVVM conversion to preserve branch argument
types and finish MemRef lowering. No private replacement conversion was added.

The checked CUDA adapter exposes only existing internal CandidateInput/Output
descriptors and `Code` results. Every driver call runs in a fresh joined private
worker with default host FP state. It owns its context, device allocations and
heap-backed H2D/D2H staging; no device transfer references caller storage. It
validates shape arithmetic, pointer alignment/range, distinct private output,
actual `sm_89`, and the qualified launch envelope before device work. Inputs may
alias each other. Host output is copied only after successful execution,
quiescence, transfer and cleanup. Unknown completion or cleanup failure retains
owned resources in a reachable quarantine and poisons further adapter attempts.
This deliberately expensive correctness path makes no scheduling-performance or
persistent-residency claim.

The common integration-owned qualification envelope is M/N/K <=65,535, output
elements <=2^20 and scalar products <=2^26. These are bounded validation/device
watchdog protections for this realization, not semantic tensor limits, measured
performance thresholds or automatic dispatch decisions. Session must enforce
the same envelope before host allocation and before K-zero shortcuts.

### Independent adversarial review and losing implementation

The first adapter managed CUDA contexts on the calling thread. Independent AMD
lane review falsified its claimed caller-state preservation: a failed context
pop can leave caller context-stack state changed; CUDA discovery can mutate
caller errno/FP state; raw pointer guards were incomplete; and failed synchronous
copies do not by themselves prove host-memory quiescence.

That implementation was replaced before integration. Worker isolation,
heap-owned staging/quarantine, complete raw descriptor guards, and publication
after successful cleanup directly address the four counterexamples. Independent
re-review found no blocking defect in adapter source SHA-256
`65d57ca56fb0d7d49db351309aac203ae45ed9619113379edc63c8cec6d16f45` and independently
replayed all 33 then-existing ASan/UBSan fault modes successfully. Its suggested
extra pending-D2H case was added and passed, increasing the suite to 34 modes.

Thread isolation adds a real deployment trust obligation. Inspected libstdc++
15 `std::thread` start/join call `pthread_create`/`pthread_join`; those entry points,
thread state/RTTI/destructor symbols and system-error/terminate edges must not be
replaceable by authenticated host source. The integration owner reserves the
bounded used symbol closure alongside actual vendor-DSO exports. Conforming
allocator, exception and standard/POSIX thread runtime plus a trusted loader
remain preconditions, not a general hostile-process sandbox.
[The upstream implementation](https://github.com/gcc-mirror/gcc/blob/releases/gcc-15/libstdc++-v3/src/c++11/thread.cc)
matches local `/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.35` disassembly.

### Exact implementation validation

- Shared GPU issuer: **18/18** deterministic/structural/adversarial checks pass,
  generating both NVVM and ROCDL LLVM IR without vendor hardware requirements.
- CUDA adapter mock: **34/34** modes pass with ASan+UBSan: success, each of 31
  individual Driver API failure sites, pending H2D plus failed synchronization,
  and pending D2H plus failed synchronization. Every fault preserves sentinel
  output, caller errno/FP/TLS, and required pending staging ownership. Test-only
  fake Driver ABI functions are not linked into production artifacts.
- Real CUDA adapter: **76/76** cases, **17,761** strict f32 bit comparisons pass,
  using cubins generated by the new shared issuer. Includes seeded rectangular
  shapes, aliasing inputs, zero dimensions, FMA/order discriminators, subnormal,
  signed-zero and exceptional arithmetic; raw output alias and oversized K-zero
  cases reject. An unrelated caller CUDA context and unusual caller FP state
  remain unchanged.
- CUDA Compute Sanitizer on that same real adapter: **76/76**, **0 errors**,
  **0 bytes leaked in 0 allocations**. No API-error suppression is used here.
- Full standalone regressions, source authentication/link closure, packaging,
  Session dispatch and hosted CI are integration-owner responsibilities. These
  local leaf tests do not assert that those separate surfaces passed.

Local raw artifacts remain under
`/home/hamza-usta/mdslc-work/multitarget-v1/builds/nvidia/`: `gpu-issuer-test`,
`nvvm.{fill,gemm}.ll`, `production-{fill,gemm}.{ptx,cubin}`,
`cuda-adapter-test-asan`, and `cuda-adapter-physical-test`. All compilation used
the inspected 21.1.8 toolchain, at most one job, and task-local SSD TMPDIR.

### Current integration boundary

The reusable device issuer and adapter now support integrating a separately
forced, authenticated generated-NVVM candidate with explicit staging in the
existing Session. This report does not grant device code execution authority
to serialized IR, change default CPU/provider selection, promise host FP flags
from device arithmetic, or claim platform portability/performance/residency.
The exact admitted-source -> frozen driver/runtime/images -> executable chain
and ordered source failure-frontier tests must survive integration review and CI
before the candidate becomes a canonical supported bounded route.
