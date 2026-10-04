# Isolated strict two-GEMM GPU feasibility research v1

Status: bounded isolated compiler/physical feasibility PASS; no production,
source/runtime, independent final acceptance or performance claim.
Base: `6e25df7e85ae3419d9e688541065433ff3151151`.
Owned branch: `research/mdslc-gpu-strict-fused-pair-v1`.
Targets: physical RTX 4060 `sm_89` and Radeon 890M `gfx1150`, exact LLVM/MLIR
21.1.8. Only `compiler/tests/research/gpu-fused-pair` and this report are owned.

## Predeclared scope and falsifiers

The existing closed CPU pair stage builder constructs genuine rank-2
`C=A[M,K]*B[K,N]; E=C*D[N,P]`, exact structured pair, Transform serial row
tiling/fusion `[4,0,0]`, and One-Shot Bufferize with caller-owned workspace
`[min(4,M),N]`. Its authenticated initial/scheduled/bufferized self-checks are
reused, not relaxed. No user MLIR, Transform or LLVM enters this experiment.
Original source Program and its whole-region paired witness remain immutable.
No prototype artifact, successful verifier, manifest or test grants execution
authority to source operations.

Hypothesis 1: direct reuse of the existing GPU parallel mapper on the row-panel
pair yields separate fill/producer/consumer launches inside a host row loop,
not one fused GPU kernel. Preserve its actual IR and outcome as a losing option.
Hypothesis 2: enclose the upstream serial loop-lowered pair in one standard
`gpu.launch`, outline using upstream GPU utilities and lower through existing
NVVM/ROCDL conversions. Grid and block are each `(1,1,1)`: one thread serially
executes every row panel. This deliberately serial correctness proof makes no
parallel/HPC schedule or speed claim. The wrapper introduces no arithmetic or
handwritten indexing. Concurrent panels sharing this one global scratch are
rejected, not silently accepted.
Hypothesis 3: stop and preserve exact compiler/machine/arithmetic counterexamples
if strict arithmetic, f32 producer boundary or bounded workspace cannot survive.

Gates: exactly two f32 multiply/add sites, no reassociation/contraction/fastmath,
increasing first K and second N reductions, positive-zero seeds, C f32 storage
boundary, no producer recomputation, no FTZ, no hidden allocation/free/copy or
unknown device calls, exact five-memref expanded 35-field ABI, output/scratch
canaries, input immutability and independent strict two-step arithmetic oracle.
Cover row tails, K=0 with nonempty E and D Inf/NaN (must evaluate 0*Inf), N=0,
empty final output, subnormals, signed zero, FMA and reduction-order adversaries.
Validate full logical C extent even when final output is empty; enormous empty
shape cases are checked host protocol only, never display-GPU work.

All input handles MAY alias. Output and workspace are private and disjoint from
each other/inputs, with exact row-major capacity and live-object bounds. Device
workspace capacity is at most `min(4,M)*N` floats, not full `M*N`; this is a
structural footprint statement, not total peak RSS or measured memory saving.
Physical work is restricted to `M<=65,K<=128,N<=128,P<=128` sampled cases and
`M*N*(K+P)<=2^18`, with finite process timeouts. These are diagnostic work bounds,
not evidence for arbitrary display-GPU work or unrestricted shape execution.

GPU failure cannot inherit the CPU guard-only first-frontier retirement proof:
discovery, transfers, launch and completion may fail recoverably. A single
fused kernel's failed completion does not identify producer versus consumer
failure. The standalone launcher checks operations, exposes output only after
known successful completion and terminates without reclaiming unresolved live
resources on an uncertain completion. Source failure order/retirement is an
independent unresolved boundary. Empty E arithmetic bypass does not prove that
the original first GPU GEMM could be bypassed at its source frontier.
The corrected independent research law at
`63c5533` permits consideration of a separately declared **combined** GPU
realization with explicit shared failure/completion ownership. That law is not
implemented or qualified by this isolated harness; possible original per-GEMM
implementation failures do not categorically forbid every new combined route.

## Pinned upstream mechanisms

The official 21.1.8 [`GPUUtils.h`](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/GPU/Utils/GPUUtils.h)
documents capture order seeding for `outlineKernelFunc`.
[`KernelOutlining.cpp`](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp)
implements outlining, cloning the launch body and replacing its terminator,
while recording known grid/block sizes. Seed captures as A,B,D,E,scratch;
do not derive ABI from accidental first-use order.
[`GPUDialect.cpp`](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/GPU/IR/GPUDialect.cpp)
owns standard launch semantics/building. Target conversion and final tool flags
follow the existing [staged GPU contract](../STAGED_GPU_CANDIDATES_V1.md).
The actual AMD kernel descriptor is decoded using pinned
[`AMDHSAKernelDescriptor.h`](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h)
offsets: RSRC1 f32 round mode 0, denorm mode 3, IEEE bit 1. This verifies image
controls rather than inferring them solely from LLVM attributes.

## Evidence ledger

The preceding hypotheses and gates were declared before the prototype and any
physical GPU execution. Initial harness failures are retained: the first tiny
tool compile had a mistaken include root (corrected); the first execution
command failed its host-only order negative control **before device discovery**.
The fixture `[2^24,-2^24,1]` does not distinguish forward and reversed addition
under f32 nearest. It was corrected to `[2^24,1,-2^24]` (forward +0, reverse 1).
Neither was a GPU compiler or device arithmetic failure. Physical outcomes
below were obtained only after the distinguishing control passed. A later
descriptor-check edit had a Python `else` indentation/syntax failure, fixed
before the strengthened inspection was counted; no artifact was changed.

### Compiler and losing-option evidence

The direct existing parallel mapping passes succeeded but emitted four launch
sites: whole E fill plus panel fill, producer and consumer inside the serial
host row loop. See the saved `naive-parallel-mapping.mlir`: bounded scratch alone
does not make this one fused device kernel. It was not selected or timed.

The selected route reused the closed issuer's exact pair stages, then standard
serial Linalg loop lowering and ExpandStridedMetadata, one structural launch
wrapper, upstream `outlineKernelFunc` seeded with A/B/D/E/scratch, and existing
NVVM/ROCDL target conversions. The resulting kernel is
`__matcore_research_strict_fused_pair_kernel`, five rank-2 memrefs in that order,
35 expanded fields (each `ptr,ptr,i64,i64,i64,i64,i64`), known grid/block `(1,1,1)`.
The exact pair source loop graph is not handwritten GPU arithmetic. Two static
f32 multiply/add sites survive in exported LLVM; backend unrolling creates more
machine sites without granting reassociation permission.

Both images passed actual ELF target checks, unique executable-kernel export,
empty import set, separate FMUL/FADD or v_mul_f32/v_add_f32 and forbidden
FMA/matrix/FTZ checks. AMD descriptor also passed nearest-even, gradual f32 and
IEEE controls, zero private/group segments. Ten inspection corruption controls
were rejected (wrong CPU image, fused arithmetic, missing multiply, NVIDIA FTZ,
AMD wrong rounding/FTZ/non-IEEE descriptor). These are inspection falsifiers,
not executions of forged images and not a full CFG/index theorem.

NVIDIA ptxas reports 30 registers, zero stack frame/spill loads/spill stores and
zero barriers. AMD metadata reports 58 SGPRs, 7 VGPRs, zero spills, zero
private/group segment, no dynamic stack. The five caller buffers remain the
only tensor storage. These local counts are not a performance or RSS result.

Artifact identities (exact LLVM21.1.8 + ptxas13.3.73/lld21.1.8):

| Target | Image bytes | Image SHA256 | Exported LLVM SHA256 |
|---|---:|---|---|
| NVVM sm_89 | 12200 | `58e6562902d68aa161a60d80239ad63b8f1cde22a8fdf32f2dded351d888ba85` | `11006b35cb5c33050bec957c8d06baa316f43713a4df736c3e5908ee614f4f13` |
| ROCDL gfx1150 | 7976 | `c39b0089ab811392be0151e2ab394b129bcb52a4d0c7b8ab19f39ee06de1fd3f` | `418d95e7311221915f8b7876a3f164e0b47844b05969b2eda3276bd736671fe7` |

### Physical arithmetic and memory diagnostics

Both physical runs passed **281 launches + 264 host arithmetic bypass cases**,
17305 strict output comparisons, 2554 span-canary checks and 79676 immutable
input comparisons. NaN classification is compared, all non-NaN results bitwise;
unspecified NaN payloads are not silently promoted to exact payload semantics.
Hostile calling-thread FP controls/flags and errno survived a joined worker.
NVIDIA was `NVIDIA GeForce RTX 4060 Laptop GPU`, sm_89, Driver API version
13040. AMD was `AMD Radeon Graphics`, actual `gfx1150`, runtime 70253211;
`HSA_OVERRIDE_GFX_VERSION` absent and exact architecture/end-or-':' identity
checked. Neither run substituted a CPU result or skipped a missing device.

Cases cover tiny/zero shapes, rows 0..9 and mixed larger tails, increasing first
and second reduction adversaries, both FMA sites, rounded producer C boundary,
normal/subnormal/underflow/Inf/NaN/signed zero, null K0 A/B with nonempty C/E and
Inf/NaN D, and A/B/D all aliasing the same immutable physical input. Full C
overflow `[INT64_MAX,0]*[0,2]` with final D `[2,0]` is rejected host-side before
empty E bypass; no enormous shape enters the device.

Two actual cap-boundary shapes, `(M,K,N,P)=(8,128,128,128)` and
`(64,32,64,32)`, have `M*N*(K+P)=262144`; `(65,31,64,32)` exercises a last-row
tail at 262080. The just-over-cap `(65,32,64,32)` is rejected by the host bound.
This samples the proposed bounded work range, not exhaustive numeric execution.

CUDA Compute Sanitizer 2026.2.1 memcheck with 32-byte allocation padding ran the
entire corrected suite: **0 errors**, exit 0. A separately authorized, fresh,
bounded CUDA-only process deliberately lied about scratch capacity (descriptor
4x3 but physical payload one float + two canaries), solely under memcheck. It
reported an invalid generated-kernel global write, then synchronization error
719. The harness fail-stopped without exposing output/reclaiming live buffers;
tool exit 86, two errors. This establishes that instrumentation observes actual
generated device accesses. No malformed AMD launch was performed; AMD has
sampled canary/input/oracle evidence, not GPU memory-sanitizer qualification.
Canaries and sampled sanitizer execution are not arbitrary-shape bounds proofs.

### Reproduction and retained boundary

Run `bash compiler/tests/research/gpu-fused-pair/run.sh build OUTPUT` using the
already qualified source-candidate stage libraries, then individually reserve
each device and run `execute-nvvm`, `execute-rocdl`, `memcheck-nvvm`.
`negative-memcheck-nvvm` is intentionally undercapacity and allowed only as the
instrumented CUDA diagnostic; never treat it as a supported kernel precondition.
All builds are one compiler process at a time under the shared heavy-link lock,
external SSD TMPDIR, no LLVM/CUDA/toolchain build or system change.

Exact bufferized/outlined/LLVM stages, losing IR, object report and
physical/memcheck logs are retained in the
[research evidence snapshots](../../../compiler/tests/research/gpu-fused-pair/evidence)
and all stages locally in
`/home/hamza-usta/mdslc-work/region-optimization-v1/builds/gpu-fused-pair-research`.
No benchmark was run. No source gate, CLI refusal, runtime dispatch or original
single-GEMM candidate changed. The next engineering boundary, if separately
approved, is a closed build-issued GPU pair primitive with pinned structural
and actual-image corruption gates; combined adapter/source failure law remains
a separately owned and separately qualified dependency.
