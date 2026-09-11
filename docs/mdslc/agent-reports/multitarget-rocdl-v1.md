# AMD correctness lane: real MLIR / ROCDL / gfx1150 execution

Status: **PROVEN WITHIN A BOUNDED RESEARCH CONTRACT**, not integrated target
support. Starting canonical SHA `9c2149a25e12f5192c168f3e047097505263fc1a`;
branch `research/mdslc-rocdl-correctness-v1`. No production files changed.

## Observed environment and source identity

- Exact selected `mlir-opt`, `mlir-translate`, LLVM/Clang and LLD: 21.1.8.
  MLIR prefix: `/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`.
- HSA agent and HIP queried physical target: `gfx1150`, AMD Radeon Graphics.
  HIP reports grid limits `2147483647,65536,65536`. No gfx spoofing is accepted.
- Actual HIP header/library prefix: `/opt/rocm`; loaded runtime
  `libamdhip64.so.7.2.70201`. This is host-side runtime use, not HIP source-kernel
  compilation or a proof of every ROCm subsystem.
- The canonical private strict issuer was freshly built by the integration
  owner. Its SHA-256 was
  `46137779068edfbf944975adcf47e23f718eca788086b11989400b1ca10c0fba`.
- Issued tensor structured input SHA-256:
  `eb80297723b75c3d23ea781f483eaa269ee088cd89f7da527847710056a0f804`.
  It is the existing strict positive-zero-fill + ordered-operand GEMM specimen,
  not an inspection certificate promoted into execution permission.

## Pipeline and ownership

```text
existing private issuer -> tensor Linalg fill + matmul
  -> One-Shot Bufferize (function boundaries, identity layout)
  -> Linalg parallel loops -> GPU mapping/conversion
  -> canonicalization + index sinking + GPU outlining
  -> affine/SCF lowering -> GPU-to-ROCDL (gfx1150, HIP)
  -> operation-preserving parsed LLVM-dialect device extraction
  -> MLIR LLVM translation -> llc 21.1.8 -> ld.lld HSA code object
  -> HIP module load -> device fill -> device GEMM -> synchronization/copy
  -> independent strict host arithmetic oracle
```

One block computes one output position; its K loop remains scalar, increasing,
and uses separate f32 multiply/add. Fill is an actual separately derived device
kernel, not silently dropped or replaced with host writes. No handwritten GPU
kernel, CPU fallback, external provider call or benchmarking participates.

The extraction utility is a research bridge using real MLIR parsing/verifying,
not text rewriting of instructions. It does **not** grant source authority to
caller-supplied MLIR. A production issuer must invoke this derivation internally
from the existing trusted primitive and validate its fixed ABI/schedule.

This uses documented upstream mechanisms rather than private instruction
selection: [GPU compilation](https://mlir.llvm.org/docs/Dialects/GPU/#gpu-compilation),
[21.1.8 GPU-to-ROCDL implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp),
[LLVM AMDGPU target/ABI](https://llvm.org/docs/AMDGPUUsage.html).
The current documentation can evolve; exact locally tested tools remain pinned.

## Mechanical result, 2026-09-11

The complete documented script passed from a fresh SSD-backed `repro` directory:

| Surface | Exact result |
| --- | --- |
| Existing issuer -> upstream MLIR transformations/verifiers | PASS |
| LLVM verification, LLVM-to-gfx1150 objects and actual link | PASS |
| ISA/metadata inspection | Separate `v_mul_f32`/`v_add_f32`; no fused arithmetic; gfx1150, wave32 |
| Real HIP module loading and physical execution | **76 cases, 17,761 output comparisons, 136 kernel launches, PASS** |
| Exact strict comparison | Bitwise equality, or NaN-class equality; zero failures |
| Storage checks | Guarded device allocations and unchanged inputs, including input/input alias |
| FMA mutation | Expected physical failure: `-2^-46` instead of `+0` |
| Flush-to-zero mutation | Expected physical failure: `+0` instead of subnormal `2^-127` |

Positive cases cover seeded rectangular/noncommuting matrix products,
zero M/N/K, signed zero, actual input alias, infinity/NaN/invalid products,
subnormal inputs/results, increasing-K cancellation and a FMA discriminator.
Ordinary finite random values are bounded binary fractions; this is not a
claim over all possible f32 programs. A volatile separate-f32 host oracle is
compiled with `-ffp-contract=off -fno-fast-math` and requires nearest-even.

Explicit LLVM machine flags are necessary: merely seeing separate `fmul` and
`fadd` in LLVM is insufficient. The same IR with `-fp-contract=fast` really emits
different executed arithmetic. IEEE denormal preservation must also survive the
target machine boundary. Both negative controls discriminate executed numerical
behavior, not compilation/loading failures.

Recorded build root:
`/home/hamza-usta/mdslc-work/multitarget-v1/builds/amd/repro`.
Final fill HSACO SHA-256:
`f3f06ababceb7cbcf40da57308a3e9dd307436d60b0d5819dbf5fe1b7e513853`.
Final GEMM HSACO SHA-256:
`24e8a24dbc67e2fdde670d8289271f4b0ca90d84a44ea86a3f627634d8f94fd9`.
The script writes its complete identity table, tool versions, stage artifacts,
disassembly, metadata and positive/negative logs outside the repository.

## Rejected alternatives and unresolved obligations

- Initially tested a separate buffer Linalg fixture with host positive-zero
  initialization. Replaced it with the actual canonical issuer's tensor artifact
  and derived fill kernel, eliminating semantic duplication and implicit fill
  consumption. The standalone mirrored mathematical fixture was removed.
- Bare-pointer GPU ABI is not used: upstream requires static memref shapes;
  this experiment retains dynamic canonical descriptor arguments.
- No future MLIR distribution, LLVM source build, system package change, gfx
  override, provider fallback or tmpfs build was required.
- Input descriptors do not prove noalias. Only the fresh private destination is
  isolated. Kernel arguments retain actual input alias in an execution fixture.
- The prototype bound is M/N/K <=65535; more general launch tiling is unproven.
- Host HIP operations, module/kernel identity and original host symbol ownership
  must join the driver's trusted artifact closure before registry integration.
- A production adapter must preserve caller context, private output, explicit
  transfer/completion failures and checked resource cleanup. The standalone
  runner does not prove ordered region effects, failed-publication behavior,
  runtime concurrency isolation or a driver-failure recovery theorem.
- No sanitizer execution, hosted GPU test, performance claim, fusion,
  persistent residency, GPU public API, arbitrary GPU or general dtype support
  follows from this experiment.

## Exact next connection

Reuse one issuer-owned strict structured-to-GPU derivation shared with NVVM;
add a force-only synchronous gfx1150 candidate behind the existing private
output adapter. Bind the generated code and actual HIP library to the driver's
trusted artifact/symbol closure, preserve current device/context, reject
unavailable/oversized/illegal requests without fallback, and make all failure
paths leave public storage untouched. Then test it through authenticated region
source, including later-operation failure after an earlier publication.
