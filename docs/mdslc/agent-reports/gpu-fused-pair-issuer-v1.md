# Closed strict GPU pair issuer v1: component qualification

Status: owner-tested isolated issuer/image component; independent final acceptance
and source/runtime composition remain separate. No production source/default
execution authority or performance claim is issued by this report.

## Checkpoint and owning boundary

Base `6e25df7e85ae3419d9e688541065433ff3151151`; shared build hooks `b0fe264`
(root's `3e168a1`); preserved interrupted issuer `60622b06739c98dd4640596eff54982bfb98817a`;
gate/test corrections `836e122` and `7dfcc556c935e844bb0207d97f1f7f55f7d359d9`.
The [interrupted snapshot](gpu-fused-pair-issuer-wip-v1.md) retains the initial
failures. Isolated research `40cac19`, hygiene follow-up `9a9c484`, and its
independent acceptance establish the predecessor only, not these production images.

Owned implementation: `MatcoreGpuFusedGemmCandidate.{h,cpp}`, its reviewed
`MatcoreGpuFusedGemmOutlinedV1.inc`, private CLI, `StrictGpuFusedPair.cmake`, image
gate/embedding, and `generated_gpu_fused_pair` issuer/object/physical fixtures.
The root owns adapters, source admission, error/completion law, RuntimeTests and
shared integration. The original single-GEMM path is unchanged.

The issuer accepts only a closed target enum and compiler context. No MLIR,
LLVM, Transform program, image, callback, manifest or serialized diagnostic can
authorize execution. Its exact graph predicates are diagnostic issuer
self-consistency checks, not general verification of arbitrary equivalent IR.

## Issued derivation and leaf obligations

The closed CPU pair stage builder provides genuine rank-2 strict
`C=A[M,K]*B[K,N]; E=C*D[N,P]`. Standard MLIR Transform row tiling `[4,0,0]` and
producer fusion preserve each producer element exactly once. One-Shot
Bufferization maps the sole expected allocation to caller workspace; unexpected
requests fail closed. Standard serial Linalg lowering and GPU launch/outlining
wrap that derived body as one thread in one block in one kernel, then upstream
GPU-to-NVVM/ROCDL and LLVM target lowering produce the images. No GPU arithmetic,
PTX/HIP math, custom index loops or algebraic reassociation is handwritten here.
This deliberately serial feasibility recipe is not an HPC schedule.

Kernel: `__matcore_strict_fused_gemm_f32_v1_kernel`. Expanded launch ABI is exactly
35 fields: descriptors `A,B,D,E,workspace`, each
`ptr,ptr,i64,i64,i64,i64,i64`. All have offset zero and row-major strides
`{columns,1}`. Workspace is `[min(4,M),N]`; the structural scratch bound is not a
claim about overall peak memory or residency. Inputs MAY alias. E and workspace
must be private writable live allocations, mutually disjoint and disjoint from
all inputs. The kernel allocates/frees/copies no tensor storage.

Leaf precondition: positive M/N/P; K may be zero with null empty A/B. First K=0
still executes the consumer, so `+0*Inf/NaN` becomes NaN. Empty E and N=0 host
bypasses are not leaf/source authority and do not eliminate original guard or
failure obligations. Full logical M*N signed-index/byte extent remains required
even when the final output is empty or the panel fits. Initial target dimensions
are at most 65535, both original GPU work bounds still apply, and combined
`M*N*(K+P) <= 2^18`. These are qualification bounds, not language restrictions.

The exact outlined graph binds operands, indices, CFG, increasing reductions,
positive-zero seeds, f32 producer stores, scratch, and launch sizes. Only absent
optional `nontemporal=false` is normalized on load/store comparison clones after
ordinary verification; `true` and every other attribute remain exact. No global
attribute erasure occurs. LLVM whole-graph SHA pins, excluding only ModuleID's
diagnostic comment, are:

- NVVM: `561394a4bf68154b455da4d7315ed18e1ed9e7aecc7cdc246ce98f6a80afb666`.
- ROCDL: `870ce35559e60007d9107bffa0b9a5d10f84af88a94f37510bab1d03ff9151c9`.

## Actual machine gate and toolchain

Coherent Clang/LLVM/MLIR **21.1.8**, Ubuntu revision `1:21.1.8-6ubuntu1`;
MLIR prefix `/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`.
Clang `/usr/bin/clang++-21`; LLVM tools `/usr/lib/llvm-21/bin`. CUDA ptxas
**13.3.73**, PTX8.0/sm_89, `--fmad=false`; AMD gfx1150 with llc/lld21.1.8,
IEEE denormal settings. No toolchain/source build or system change occurred.

The mandatory pre-embedding gate inspects the actual image, not just LLVM IR:
exact ELF target, no undefined imports, exactly one executable T/t symbol,
separate f32 multiply/add and no fused/matrix variants. NVIDIA binds exactly one
disassembled kernel, all 35 parameter records and their exact ordinal/offset/
8-byte size, 280-byte argument extent and zero stack. It rejects FTZ and FMA
suffix variants. Ptxas emits `@P0 CALL.REL.NOINC 0x1f40` at byte0x1f20; that is
allowed only as a bounded internal transfer to the same sole kernel's
unpredicated EXIT0x1f40. This is not a blanket machine-call-free claim or a proof
of general NVIDIA modifier semantics. Independent nvdisasm/CFG inspection
confirmed the taken exit edge and untaken loop branch on this image.

AMD binds the actual exported 64-byte kernel.kd symbol to the inspected .rodata
section/address, all 35 explicit argument kinds/offsets/sizes, no extra explicit
field, one-thread workgroup, no spills/dynamic stack, zero private/group bytes,
and its actual RSRC1 nearest/gradual/IEEE controls. The descriptor's 536-byte
total argument extent includes target ABI hidden fields, not extra explicit
launch arguments. Trusted pinned upstream backends/tools remain prerequisites;
this is a bounded drift gate, not a universal binary proof checker.

Final production-image identities:

| Image | SHA256 |
| --- | --- |
| NVVM cubin | `6c82e0d54d351987b5b3a8cd033df8875156185067832afa6d3367128f3d04ae` |
| ROCDL hsaco | `d279b1da34ee09f05263893ee26849191e8a73466b59a1b71b8500eab0b9ffcd` |

## Component outcomes and retained losing controls

The tiny build reused the qualified CPU-stage static archives from the external
`builds/source-candidate/lib` and linked our current issuer/test against coherent
shared MLIR21.1.8. This is not the root's pending fresh full composed build.

- Closed issuer test: **147 checks, 48 valid-IR corruptions rejected, PASS** for
  both targets, repeat determinism and unknown-target refusal. Corruptions cover
  row step, zero seed, operand/order/orientation, consumer bound, workspace alias,
  extra effects/allocation, launch races, nontemporal true, IV/GEP changes,
  contraction/FP modes, unknown imports/calls and wrong target.
- CLI refusal checks: PASS; no manual IR/Transform ingress admitted.
- Actual-image gate: PASS with **17 NVVM / 16 ROCDL** rejected captured-inspector
  corruptions. These are synthetic inspection attacks, not executed forged
  images. They cover FMA/FTZ/imports/extra global or local executable functions,
  parameter size-prefix/offset/extra fields, stack and target-specific descriptor
  binding/FP modes. NVIDIA also rejects a true redirected non-EXIT destination
  and another disassembly Function containing a decoy EXIT.
- Normal physical NVIDIA and AMD: each **285 kernel launches +264 arithmetic
  host bypass cases; 148377 strict outputs, 2586 canary checks, 407354 immutable
  input checks**, all PASS. Full independent C is computed on the host with
  separate volatile f32 operations, +0 seeds and increasing reductions; every
  expected output is frozen before any GPU API. Compare NaN classification
  (payload unspecified), otherwise exact bits. Host caller FP controls/flags,
  rounding mode and errno were preserved in each joined-worker run.
- CUDA Compute Sanitizer **2026.2.1.0** normal: same285 launches and strict
  counts, **ERROR SUMMARY: 0 errors**, exit0.
- Its fresh bounded undercapacity negative: physical scratch1float+2canaries
  behind a deliberately forged4x3 descriptor. The instrumented generated kernel
  reported invalid global write at `kernel+0x9b0`; synchronization returned719,
  fail-stop exposed no output and reclaimed no possibly-live resources.
  Sanitizer exit86 /2 errors is the expected negative-control PASS, not a valid
  execution. No malformed AMD launch was made.
- Real CUDA host-ASan+UBSan experiment: **FAIL at cuInit status2**, process exit2,
  before image loading/launch. This is unqualified real-driver compatibility,
  not a device arithmetic mismatch or a baseline product regression. It is
  retained and not retried with relaxed ASan settings. Real vendor physical
  registration is now normal-only for both targets. API-mock host-ASan/UBSan,
  CUDA device memcheck, and physical HIP normal are distinct scopes; physical
  HIP/global host-ASan remains separately unqualified.

Physical identities: NVIDIA GeForce RTX4060 LaptopGPU sm89, driverAPI13040
(actual DSO615.71.09); AMD Radeon Graphics gfx1150, runtime70253211, actual
HIP7.2.70201, `HSA_OVERRIDE_GFX_VERSION` absent. Architecture delimiter identity
is checked rather than accepting a spoofed gfx1150 prefix.

The suite includes tiny/tail M0..9, fixed seeded rectangles, both FMA and
reduction-order controls, first-result f32 boundary, normal/subnormal/Inf/NaN/
signed-zero cases, genuine null A/B on K0, all three physical input views
aliased, and exact-work-cap/tail cases `(8,128,128,128)`, `(64,32,64,32)`,
`(65,31,64,32)`. Skinny exact dimension-bound cases `(M,K,N,P)` are
`(1,65535,1,1)`, `(1,0,65535,1)`, `(65535,1,1,1)` and `(1,0,1,65535)`.
Dimensions65536 and combined over-cap `(65,32,64,32)` are host-refused; huge
`(INT64_MAX,0,2,0)` fails full C extent even though final E is empty. No huge
shape enters a display-GPU kernel. Sampled canaries do not prove arbitrary read
bounds; the closed exact graph is a separate structural obligation.

Retained engineering failures: optionalfalse parser/builder mismatch; scratch
GEP mutation anchor spelling; CMake semicolon-list parser rejection of the valid
internal exit transfer; literal backslash-n tokens in preserved max-dimension
host fixtures; the weak reduction-order witness rejected in predecessor
research. Corrections do not erase those failures or substitute them for device
math results. Naive parallel mapping remains a documented losing option: it
produces multiple launches/host row orchestration and does not realize this
one-kernel private-panel interface.

## External raw evidence and reproduction

Raw generated artifacts/logs remain outside the repository at
`/home/hamza-usta/mdslc-work/region-optimization-v1/builds/gpu-pair-issuer-tiny`.
No raw .log or machine image is tracked. Combined stdout/stderr was saved;
subprocess exit identities are reported above.

Frozen physical fixture source SHA256:
`8ebe0dc8dcde5b284527fff7edf1b92be26d2e86d050e57b7094b63a68ad63fb`.
Normal CUDA executable SHA256:
`03440ca60adec8a65aea5a8bd04e370e30829ebbde748c8a527d036c18dd5cd0`;
normal HIP executable:
`f69add23c96204aab72caa9161b20c07b0c116e0084612fe40cd95db8b335cb8`;
failed CUDA host-ASan executable:
`dbc9d652537faa02bcb6bdc7f4b1dda121692ed6971d2962421fb89f0bb1c52e`.

| Raw log | SHA256 |
| --- | --- |
| nvvm-physical-normal.log | `2ce03bec6a6c285e5bebf5228c13e936d8778bb97f80f7a2e2345ebe9f62bdf0` |
| rocdl-physical-normal.log | `cc376387c6b03ce0bd7edf3a9c505db7c3a1437f6cdefb76fb7e6db59861bd00` |
| nvvm-memcheck-normal.log | `0493a93a16905d0420034250f1d1061fba73c3c6bb25e05a865cfb71ff88d647` |
| nvvm-memcheck-negative.log | `abf913fda44fb900baa1449259fd76fcc8a4b36ae90621e2a4ebdc2ff8727c67` |
| nvvm-physical-asan.log (failed) | `0f2708f0d63c0eb95882c7091fe1554e04d49286bf01bb930d3118a5b62a5583` |

Normal execution command: `timeout 120s nvvm-physical-normal nvvm.pair.cubin`
(HIP substitutes `rocdl-physical-normal rocdl.pair.hsaco`). Failed host-ASan
command: `timeout 120s env ASAN_OPTIONS=detect_leaks=0 nvvm-physical-asan
nvvm.pair.cubin`, exit2. Each executable/image uses the absolute external prefix
above; output was piped with `set -o pipefail` and `2>&1 | tee LOG`.
Host builds use `/usr/bin/clang++-21 -std=c++20 -O1 -fno-fast-math
-ffp-contract=off -frounding-math -pthread`, CUDA's real driver DSO or HIP's real
runtime DSO and headers. The failed host-ASan binary additionally uses
`-fsanitize=address,undefined -fno-omit-frame-pointer`.

Memcheck commands invoke the checked repository wrapper:

```sh
cmake -DEXECUTABLE=PREFIX/nvvm-physical-normal -DIMAGE=PREFIX/nvvm.pair.cubin \
  -DMODE=normal -DSANITIZER=/usr/local/cuda/bin/compute-sanitizer \
  -P compiler/tests/generated_gpu_fused_pair/memcheck.cmake
# Replace MODE with negative; its fresh context is the only malformed launch.
```

From a fresh GPU-enabled composed build using the documented pinned tuple and
BUILD_TESTING=ON, the focused component tests are
`ctest --test-dir BUILD -R '^generated_gpu_fused_pair\.(issuer|cli|nvvm\.object|rocdl\.object|nvvm\.physical\.normal|rocdl\.physical\.normal|nvvm\.memcheck\.)' --output-on-failure -j1`.
This owner record does not claim that fresh composed command has run yet.

## Still outside acceptance

Fresh source/runtime integration, its combined error/frontier/poison/quarantine
law, actual embedded-DSO linkage, source admission and immutable witness,
publication/observation ordering, installed/relocated closed-source packages,
complete local regressions, hosted CI and independent final acceptance are the
integration owner's work. CPU guard-only f1 retirement cannot be copied into
GPU execution; the separately proposed combined GPU realization is not issued
by this isolated harness. Unknown completion/cleanup must not expose output or
reclaim live resources. No automatic target policy, public API, residency,
asynchronous support, parallel speedup, other hardware, broad fusion, rank-N or
Issue15/20 closure follows from this component.
