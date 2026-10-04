# Compiler-issued strict two-GEMM CPU leaf v1

2026-10-04 checkpoint, branch `mdslc/strict-fused-pair-issuer-v1`, research
base `198ee591ab3379bc5ea7d42e3c15fb980f08373b`. This implements an isolated
closed issuer, not source matching, runtime binding, default dispatch or a
performance milestone. Original whole-region witness and existing single-GEMM
verifiers are unchanged. The research branch remains immutable.

## Closed scope and staged checks

The only candidate is strict rank-2 f32 `C=A[M,K]*B[K,N]; E=C*D[N,P]` with
exactly one private lhs use of C. It has no epilogue or publication. Compiler
code constructs and verifies an original built-in ClosedRegion Program; it then
constructs the structured pair using the existing canonical contraction model.
No supplied source, MLIR, LLVM, Transform program, candidate or target option is
accepted by the tool. An emitted manifest or successful public-internal verifier
is self-consistency evidence, never source/runtime execution authority.

Actual scheduling uses coherent upstream MLIR 21.1.8:

1. Tile the consumer with serial `transform.structured.tile_using_for [4,0,0]`.
2. Fuse the producer and its positive-zero fill into the row loop.
3. Apply upstream tensor empty folding and canonicalization.
4. One-Shot bufferize function boundaries to identity-layout memrefs, mapping
   exactly one certified allocation request to caller workspace with
   `allocationFn`. Fail closed on unexpected count, type, layout, dimensions,
   defining affine map, loop, IV, bound or input origins.
5. Canonicalize/CSE/canonicalize, then drop equivalent buffer results. The exact
   buffer graph has no tensor allocation, deallocation or copy.
6. Use upstream Linalg-to-loops, expand-strided-metadata, lower-affine and
   SCF/Arith/MemRef/Func/CF-to-LLVM conversions, then translate to LLVM IR.

Structured, scheduled and bufferized operation graphs are compared exactly
against compiler-private verification references, ignoring locations alone.
The references do not issue executable loops. Explicit scalar checks reject
commuted operands that a generic equivalence predicate could otherwise accept:
each matmul is input0*input1, then accumulator+product, all f32, no fastmath, with
positive-zero fills and canonical increasing reduction topology. Fixed row
tiling has no consumer column/reduction tiling; each C element is produced once
and stored as f32 before its consumer reduction. Every E element reduces N in
increasing order, and every C element reduces K in increasing order.

LLVM checks establish exact symbols/35-argument descriptor expansion, five
wrapper bindings, baseline attributes, two fmul/two fadd sites, six scalar float
loads, four float stores, input/output/workspace provenance, positive-zero seeds,
metadata-only `[2xi64]` stack storage, and only smin.i64 plus wrapper-to-leaf
calls. A pinned whole-graph fingerprint closes unexamined integer IV/bounds,
address expressions, control flow, metadata and attributes. Only the diagnostic
ModuleID comment is ignored. This is an exact 21.1.8 drift gate, **not** a general
LLVM semantic equivalence/proof checker or arbitrary LLVM import route. A future
recipe/toolchain revision must independently requalify it.

## Private ABI and remaining source obligations

Leaf: `__matcore_strict_fused_gemm_f32_v1`. C interface:
`_mlir_ciface___matcore_strict_fused_gemm_f32_v1`.

Five descriptor pointers in order A, B, D, E, workspace. Each canonical
descriptor is `{float *allocated, float *aligned, int64 offset, int64 sizes[2],
int64 strides[2]}`, 56 bytes/alignment 8, with offset zero and strides
`{columns,1}`. Shapes are `[M,K]`, `[K,N]`, `[N,P]`, `[M,P]`, `[min(4,M),N]`.
Live capacities must cover their checked extents. During each row iteration the
workspace is reinterpreted to `[min(4,M-i),N]`, offset zero, strides `{N,1}`;
the backing capacity is exactly the maximum `min(4,M)*N` bound. E/workspace are
private writable storage mutually disjoint and disjoint from all input storage.
Input-input aliases, including identical handles, are permitted; no input
noalias fact is manufactured.

The leaf requires **nonempty E**. A reviewer identified that the serial row
increment can overflow/hang on otherwise valid empty-product descriptors
`M=INT64_MAX,K=N=P=0`. The future source adapter must check/retire both original
guard frontiers first, then skip M=0 or P=0 without entering the leaf. For
nonempty E, the validated E byte extent bounds M so the row IV+4 is safe.
K may be zero: empty/null A/B must not be dereferenced, but N>0/nonempty E must
still evaluate the consumer (C is +0 and `0*Inf` is NaN). N=0/nonempty E yields
positive-zero E; the bounded leaf supports it, while a correctly guarded adapter
may retire it as a source zero-reduction case.

The leaf performs no logical shape checks, allocation, failure retirement,
publication, FP-state setup/restoration or source-frontier handling. The caller
retains both original shape/profile/host/capacity/race/FP guards, full signed
index and byte-count checks including **logical full C[M,N]**, and original
completed-frontier/error attribution. In particular `A[INT64_MAX,0], B[0,2],
D[2,0]` must fail logical C extent overflow at the first GEMM even though reads,
final output and the panel are small/empty. Do not invoke the unchecked leaf on
that case. A source matcher must reject C publications, observations, live shape
references, branch captures, additional uses, C*C, rhs chains, intervening effects
and unsupported profiles; it cannot erase the original witness to authorize a
derived schedule.

## Optimized object boundary

Both actual optimized objects are build-time gated by
`lib/regions/strict_fused_pair_object.cmake`; the tests reuse the same gate.
There are exactly two strong arithmetic ABI exports. The normal object imports
only `memset`, with scalar `mulss`/`addss` and no FMA instructions. Therefore
preoptimization LLVM call freedom does **not** imply machine-code call freedom.
Trusted conforming memset must have only the bounded private zero-fill writes,
no recoverable failure/arbitrary host effects, and preserve FP control state.
This checkpoint tests the actual host implementation and FP controls; it does
not claim to authenticate arbitrary dynamic symbol interposition.

The ASan object additionally defines exactly the common8
`___asan_globals_registered` bookkeeping symbol and imports this explicit set:
`__asan_init`, `__asan_memset`, `__asan_register_elf_globals`,
`__asan_report_load4`, `__asan_report_load8`, `__asan_report_store4`,
`__asan_version_mismatch_check_v8`, `__start_asan_globals`,
`__stop_asan_globals`. No prefix-wide sanitizer or provider/allocator allowlist
is used. ASan runtime internals are instrumentation support, not production leaf
scratch allocation. Both generated function definitions carry `sanitize_address`
before the actual Clang instrumentation pass, rather than merely sanitizing the
caller.

CMake exposes both current/parent-scope `MDSLC_FUSED_PAIR_NORMAL_OBJECT` and
`MDSLC_FUSED_PAIR_ASAN_OBJECT`, corresponding
`matcore_generated_fused_pair_{normal,asan}_object` dependency targets, and
parent-scope `*_SANITIZED`. This is a private build integration seam. It adds no
runtime capability definition, installed issuer, public API or dispatch change.

## Executed evidence

Toolchain: MLIR prefix
`/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`,
LLVM `/usr/lib/llvm-21`, `/usr/bin/clang++-21`, all 21.1.8. Linux x86_64 baseline
on AMD Ryzen AI 9 HX 370. Fresh Release/Ninja output:
`/home/hamza-usta/mdslc-work/region-optimization-v1/builds/fused-issuer`.
All builds used one compiler job, SSD TMPDIR and the shared heavy-link flock.
No LLVM/toolchain build, full compiler campaign, package build or timing ran here.

Focused CTest: **9/9 passed**, zero skips:

- 61 candidate/stage/LLVM checks; corrupt authority attrs, seed sign, fastmath,
  accumulator/input orientation, symbol/linkage, target features, C*C, row tile,
  full intermediate, workspace/output substitution, wrong panel columns, extra
  operations, stack tensors, malloc calls, wrapper bindings, integer increment,
  element address and unsafe FP-function attributes are rejected. Deterministic
  issuance is checked; the unchanged single-GEMM verifier rejects the pair.
- CLI rejects supplied source/MLIR/LLVM/Transform/target/candidate/ISA inputs.
- Normal and ASan execution each pass **13,617 checks** against a two-stage
  volatile-f32 strict oracle, including input preservation, exact output/panel
  canaries and FP controls. NaNs compare by classification; other results,
  including signed zero, compare bitwise, consistent with unspecified NaN payload.
- The 3,360 exhaustive geometries cover M0..9/K0..6/N0..7/P0..5: **2,520** invoke
  the nonempty-E leaf and **840** explicitly skip empty E. Another 41 cases make
  2,561 leaf invocations per mode. They include row tails, distinct dimensions,
  identical/partial input aliasing, FMA/order/f32-boundary/reassociation
  discriminators, NaN/Inf/signed zero/subnormals, K=0 with nonempty C/E and every
  special D, N=0 +0 output, and the four predeclared representative correctness
  shapes. Empty-result source retirement remains unimplemented here.
- Isolated undersized A, B and D negative controls all fail with ASan
  heap-buffer-overflow **READ of size 4**, frame #0 the actual generated leaf or
  its inlined C wrapper. The caller performs no OOB read. Example D control:
  `_mlir_ciface___matcore_strict_fused_gemm_f32_v1` at executable `+0x11ac46`,
  BuildId `7b7c45c38249b50eb7845cd6db500bb7ff0194b3` in this build.
- Normal/ASan object ABI/import/arithmetic gates pass during generation and CTest.

Development failures were recorded and corrected: ConstantIndexOp's inherited
static create overload is not its typed builder; the programmatic Transform
path needs Linalg/Tensor tiling external models explicitly registered; ASan's
common8 registration global must be explicitly distinguished from arithmetic
exports. No negative control was changed into a skip.

Normal stage SHA256 values:

| Stage | SHA256 |
| --- | --- |
| semantic | afef0c11ecd941a425ebb80ad533ab8175739b3a5d44c00c4536dc4c34657199 |
| structured | a084c80afa3236c938ef6759f78e858153c4c58756bff08a64812b48e0e1fb2e |
| Transform | de585a890141cccef1a39a5271bb6d9be19a1b9cf14a12bac2d2bb6c28aa23da |
| scheduled | 78974a7f936e1e998f0896da4e97a6678a613cf1600c0d167de297b34d76ff27 |
| bufferized | a85e48760d6e68e5c0403d3130afa5b4770f8a6c74b3d74cf6c9a851014e2f43 |
| LLVM normal | bc1cdf4cda81e685971c4a0ac03b33d21479b050405499e263eb36d7bf5b237c |
| LLVM ASan | 02432a532f0aa06cf2ba9c4cd81bdc4f852512d19676cfe415973a5451c8ea52 |
| object normal | c16449759cc5e6472113308c838f0bd5505b3b4b3472472bdc08bfec4b7243a6 |
| object ASan | 10892a501e67ebdb741f01cd4c96fe7ee15b95c7b3a4389dfa93bf155147d095 |

The structural claim is C scratch at most `min(4,M)*N`, not full M*N. It is not
an overall peak RSS/memory or speedup claim. The predeclared same-source fused/
unfused timing protocol remains future work after source/runtime authority:
same toolchain/candidate/FP context, full bitwise output, correctness first,
shapes (8,13,11,7), (65,63,97,31), (257,64,96,73), (3,128,256,64), warmup2/
repeats10, allocation-inclusive source time and every regression reported.
No benchmark or dispatch recommendation follows from this proof.
