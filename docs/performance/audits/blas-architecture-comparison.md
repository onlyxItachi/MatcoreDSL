# Mature BLAS architecture comparison

Date: 2026-07-26

Status: Milestone 6 architecture archaeology. This report changes no production
kernel, planner rule, public ABI, or installed interface. No OpenBLAS or BLIS
implementation code was copied.

## Claim vocabulary

- **measured**: observed in a guarded MDSLC benchmark or exact emitted artifact
  on the declared validation host.
- **derived**: calculated from reviewed source and declared dimensions.
- **source-backed**: directly established by the cited implementation or
  authoritative documentation.
- **hypothesis**: plausible but not established by the available measurements.
- **proposed**: a candidate experiment or internal design change, not an
  accepted production decision.

## Evidence identity and boundaries

| Subject | Exact revision or version | Role in this audit |
| --- | --- | --- |
| MDSLC | `951239f1bee5541a4cf5ad72fab2192de07cf89d` | Clean post-Milestone-5 production source baseline. Later Milestone 6 commits changed audit instrumentation and documents, not the runtime files compared here. |
| OpenBLAS | tag `v0.3.32`, commit `8cecf899e21d99f9d8766ed34bfeeb3e2992c844` | Upstream source architecture and the external provider version used by MDSLC's accepted calibration. |
| BLIS | tag `2.1`, commit `caf0db6be1202d9c83c79b51e75eceb96aa8b556` | Upstream framework, context, microkernel, packing, small-GEMM, and threading contracts. |
| LLVM | tag `llvmorg-21.1.8`, commit `2078da43e25a4623cab2d0d60decddf709aaea28` | The same major/minor toolchain family used to build and inspect MDSLC. |
| Goto and van de Geijn | *Anatomy of High-Performance Matrix Multiplication*, ACM TOMS 34(3), 2008 | Algorithmic basis for layered packing, cache blocking, and TLB-aware reasoning. |

- **source-backed** — OpenBLAS and BLIS were inspected only as upstream
  architectural prior art. Source references below are immutable revision
  links. Neither project is a source dependency of MDSLC's native kernel.
- **source-backed** — The Goto paper warns that an L1-resident-panel model is
  too restrictive: a packed matrix can be streamed from L2, and TLB reach can
  constrain the blocking choice before cache capacity does. It also presents
  multiple possible inner kernels rather than treating loop order as an
  incidental detail.
- **source-backed** — LLVM documents `llvm-mca` as a static analyzer driven by
  LLVM scheduling models. Its throughput and resource-pressure results are not
  physical performance-counter measurements. This audit does not use an
  `llvm-mca` estimate as proof of achieved hardware throughput.
- **measured** — On the accepted Ryzen AI 9 HX 370 single-thread calibration,
  OpenBLAS 0.3.32 pthread was fastest on 26 of 30 declared shapes. MDSLC's
  packed AVX2 path reached 135.33 GFLOP/s versus OpenBLAS 154.54 GFLOP/s on
  1024-cube, while MDSLC won the K-shallow 4096x4096x64 case, 135.85 versus
  123.58 GFLOP/s. These are host- and run-bounded results, not universal
  library rankings.

## The MDSLC architecture reviewed

- **source-backed** — MDSLC uses one common packed-kernel geometry for AVX2 and
  AVX-512: `MR=4`, `NR=16`, `MC=128`, `NC=256`, and `KC=256`
  (`compiler/lib/runtime/cpu_gemm_backend.h`).
- **source-backed** — Its transient single-thread loop order is
  `NC -> KC -> MC`. A `KC x NC` B panel is packed once and reused across all M
  blocks; A is packed once for each `MC x KC` block and therefore once per N
  macro-panel (`cpu_packed_avx2.cpp`, `cpu_packed_avx512.cpp`).
- **source-backed** — The AVX2 microkernel computes a 4x16 tile with eight YMM
  accumulator chains. The AVX-512 microkernel computes the same 4x16 tile with
  four ZMM accumulator chains. Both advance one K element per loop iteration;
  neither contains explicit K unrolling, software pipelining, or prefetch
  inputs.
- **source-backed** — Incomplete M/N tiles are staged through a 4x16 stack
  buffer. Full and edge tiles enter the same checked microkernel symbol.
- **source-backed** — Parallel packed GEMM packs the full B matrix once into
  caller-owned workspace, publishes it read-only before workers start, and
  assigns `MC=128` row bands of C. It has no N-way macro-tile decomposition.
- **source-backed** — The runtime already has an explicit, versioned workspace
  contract and a caller-owned prepacked-B descriptor with dimensions, block
  geometry, source identity, storage extent, and provenance checks. Large
  hidden allocations are intentionally excluded.
- **source-backed** — OpenBLAS integration calls row-major CBLAS SGEMM and
  controls the provider's local thread count. The deterministic planner keeps
  native and external candidates distinct and explains legality and selection.

These properties matter because MDSLC is not starting from an unblocked triple
loop. It already implements the central packed-GEMM layering. The comparison
below identifies missing specialization and control contracts around that
layering.

## Architecture comparison

| Concern | MDSLC Milestone 5 | OpenBLAS 0.3.32 | BLIS 2.1 | Implication for MDSLC |
| --- | --- | --- | --- | --- |
| Dispatch | Fixed registry of explicit stable variants; planner selects one legal implementation. | Function tables select transpose/threaded kernels; dynamic architecture initialization selects a target implementation; SGEMM may take direct, small, GEMV-forwarded, regular packed, or threaded paths. | Architecture context supplies kernels, packing functions, storage preference, and block sizes; control nodes compose the operation. | **source-backed** — MDSLC has clean semantic dispatch but fewer shape- and architecture-specific implementation choices. |
| Blocking | One `4x16 / 128x256x256` configuration is shared by AVX2 and AVX-512. | `P/Q/R` and register unrolls are target-specific. The packed driver also adjusts the final K and M blocks instead of always using a fixed remainder. | `MR/NR/MC/KC/NC`, pack schemas, kernels, and small-path thresholds live in the architecture context. Zen 3 F32, for example, records `MR=6`, `NR=16`, `MC=144`, `KC=256`, `NC=4080`, with separate small-path values. | **source-backed** — MDSLC lacks a versioned internal kernel-configuration record. **hypothesis** — This contributes to shape-family losses, but the existing limited block sweep did not prove that a simple constant change is enough. |
| Packing | Explicit caller workspace; A and B packed into one documented schema; optional persistent caller-owned packed B. | Regular level-3 code packs A and B into internal workspace. The single-thread driver packs B segments and reuses them across later A blocks. | Packing nodes and schemas are first-class control/context data. The macro-kernel consumes panel strides rather than assuming only one layout. | **source-backed** — MDSLC does not repack B per M block. The missing opportunity is not “pack B once”; it is better panel scheduling, schema/config specialization, and reducing per-call packing where identity permits. |
| Microkernel contract | Internal symbols receive K, packed A/B, C, leading dimension, tile sizes, and accumulate state. Next-panel and prefetch information are absent. | Kernel signatures and packing/copy functions are architecture-specific compile-time components. | The documented GEMM microkernel ABI includes actual M/N/K, alpha/beta, arbitrary C strides, an architecture context, and `auxinfo` containing next-A, next-B, and panel strides. | **source-backed** — MDSLC's ABI is sufficient for current kernels but provides no framework-level way to communicate next-panel prefetch or multiple pack schemas. |
| Small/skinny path | Packed path pads every M edge to four rows and every N edge to 16 columns; no native no-pack GEMM family. | SGEMM has target-specific direct/no-pack predicates, small-kernel permits, and GEMM-to-GEMV forwarding for one-dimensional cases. | “sup” paths provide alternate small/skinny kernels, thresholds, and block sizes while retaining the conventional packed path as a floor. | **source-backed** — MDSLC pays known packing and padded-FMA costs on vector-like and small shapes for which mature libraries may select a different algorithm. |
| Thread decomposition | Persistent workers own independent M row bands; shared B is completely packed before dispatch. Actual workers are limited by `ceil(M/MC)`. | The reviewed level-3 threaded path factors work in M and N. Threads pack local B regions, publish cache-line-separated readiness, and reuse other workers' packed panels while computing their own A regions. | Four of five loops around the microkernel may be parallelized. Documentation ties JC/IC/JR choices to cache sharing and supports a local per-call runtime object. | **source-backed** — M-only decomposition is simple and race-free but underexposes parallelism when M is small and N is large. A 2-D schedule requires an explicit shared-panel lifecycle, not merely more queue tasks. |
| Memory ownership | Workspace and prepacked storage are explicit and caller-owned; insufficiency fails before C mutation. | The CBLAS entry obtains provider-managed BLAS memory for the regular packed path. | Framework memory brokers and packing controls manage internal blocks; an expert runtime object can be local to a call. | **source-backed** — MDSLC's explicit ownership is a product advantage and must not be discarded to imitate a library-internal allocation model. |
| Instrumentation | Benchmark reports end-to-end, reused workspace, prepacked B, and compute-only diagnostics separately. | The reviewed packed driver contains optional internal timing sections for A copy, B copy, kernel, and waiting. | Framework separation makes loop, pack, and kernel experiments possible, while public performance documentation distinguishes large conventional and small paths. | **proposed** — Add internal phase counters at MDSLC's own boundaries; do not infer phase cost by comparing inequivalent complete paths. |

## OpenBLAS findings

### Layering and shape dispatch

- **source-backed** — The CBLAS/Fortran GEMM interface validates arguments and
  then may dispatch a compatible row-major NN F32 call to a direct SGEMM path.
  It separately tests small-kernel eligibility, forwards M=1 or N=1 cases to
  GEMV where enabled, obtains internal workspace for the regular packed path,
  chooses a thread count, and finally enters single- or multi-thread level-3
  code.
- **source-backed** — The reviewed Skylake-X small-kernel permit rejects
  problems above `100^3` operations and applies transpose-dependent
  restrictions. The direct-path predicate has a different work threshold and
  an N-alignment condition. These constants are implementation examples, not
  proposed MDSLC thresholds.
- **source-backed** — Dynamic x86 dispatch checks both CPUID features and OS
  extended-state support before selecting AVX-family implementations. Target
  parameter blocks separately define register unrolls and `P/Q/R` blocking.

**Conclusion:** OpenBLAS presents one SGEMM API but does not execute one SGEMM
algorithm. This is the most important architectural distinction from MDSLC's
current native packed family.

### Regular packed driver

- **source-backed** — The single-thread driver iterates an N block, a K block,
  and then M blocks. It sizes the final K block and derives an M block from an
  L2-area target, packs the first A block, packs B segments, computes, and then
  packs subsequent A blocks while reusing the packed B data.
- **source-backed** — This is broadly the same reuse objective as MDSLC's
  `NC -> KC -> MC` path. Mature-library advantage cannot honestly be attributed
  to OpenBLAS having discovered B-panel reuse that MDSLC omitted.
- **source-backed** — OpenBLAS additionally specializes segment width around
  the microkernel, architecture parameters, copy kernels, direct paths, and
  threaded scheduling. Those layers provide more tuning degrees of freedom
  than MDSLC's common AVX2/AVX-512 geometry.

### Threaded driver

- **source-backed** — The reviewed OpenBLAS threaded driver represents each
  worker with M and N coordinates. It factors both dimensions, partitions N
  panels, packs B regions, exposes their readiness through cache-line-spaced
  shared state, reuses those panels across local A regions, and waits before
  storage reuse.
- **source-backed** — This is not a free optimization: it introduces ownership,
  publication, waiting, and storage-reuse protocols. MDSLC's current
  full-prepack-before-dispatch design deliberately avoids these races and
  barriers.
- **proposed** — Evaluate panel-granular shared-B publication only after a 2-D
  scheduler has an explicit state machine and sanitizer-tested ownership. Do
  not transplant OpenBLAS's synchronization design or implementation.

## BLIS findings

### Control and context separation

- **source-backed** — BLIS separates operation control from architecture
  context. Control data identifies the macro-kernel, microkernel, register tile,
  and packing behavior; the architecture context provides tuned kernels,
  storage preferences, block sizes, and small-problem thresholds.
- **source-backed** — GEMM's NC/KC/MC loops are separate control variants that
  recurse into the next layer. The microkernel-facing macro-kernel consumes
  packed panel dimensions and strides and gets its microkernel from control
  data.
- **source-backed** — The Zen 3 context demonstrates that regular and “sup”
  paths can use different KC/NC values and explicit M/N/K thresholds. Its exact
  values are valid evidence of architecture specialization, not evidence that
  the same values are appropriate for MDSLC's Zen 5 validation host.

### Microkernel boundary

- **source-backed** — BLIS's documented GEMM microkernel takes actual edge M/N,
  K, alpha/beta, packed A/B, C row and column strides, auxiliary data, and the
  context. Packed A is column-panel oriented and packed B is row-panel
  oriented, matching the conceptual structure used by MDSLC.
- **source-backed** — Auxiliary data can provide the next A and B micro-panel
  addresses and current panel strides. The documentation explicitly frames
  next-panel pointers as optional prefetch inputs and guarantees they are
  initialized.
- **source-backed** — BLIS documents moderate K unrolling as common but does
  not prescribe it as universally optimal. MDSLC therefore needs measured
  evidence before treating unroll depth or software prefetch as a foregone
  improvement.

### Small GEMM and threading

- **source-backed** — BLIS performance guidance explicitly warns that
  square-only measurements miss important skinny regimes. Its “sup” path is an
  alternate implementation for small or skinny problems, selected by
  architecture thresholds while the conventional packed path remains
  available.
- **source-backed** — BLIS exposes parallelism around JC (N), IC (M), JR (N),
  and IR (M), while the K loop remains serial absent a reduction. Its
  documentation maps loop choices to cache-sharing topology: for example, IC
  parallelism lets private-L2 workers share a packed B panel through a shared
  LLC.
- **source-backed** — A local `rntm_t` allows per-call thread choices without
  relying only on global environment state. This is conceptually close to
  MDSLC's explicit execution context, but BLIS's entire object/control API is
  not an appropriate dependency or public-surface template for MDSLC.

## What the comparison explains—and what it does not

### Why OpenBLAS wins most large regular shapes

1. **measured** — It does win most declared single-thread shapes on this host.
2. **source-backed** — Its upstream design combines target-selected copy and
   compute kernels, target block/unroll parameters, multiple shape paths, and a
   more flexible threaded decomposition.
3. **source-backed** — MDSLC uses one register and cache geometry for two ISAs,
   a one-step K loop, checked full/edge calls through the same symbol, and
   M-only parallel decomposition.
4. **hypothesis** — The measured large-square gap is the aggregate result of
   microkernel instruction scheduling, edge-independent per-call overhead,
   block selection, and mature copy kernels. Architecture source alone cannot
   assign a percentage to each cause.

The answer is therefore not “OpenBLAS packs while MDSLC does not,” nor “MDSLC
re-packs B for every M block.” Both are false for the reviewed regular paths.

### Static blocking

- **measured** — MDSLC's controlled `NC=512` and `KC=128` experiments did not
  produce a stable universal improvement; changes helped some shapes and hurt
  others within a noisy host environment.
- **source-backed** — Both mature libraries encode architecture- or path-
  specific blocking rather than one universal configuration.
- **hypothesis** — MDSLC needs a small family of deterministic configurations,
  not one new global constant. The exact family and crossovers remain a
  Milestone 7 measurement question.

### Packed-B reuse

- **source-backed** — Single-thread MDSLC packs each transient B `(NC,KC)` panel
  once per call and reuses it across M. Parallel MDSLC packs the full B once,
  before dispatch, and all workers read it.
- **measured** — Existing prepacked-B measurements show the important remaining
  distinction: repeated row-vector-like calls remove a dominant per-call B-pack
  cost, whereas 1024-cube improves only modestly.
- **proposed** — Keep caller-owned explicit prepacking as the durable repeated-
  execution contract. If context-owned caching is explored, require explicit B
  identity, lifetime, invalidation, size bounds, and concurrency semantics.

### Parallel rectangular behavior

- **derived** — MDSLC exposes at most `ceil(M/128)` row-band tasks. A problem
  with M=64 cannot use more than one native packed worker regardless of N.
- **source-backed** — OpenBLAS and BLIS both have mechanisms to distribute work
  along N as well as M and to share packed panels under explicit coordination.
- **proposed** — Add a deterministic `(m_ways,n_ways)` plan and a 2-D
  macro-tile scheduler. Keep K serial unless a separately specified reduction
  contract is introduced.

## Missing internal backend contracts

These are private compiler/runtime design candidates, not public API proposals.

1. **proposed — `CpuGemmKernelConfigV1`.** Record ISA requirements, MR/NR,
   MC/NC/KC, pack-schema IDs, pack functions, full-tile and edge-kernel
   functions, alignment, and supported accumulate semantics. AVX2 and AVX-512
   should not be forced to share a configuration.
2. **proposed — shape-path classification.** Deterministically classify
   no-pack-small, vector-like, tall-skinny, short-wide, regular packed, and
   repeated-prepacked problems. Classification must be inspectable planner data,
   not implicit branching in a microkernel.
3. **proposed — packed-panel descriptor.** Describe schema, logical and padded
   extents, panel strides, alignment, ownership, identity, and readiness. Build
   on the existing prepacked-B provenance checks rather than weakening them.
4. **proposed — microkernel call contract v2.** Keep it internal and allow
   actual edge sizes, C strides, separate full/edge functions, next-panel
   pointers, and an explicit prefetch policy. Do not pass a C++ object through
   the stable C ABI.
5. **proposed — 2-D thread plan.** Record M/N ways, task geometry, shared-panel
   ownership, barrier points, affinity, and per-worker workspace. Report every
   choice in planner diagnostics.
6. **proposed — phase instrumentation.** Measure pack A, pack B, kernel, C
   edge/copy, worker dispatch, and waiting at native boundaries. Keep it
   diagnostic-only and outside default hot paths.

## Ranked lessons for Milestone 7

| Priority | Finding | Confidence | Expected impact | Required proof before promotion |
| --- | --- | --- | --- | --- |
| 1 | Add shape-specific native paths, beginning with no-pack/vector-like and rectangular scheduling. | High: source-backed and existing packing measurements | High in small/vector-like and M-small/N-large regions | Forced correctness, fair complete-call timing, planner regret, sanitizer and artifact gates |
| 2 | Introduce distinct AVX2 and AVX-512 kernel configurations and hot full-tile entry points. | High: exact MDSLC artifacts plus mature-library architecture | Medium to high for regular GEMM | Exact assembly, counter/timing evidence, tails unchanged |
| 3 | Add deterministic 2-D M/N decomposition with explicit packed-B panel ownership. | High for current task underexposure; medium for final speedup | High for rectangular and high-thread-count cases | Race/TSan stress, barrier-cost measurement, equal-thread OpenBLAS comparison |
| 4 | Expand the internal microkernel contract for multiple schemas, edge kernels, and optional next-panel hints. | High architectural confidence | Enabling rather than independently measurable | At least two real kernels use it without public ABI growth |
| 5 | Explore K unrolling, software-pipelined loads, and prefetch only as isolated microkernel experiments. | Medium; mature libraries support the mechanisms but host benefit is unproven | Unknown to medium | Exact loop assembly, `llvm-mca` as a diagnostic, physical counters and timing as the decision evidence |
| 6 | Evaluate panel pipelining/double buffering after the 2-D ownership model exists. | Medium | Unknown | Packing/wait phase evidence demonstrates a bottleneck |

## Deliberate non-recommendations

- Do not copy OpenBLAS or BLIS kernels, synchronization, thresholds, or block
  constants.
- Do not replace MDSLC's explicit workspace with hidden runtime allocation.
- Do not expose planner, packing, thread-pool, topology, or microkernel types in
  the public header.
- Do not adopt BLIS's general object/control framework when a small typed
  internal record is sufficient.
- Do not infer an optimal threshold from the upstream source or one benchmark
  matrix.
- Do not add runtime autotuning or a global mutable packed-B cache.
- Do not claim that a static LLVM scheduling estimate proves physical execution
  throughput.

## Immutable upstream references

### OpenBLAS v0.3.32

- [GEMM interface dispatch, direct/small/GEMV paths, workspace, and threading](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/interface/gemm.c)
- [Single-thread packed level-3 loop](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/driver/level3/level3.c)
- [M/N-partitioned threaded level-3 loop and shared B panels](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/driver/level3/level3_thread.c)
- [Architecture-specific blocking and register unroll parameters](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/param.h)
- [Dynamic x86 capability and OS-state checks](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/driver/others/dynamic.c)
- [Direct SGEMM performance predicate](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/kernel/x86_64/sgemm_direct_performant.c)
- [Skylake-X small SGEMM permit](https://github.com/OpenMathLib/OpenBLAS/blob/8cecf899e21d99f9d8766ed34bfeeb3e2992c844/kernel/x86_64/sgemm_small_kernel_permit_skylakex.c)

### BLIS 2.1

- [GEMM control initialization](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/frame/3/gemm/bli_gemm_cntl.c)
- [GEMM macro-kernel and microkernel boundary](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/frame/3/gemm/bli_gemm_ker_var2.c)
- [Zen 3 architecture context and block sizes](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/config/zen3/bli_cntx_init_zen3.c)
- [Microkernel contract and auxiliary next-panel data](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/docs/KernelsHowTo.md)
- [Small and skinny GEMM performance architecture](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/docs/PerformanceSmall.md)
- [Loop-level multithreading and topology guidance](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/docs/Multithreading.md)
- [Small/unpacked GEMM loop implementation](https://github.com/flame/blis/blob/caf0db6be1202d9c83c79b51e75eceb96aa8b556/frame/3/bli_l3_sup_var1n2m.c)

### Algorithm and scheduling references

- [Goto and van de Geijn, *Anatomy of High-Performance Matrix Multiplication*](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf)
- [UT Austin FLAME high-performance linear-algebra references](https://www.cs.utexas.edu/~flame/laff/pfhp/references-1.html)
- [LLVM 21.1.8 `llvm-mca` documentation](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/CommandGuide/llvm-mca.rst)
- [LLVM 21.1.8 AMD Zen scheduling model source](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/X86/X86ScheduleZnver4.td)

## Audit verdict

**source-backed** — MDSLC has the correct foundational packed-GEMM layers,
explicit workspace ownership, deterministic variant selection, and safe
read-only packed-B sharing. It does not need a new conceptual GEMM foundation.

**measured and source-backed** — Its main architectural deficit relative to
mature BLAS is the narrowness of the implementation space: one common blocking
and register geometry, one packed shape path, a limited microkernel contract,
and M-only native parallel decomposition. Existing measurements already expose
the consequences on vector-like inputs and most large regular shapes.

**proposed** — Milestone 7 should first create the small internal contracts that
allow multiple kernels and 2-D schedules, then promote only the candidates that
pass complete-call correctness, artifact, counter, performance, planner-regret,
package, and sanitizer gates. Mature BLAS source is a design comparison, not a
threshold database or code source.
