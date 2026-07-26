# GEMV and GEVM kernel architecture audit

Date: 2026-07-26

Scope: future private CPU backend contracts for matrix-times-vector (`gemv`) and
vector-times-matrix (`gevm`). This audit adds no source-language declaration,
Matcore IR operation, C ABI entry point, planner candidate, or executable
kernel.

## Evidence labels and boundary

- **[source-backed]** A claim marked `source-backed` follows from the cited
  repository source or named primary reference.
- **[derived]** A claim marked `derived` is an arithmetic or algorithmic
  consequence of an explicitly stated model.
- **[measured]** A claim marked `measured` cites already reviewed MDSLC
  measurements; this lane did not run a new performance experiment.
- **[hypothesis]** A claim marked `hypothesis` requires a controlled benchmark
  or hardware-counter experiment before it may affect production planning.
- **[proposed]** A claim marked `proposed` is a design candidate, not an
  implemented or accepted contract.

**[source-backed]** Repository conclusions were checked against the production
source at `951239f1bee5541a4cf5ad72fab2192de07cf89d`. The Milestone 6 branch had
not changed the IR, public header, planner, topology, or runtime semantic
contracts when this audit was written.

**[source-backed]** The current boundary is GEMM-only:

| Current contract | Exact source anchor |
| --- | --- |
| public `matrix_view`, explicit `out`, and the only annotated operation | `compiler/include/matcore/mdsl.h:12-52` |
| Matcore IR v1 has only `OperationKind::Gemm` and rank-2 tensor values | `compiler/lib/ir/matcore_ir_v1.h:13-105` |
| the verifier requires rank 2, canonical GEMM roles, shape equalities, effects, and policy | `compiler/lib/ir/matcore_ir_v1.cpp:92-145,351-457` |
| the public tensor descriptor can represent ranks up to eight, but exported execution and reports are GEMM-specific | `compiler/include/matcore/runtime_c.h:24-36,108-138,356-480,578-741` |
| the private planner problem and registry are GEMM-specific | `compiler/lib/planner/cpu_planner.h:136-189`; `compiler/lib/planner/cpu_planner_v3.h:25-46,140-199` |
| the persistent executor accepts operation-independent task callbacks and owns no packing buffers | `compiler/lib/runtime/cpu_execution_context.h:90-127` |
| the topology model already represents processors, cores, caches, sockets, and NUMA nodes | `compiler/lib/platform/cpu_topology_v1.h:22-58,177-215` |
| GEMM packed-B storage has explicit source identity and caller-owned lifetime, but a GEMM-specific `KC/NC/NR` format | `compiler/lib/runtime/cpu_gemm_backend.h:45-75`; `compiler/include/matcore/runtime_c.h:541-576` |

**[source-backed]** Netlib LAPACK 3.12.1 defines SGEMV as either
`y := alpha*A*x + beta*y` or `y := alpha*A**T*x + beta*y`; its CBLAS surface
represents the distinction with a layout and transpose argument. The proposed
MDSLC name `gevm` is therefore a project-level semantic spelling for the
row-vector form, not a separate standard BLAS routine. See the
[Netlib SGEMV contract](https://www.netlib.org/lapack/explore-html/d7/dda/group__gemv_ga0d35d880b663ad18204bb23bd186e380.html)
and the
[LAPACK 3.12.1 CBLAS declaration](https://www.netlib.org/lapack/explore-html/de/da0/cblas_8h_source.html#l00201).

## Proposed semantic contracts

### GEMV

**[proposed]** The narrow first semantic contract should be:

```text
A: f32[M,K], row-major contiguous, read-only, host
x: f32[K],   contiguous,           read-only, host
y: f32[M],   contiguous,           write-only, host

y[i] = sum(k = 0..K-1, A[i,k] * x[k])
```

**[proposed]** `M` and `K` should be positive, accumulation and output should be
F32, execution should be synchronous, the target should be CPU, and fallback
should remain `error`. The operation should allocate and copy nothing.

**[proposed]** Output must not overlap either input. Matrix and vector may
overlap because both are read-only, provided an implementation does not impose
a stronger `restrict` promise between them.

**[proposed]** The initial operation should overwrite `y`; it should not
silently inherit BLAS `alpha`/`beta` behavior. Adding nonzero `beta` would make
the output read-write instead of write-only and must be represented as a
versioned semantic/effects change.

### GEVM

**[proposed]** The narrow first semantic contract should be:

```text
x: f32[K],   contiguous,           read-only, host
A: f32[K,N], row-major contiguous, read-only, host
y: f32[N],   contiguous,           write-only, host

y[j] = sum(k = 0..K-1, x[k] * A[k,j])
```

**[proposed]** GEVM should use the same dtype, positivity, residency,
synchronization, allocation, copy, alias, target, and fallback rules as GEMV.

**[derived]** GEVM is mathematically `A^T*x`, but keeping distinct capture-level
operation kinds preserves the user's operand orientation, diagnostics, and
row-major access intent. Both kinds can lower to one private matvec problem
with an explicit orientation enum.

### What must remain excluded initially

- **[proposed]** Exclude strided vectors, negative strides, column-major
  matrices, transpose/conjugate flags, batched vectors, mixed dtypes,
  non-host memory, in-place output, implicit allocation, hidden matrix
  transposition, and asynchronous execution.
- **[proposed]** Reject zero or negative dimensions consistently with the
  existing positive-dimension CPU runtime rather than inventing a different
  quick-return rule at the backend.
- **[proposed]** Keep GEMV and GEVM out of the public header until the typed IR,
  verifier, generated wrapper, ABI, planner, runtime, package, and Linux/Windows
  execution gates exist together.

## Arithmetic intensity and performance ceiling

### GEMV

**[derived]** Using the conventional two-FLOP multiply-add count, GEMV performs
`2*M*K` FLOPs. An ideal unique-data traffic lower bound for write-only output is

```text
bytes_gemv_min = 4 * (M*K + K + M)
AI_gemv_max    = 2*M*K / bytes_gemv_min
```

**[derived]** For large `M` and `K`, the bound approaches `0.5 FLOP/byte`.
Write-allocate traffic, repeated vector fetches, cache conflicts, and page
walks can only lower the realized operational intensity. This is a lower bound
on algorithmic bytes, not a claim about measured DRAM traffic.

### GEVM

**[derived]** GEVM performs `2*K*N` FLOPs, with the analogous lower bound

```text
bytes_gevm_min = 4 * (K*N + K + N)
AI_gevm_max    = 2*K*N / bytes_gevm_min
```

**[derived]** Its large-problem bound also approaches `0.5 FLOP/byte`.

### Consequence

**[source-backed]** The Roofline model bounds attainable throughput by the
minimum of compute peak and bandwidth times operational intensity. See
Williams, Waterman, and Patterson, “Roofline: An Insightful Visual Performance
Model for Multicore Architectures,” CACM 52(4), 2009,
[doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785), also
indexed by the
[Lawrence Berkeley National Laboratory Roofline project](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/ppan-roofline-publications/).

**[derived]** A large cold matrix-vector product is therefore expected to hit a
memory- or cache-bandwidth ceiling long before a well-blocked GEMM reaches its
compute ceiling. For comparison, ideal GEMM intensity
`2*M*N*K / (4*(M*K + K*N + M*N))` grows with square problem size because matrix
elements are reused across many outputs.

**[proposed]** Future benchmarks must measure sustainable bandwidth at the same
thread count, affinity, page placement, cache mode, and matrix size as the
kernel. A numerical `GFLOP/s` ceiling must not be published from nominal memory
speed.

**[hypothesis]** Once the matrix is larger than the relevant shared cache,
matrix bandwidth will dominate both operations. When the matrix remains
cache-resident across repeated calls, the limiting level may shift to
L2/L3 bandwidth, instruction issue, or reduction cost; only cache-state and
counter evidence can classify that region.

## Row-major access and vectorization

### GEMV row-dot kernel

**[derived]** A row-major GEMV row is contiguous, so a row-dot kernel consumes
each matrix cache line completely and walks `x` sequentially. Every output is
an independent reduction.

**[proposed]** The primary native kernel family should process several rows at
once. It should load one vector of `x`, load the corresponding vector from each
active row, and update independent row accumulators. Reusing the `x` load across
rows increases instruction-level parallelism and reduces load pressure
relative to running one independent dot-product loop per row.

**[derived]** Each output still needs one horizontal vector reduction. More
independent row accumulators can hide FMA latency, but consume registers; MR and
K unroll must be selected separately for AVX2 and AVX-512 and verified in exact
assembly.

**[proposed]** K tails must use masked or scalar loads that never read beyond
the vector. M tails should reduce only live row accumulators; computing padded
phantom rows would waste a large fraction of work when `M` is small.

### GEVM row-stream kernel

**[derived]** A column-by-column implementation of row-major GEVM would use one
float from each matrix row before jumping by `N` floats. For large `N`, that
approach wastes cache-line payload and stresses translation; it must not be the
primary layout.

**[proposed]** Two no-pack families deserve measurement:

1. `column-blocked accumulators`: choose a cache-line-aligned N block, hold
   several output vectors in registers, loop over K, broadcast `x[k]`, and load
   contiguous chunks from row `A[k,:]`;
2. `row-stream AXPY`: loop over K, broadcast `x[k]`, stream a wider contiguous
   part of `A[k,:]`, and update a cache-resident output tile.

**[derived]** The column-blocked form writes each output once and requires no
horizontal reduction. It rereads `x` for each N block, normally from cache. The
row-stream form has the simplest full-row matrix stream but repeatedly
loads/stores output; it is plausible only while the active output tile remains
in a fast cache.

**[hypothesis]** Column-blocked accumulation should win for output blocks that
fit the vector register budget, while row-stream AXPY may win for moderate `N`
when its output remains hot. A controlled N/K sweep is required; this is not a
planner rule.

**[proposed]** N-block boundaries should be at least one cache line and a
multiple of the ISA vector width. The final block must use exact masked/scalar
stores without adjacent-worker false sharing.

## Packing, prepacking, and repeated use

**[derived]** A one-shot direct matvec reads one matrix image. Transiently
packing that matrix adds a complete source read and packed write before the
kernel reads the packed copy, adding at least two matrix-sized data streams to
an operation whose ideal intensity is already at most about `0.5 FLOP/byte`.
Transient packing is therefore not a credible default for large one-shot GEMV
or GEVM.

**[measured]** Existing GEMM evidence shows why matrix lifetime matters:
prepacking B changed `1x4096x4096` native packed GEMM from 7.882 ms to 1.324 ms,
while only changing `1024x1024x1024` from 15.868 ms to 15.537 ms
(`docs/performance/cpu/milestone-4-single-thread-calibration-2026-07-22.md:146-158`).
Those are GEMM measurements, not GEMV results, but they demonstrate that
amortization depends strongly on reuse and aspect ratio.

**[proposed]** The direct no-pack path should be the initial native matvec path.
A persistent transformed-matrix path may be evaluated only for repeated
execution and must be explicit.

**[proposed]** Any persistent transformed-matrix descriptor must be
caller-owned and record at least:

- operation orientation and format version;
- source identity, dimensions, layout, stride, dtype, and alignment;
- ISA and vector/block parameters used to create it;
- packed/transposed storage address, byte size, and alignment;
- caller-provided content generation or an explicit promise that the source
  remains unmodified;
- lifetime, invalidation, concurrency, and maximum-storage rules.

**[source-backed]** The current GEMM `matcore_packed_b_desc_v1` provides useful
identity/lifetime precedent but encodes `K/N/KC/NC/NR` and GEMM provenance
(`compiler/include/matcore/runtime_c.h:541-576`). Reusing it for matvec would
authenticate the wrong format; a new versioned descriptor is required if
pretransformation proves worthwhile.

**[proposed]** Repeated vectors against one matrix should also expose a batched
or matrix-shaped lowering opportunity. Stacking vectors converts the work into
GEMM and increases matrix reuse; the compiler should not execute a long loop of
independent GEMVs when an explicit, semantically legal batched operation is
available.

## Cache lines, TLBs, and prefetch

**[derived]** With F32 and 64-byte cache lines, an aligned contiguous matrix
stream uses 16 elements per interior line. GEMV row traversal and GEVM N-block
traversal can consume all 16; naïve GEVM column traversal may consume only one
element before jumping to another line.

**[derived]** Both operations touch approximately
`ceil(matrix_bytes/page_size)` matrix pages. Large pages can reduce TLB
pressure, but page policy and allocation belong to the caller/runtime context;
the compiler must not silently remap memory.

**[hypothesis]** Hardware stream prefetch should handle long contiguous GEMV
rows and the contiguous segment within each GEVM matrix row. GEVM's jump to the
same N block in the next row may benefit from a bounded next-row prefetch when
the row stride is large, but it may also pollute caches or exceed useful
prefetch distance.

**[source-backed]** AMD's *Software Optimization Guide for the AMD Zen5
Microarchitecture*, publication 58455, revision 1.00 (2024-08-15), is an
official reference relevant to Zen 5-class tuning analysis:
[AMD document 58455](https://docs.amd.com/v/u/en-US/58455_1.00).
It is architecture guidance, not evidence that a particular software prefetch
improves MDSLC.

**[proposed]** Software prefetch must remain off until interleaved measurements
show a repeatable end-to-end benefit across cold/hot modes without regressing
other shapes. Assembly presence alone is not acceptance evidence.

## Threading and topology

### GEMV

**[derived]** GEMV rows are independent. Partitioning contiguous row ranges
gives each worker disjoint output cache lines, full matrix-row streams, and
read-only sharing of `x`; no cross-worker arithmetic reduction is needed.

**[proposed]** Row chunks should be cache-line aligned in output space where
possible and much finer than GEMM's current 128-row macro tile. Static
contiguous assignment is preferable to cyclic single rows because it preserves
matrix locality and reduces scheduler overhead.

### GEVM

**[derived]** Partitioning N gives each worker disjoint output blocks and avoids
arithmetic reduction. Every worker rereads `x`, and on row-major storage each
worker visits a different contiguous segment of every matrix row.

**[proposed]** N partitions must begin and end on cache-line boundaries except
the final tail. This avoids two workers writing one output cache line.

**[derived]** Splitting K instead would let workers consume disjoint matrix-row
ranges, but requires one private partial output per worker and a deterministic
reduction. That changes workspace, traffic, rounding order, and failure
semantics.

**[proposed]** Use N partitioning first. Consider K splitting only when N is too
small to supply tasks and only with explicit per-worker workspace and a
declared reduction order.

### Dispatch and saturation

**[source-backed]** The existing persistent executor is operation-independent,
uses fixed workers, accepts a task count and active thread count, serializes
submissions to one context, and keeps operation workspace caller-owned
(`compiler/lib/runtime/cpu_execution_context.h:90-127`). It can host private
matvec tasks without changing its public C++ type.

**[measured]** The parallel-runtime audit measured empty submission cost of
about 1.94--2.73 us for a one-worker context and 10.57--12.41 us when a
12-worker context activated one task
(`docs/performance/audits/gemm-parallel-runtime.md:68-73`). This is an executor
diagnostic on the current host, not a matvec crossover measurement.

**[derived]** A matvec whose single-thread time is near that dispatch range
cannot benefit from native pool dispatch. Small operations must remain
single-threaded.

**[hypothesis]** Large matvec will saturate memory bandwidth before using every
physical core. The exact saturation count is host-, cache-state-, and placement-
dependent and must be measured rather than inherited from GEMM's compute-based
thread threshold.

## NUMA implications

**[source-backed]** The topology record can describe NUMA membership, but the
current execution-context contract explicitly says worker affinity does not
place, migrate, bind, or interleave pages
(`compiler/lib/runtime/cpu_execution_context.h:63-78`). Existing public runtime
policy fails closed rather than claiming cross-node placement.

**[derived]** On a multi-node machine, GEMV row partitioning can preserve
matrix-page locality if the caller first-touches corresponding row ranges on
the intended nodes. Its smaller input vector may be replicated or remotely
read, but either policy must be explicit.

**[derived]** GEVM N partitioning makes every worker traverse columns across all
matrix rows, so ordinary row-major pages are not naturally partitioned by the
worker's output columns. K partitioning improves matrix-page locality but
requires partial-output reduction.

**[proposed]** Initial matvec execution should remain single-node. Cross-node
support requires an explicit matrix placement record, first-touch or allocation
contract, vector replication policy, and operation-specific decomposition.
Synthetic topology tests cannot establish physical NUMA performance.

## Private variant families

**[proposed]** The names below are descriptive internal families, not registered
stable IDs or public commitments:

| Operation | Private family | Intended region | Key legality |
| --- | --- | --- | --- |
| GEMV | portable reference | oracle and unsupported ISA | F32 host contract |
| GEMV | compiler-vectorized row dot | small/medium direct execution | compiler artifact validated |
| GEMV | AVX2/FMA multi-row dot | direct no-pack x86-64 | OS-usable AVX2/FMA |
| GEMV | AVX-512/FMA multi-row dot | direct no-pack x86-64 | OS-usable and runtime-validated AVX-512F/FMA |
| GEMV | parallel contiguous rows | large matrix stream | enough rows, bytes, worker context, one node |
| GEMV | external CBLAS | provider baseline | authenticated CBLAS and controlled threads |
| GEVM | portable reference | oracle and unsupported ISA | F32 host contract |
| GEVM | compiler-vectorized row stream | small direct execution | compiler artifact validated |
| GEVM | AVX2/FMA N-block accumulate | direct no-pack x86-64 | OS-usable AVX2/FMA |
| GEVM | AVX-512/FMA N-block accumulate | direct no-pack x86-64 | OS-usable and runtime-validated AVX-512F/FMA |
| GEVM | row-stream AXPY | cache-resident output region | measured N/cache crossover |
| GEVM | parallel cache-line N blocks | large matrix stream | enough N blocks, bytes, worker context, one node |
| GEVM | external CBLAS transpose | provider baseline | authenticated CBLAS and controlled threads |

**[source-backed]** The exact CBLAS mappings for an initial row-major contract
would be:

```text
GEMV: CblasRowMajor, CblasNoTrans, M, K, lda=K, incX=1, incY=1
GEVM: CblasRowMajor, CblasTrans,   K, N, lda=N, incX=1, incY=1
alpha=1, beta=0
```

These argument roles follow the
[LAPACK 3.12.1 CBLAS SGEMV prototype](https://www.netlib.org/lapack/explore-html/de/da0/cblas_8h_source.html#l00201).

**[proposed]** Every accelerated family must have exact artifact inspection,
forced correctness, tail/misalignment/alias tests, fail-closed capability
gating, and an operation-specific deterministic cost model before automatic
selection.

## Planner inputs and cost model

**[source-backed]** The current planner-v3 cost model and registry describe
GEMM variants, GEMM workspace, GEMM macro tiles, and operation count
(`compiler/lib/planner/cpu_planner_v3.h:20-46,104-199`). Reusing those costs for
matvec would be semantically wrong.

**[proposed]** A matvec planner should model at least:

- matrix bytes and expected cache-residency class;
- vector and output bytes;
- direct versus explicit transformed-matrix mode;
- contiguous work units (`M` rows for GEMV, N blocks for GEVM);
- ISA width and verified implementation availability;
- horizontal-reduction cost for GEMV;
- output-tile traffic for GEVM;
- persistent-worker dispatch cost;
- bandwidth saturation thread ceiling;
- requested affinity/SMT/NUMA policy;
- external-provider availability and controlled thread count.

**[proposed]** The deterministic rule should select from fixed candidate order
using saturating integer estimates and explicit rejection reasons. It must not
benchmark at runtime or infer cache-hot state from prior calls.

**[proposed]** Execution mode must be an input fact: one-shot direct, repeated
direct, or caller-owned transformed matrix. A hidden ambient cache is
incompatible with deterministic lifetime, invalidation, and diagnostics.

## Required future benchmark matrix

**[proposed]** Before promoting a native family, measure both orientations with:

- `M/K` or `K/N` spanning 1, 2, 4, 8, 16, 31, 32, 63, 64, 127, 128, 255, 256,
  511, 512, 1024, 4096, and at least one matrix larger than LLC;
- vector lengths around AVX2/AVX-512 and cache-line tails;
- matrix rows/columns that are not vector-width or cache-line multiples;
- alignments 4, 16, 32, and 64 bytes;
- one-shot, 4/16/64 repeated calls, and explicit transformed-matrix reuse;
- hot-cache and cold-cache modes;
- 1, 2, 4, and physical-core thread counts plus a separate SMT experiment;
- native families and equal-thread CBLAS;
- independent double-precision oracle, final-output authentication, and
  allocation/transform timing separated rather than hidden.

**[proposed]** Report effective matrix bandwidth as well as GFLOP/s. For a cold
large matrix, achieved bandwidth is the more interpretable metric because the
algorithmic intensity is bounded.

## IR, backend, ABI, and lifetime decisions for the later freeze

### Typed IR

- **[source-backed]** `OperationKind`, `ValueId`, JSON parsing, and the verifier
  currently accept only canonical GEMM (`compiler/lib/ir/matcore_ir_v1.h:15-31`;
  `compiler/lib/ir/matcore_ir_v1.cpp:351-457`;
  `compiler/lib/ir/matcore_ir_v1_json.cpp:324-427`).
- **[proposed]** Do not reinterpret Matcore IR v1 data as matvec. Add an explicit
  versioned schema boundary that represents GEMV and GEVM kinds, rank-1 vector
  values, matrix/vector semantic roles, shape equations, strides, and effects.
- **[proposed]** Preserve distinct source kinds, then lower to one private
  `CpuMatVecProblem` with orientation. This keeps diagnostics precise without
  duplicating legality and runtime machinery.
- **[proposed]** Decide whether a vector is truly rank 1 or a `[K,1]`/`[1,K]`
  matrix before the API freeze. Rank-1 avoids fake orientation in shape, while
  explicit source operation kind supplies multiplication orientation.

### Public source model

- **[source-backed]** The only public storage view is the non-owning rank-2
  `matrix_view` (`compiler/include/matcore/mdsl.h:14-24`).
- **[proposed]** Decide whether to add a non-owning `vector_view`, generalize to
  a rank-aware view, or deliberately use one-dimensional matrix views. This
  audit prefers a real vector semantic type but does not change the header.
- **[proposed]** Decide whether GEVM remains a distinct public function or is
  expressed through an explicit transpose/orientation view. Do not use
  overload/ADL/textual recognition; any public operation must retain canonical
  annotated declaration authentication.

### Runtime ABI

- **[source-backed]** `matcore_tensor_desc_v0` is rank-capable, but all existing
  plan, workspace, prepack, context-execution, and candidate structures are
  named and shaped for GEMM (`compiler/include/matcore/runtime_c.h:108-138,
  356-480,578-710`).
- **[proposed]** Add new versioned matvec plan and execution entry points only
  after internal semantics are accepted. Do not overload a GEMM report with a
  different operation or candidate count.
- **[proposed]** Preserve the stable C boundary: no templates, STL types,
  exceptions, implicit allocation, or runtime-owned input/output storage.

### Ownership and identity

- **[proposed]** A repeated-matrix handle needs explicit ownership, bounded
  storage, format identity, mutation/invalidation rules, and concurrent-read
  semantics. Source-pointer equality alone cannot detect in-place content
  mutation.
- **[proposed]** Decide whether callers supply a generation token, create an
  immutable matrix object, or accept a documented “source remains unmodified”
  promise before freezing a pretransform API.
- **[proposed]** Keep transformed matrices context-neutral unless topology-
  specific placement is represented. Silently attaching them to whichever
  worker first used them would make placement and lifetime non-deterministic.

### Numerical and batching contract

- **[proposed]** Freeze overwrite versus alpha/beta semantics, reduction order
  guarantees, tolerance, NaN/Inf behavior, and zero-size behavior before public
  exposure.
- **[proposed]** Decide whether repeated vectors become a batch/matrix operation.
  A batch contract can legitimately lower to GEMM and is more important for
  performance than a global transformed-matrix cache.

## Ranked implementation sequence

1. **[proposed]** Freeze private overwrite semantics and add an internal
   double-precision oracle for both orientations.
2. **[proposed]** Add a versioned typed-IR extension and verifier with no public
   header declaration.
3. **[proposed]** Implement portable direct kernels and authenticated CBLAS
   adapters.
4. **[proposed]** Implement no-pack AVX2 GEMV multi-row and GEVM N-block kernels;
   inspect exact symbols and validate every tail.
5. **[proposed]** Run the cache/bandwidth/thread matrix before adding automatic
   planning.
6. **[proposed]** Reuse the persistent executor with operation-specific task
   granularity and deterministic one-node placement.
7. **[proposed]** Evaluate explicit transformed-matrix reuse only after direct
   repeated-call evidence shows a meaningful deficit.
8. **[proposed]** Add public declarations, generated C ABI wrappers, installed
   consumer tests, and Windows validation only when all lower layers pass.

## Conclusions

- **[derived]** GEMV and GEVM are bandwidth-oriented Level-2 operations, not
  degenerate GEMM kernels that should inherit GEMM packing and thread rules.
- **[derived]** Row-major GEMV favors multi-row dot products with horizontal
  reductions; row-major GEVM favors vectorizing the output dimension and
  avoiding a column-wise strided walk.
- **[proposed]** Both should share a private orientation-aware matvec backend
  while retaining distinct typed source/IR operation identities.
- **[proposed]** Direct no-pack execution should precede any persistent matrix
  transform. Reuse must remain caller-visible and lifetime-authenticated.
- **[proposed]** The existing generic tensor descriptor, capability/topology
  records, and persistent worker callback engine are reusable. Current GEMM IR,
  reports, costs, packed-B format, and public declaration are not.
- **[proposed]** The public API decision remains deliberately open for the later
  API/ABI/backend-contract freeze milestone.
