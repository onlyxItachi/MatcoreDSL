# Private GEVM kernel design contract

Status: design-only input from Milestones 6 and 7.

MDSLC does not currently expose, capture, verify, plan, package, or execute a
public GEVM operation. This document defines a candidate private backend
contract for later investigation. It does not add `matcore::mdsl::gevm`,
extend Matcore IR, add a C ABI entry point, register a planner variant, or
freeze an API.

Primary evidence:

- [GEMV/GEVM architecture audit](../audits/gemv-gevm-kernel-design.md)
- [HPC kernel engineering handbook](../HPC_KERNEL_ENGINEERING_HANDBOOK.md)
- [public API pre-freeze decisions](../../api/PRE_FREEZE_DECISIONS.md)

Labels used below:

- **derived**: follows from the stated algorithm/data movement;
- **proposed**: candidate private behavior requiring implementation evidence;
- **not implemented**: deliberately absent from the current product surface.

## Candidate semantic contract

**Proposed:**

```text
x: f32[K],   rank 1, contiguous,           read-only, host
A: f32[K,N], rank 2, row-major contiguous, read-only, host
y: f32[N],   rank 1, contiguous,           write-only, host

y[j] = sum(k = 0..K-1, x[k] * A[k,j])
```

The project spelling `gevm` distinguishes this row-vector-by-matrix
orientation from GEMV. Mathematically it is equivalent to `A^T * x`, but the
row-major access pattern and optimized kernel family differ from a GEMV that
streams contiguous rows of A. A future IR may encode a common matrix-vector
operation plus orientation, but the source operation kind must not be erased
before legality and diagnostics can preserve the user's intent.

The first private implementation should require:

- positive concrete K and N;
- F32 inputs, accumulation, and output;
- synchronous CPU execution;
- overwrite semantics (`alpha=1`, `beta=0`);
- no overlap between y and either input;
- no hidden allocation, matrix transform, copy, or migration;
- no strided/negative-stride vectors, column-major matrix, batching,
  transpose flag, mixed dtype, asynchronous execution, or non-host memory;
- explicit failure for every unsupported condition.

x and A are both read-only and may overlap semantically. An optimized
implementation must not assert a stronger no-alias relation unless legality
proves it. Alpha/beta, zero-size behavior, NaN/Inf policy, reduction order,
and rank-1 representation require explicit versioned decisions before public
exposure.

## Data-movement ceiling

**Derived:** GEVM performs `2*K*N` FLOPs. An optimistic unique-payload lower
bound is:

```text
4 * (K*N + K + N) bytes
```

Its large-problem arithmetic-intensity ceiling approaches 0.5 FLOP/byte. A
cold large operation is therefore ordinarily cache- or memory-bandwidth
oriented. Transiently repacking the complete matrix is a poor default because
it adds a matrix read, packed write, and packed reread to an operation that
otherwise streams A once.

Unlike a naive column-dot implementation, a row-major kernel must not traverse
`A[k,j]` with an N-element stride for one j at a time. That discards
cache-line utilization and raises TLB pressure.

## Candidate direct kernels

### N-block accumulator

**Proposed primary design:**

1. Select a contiguous live output block.
2. Initialize vector accumulators for that block.
3. For each k, broadcast `x[k]`.
4. Load contiguous vectors from row `A[k, :]`.
5. FMA into the output-block accumulators.
6. Store only the live y lanes after the K reduction.

This preserves row-major streaming and reuses each `x[k]` across the output
block. The N block, number of accumulators, K unroll, and load schedule must be
ISA-specific. AVX2 must fit broadcasts, matrix loads, and accumulators into 16
YMM registers. AVX-512 may widen the block or increase accumulator chains, but
requires exact-symbol artifact evidence and physical timing before promotion.

N tails require bounded masked or scalar loads and stores. K is a scalar
reduction loop; a K tail is not a vector-tail problem for this design.
Software prefetch remains a measured hypothesis, not a contract.

### Row-stream update

**Proposed alternative:** initialize y, then stream each A row and perform a
scaled vector update:

```text
y[:] += x[k] * A[k,:]
```

This gives simple contiguous access but repeatedly reads and writes y. It may
be competitive when y stays in L1/L2 and N is modest. It must not be selected
for large N without measured cache-traffic evidence. If overwrite
initialization is timed separately, the benchmark must report that boundary.

The two designs are separate private candidates. They must not share one ID
while silently changing measurement semantics.

## Optional transformed-matrix mode

Direct no-pack execution comes first. A transformed or transposed A may help
repeated-vector use, but it is a caller-visible storage and lifetime decision,
not a hidden optimization.

Any future transformed descriptor must record:

- GEVM orientation and format version;
- source address, dimensions, layout, stride, dtype, and alignment;
- transformation, ISA, and block parameters;
- transformed storage address, size, and alignment;
- caller-supplied content generation or an explicit immutability promise;
- lifetime, invalidation, concurrent-read, and storage-limit rules.

Pointer identity alone cannot prove source contents are unchanged. Neither the
current GEMM packed-B format nor a future GEMV transform may be reused unless
its exact layout and orientation contract match. A process-global mutable
transform cache is forbidden.

Several independent x vectors against the same A may form an explicit batch
that can legally lower to GEMM. The compiler should evaluate that semantic
opportunity rather than hiding repeated transformations or a loop of GEVM
calls.

## Parallel tasking

The first parallel strategy should partition N into contiguous,
cache-line-aligned output ranges:

- disjoint y ranges;
- disjoint contiguous A row segments per worker;
- shared read-only x;
- no cross-worker reduction;
- static deterministic assignment.

Each worker revisits its N slice across all K rows. Thread count should be
capped by live output blocks, matrix bytes, measured bandwidth saturation,
requested/available physical cores, SMT and placement policy, and one-node
legality. Small operations remain serial because persistent-worker dispatch
can dominate.

Splitting K would require per-worker partial y storage followed by a
deterministic reduction. That changes workspace, synchronization, numerical
order, and output effects. It is a later explicit candidate, never a hidden
fallback.

The current persistent executor can host N-range tasks, but GEVM needs its own
task planner and cost model. GEMM's MC/NC grid and work-per-task thresholds do
not transfer.

## Topology and NUMA

Initial support should remain single-node. A row-major GEVM streams every
matrix row, so an N partition gives each worker a column slice across the
matrix's page layout. Physical page placement may not align with those slices.

Cross-node support requires:

- explicit matrix-page placement evidence;
- declared first-touch/allocation responsibility;
- node-local worker and output placement;
- a decision between N partitioning, matrix replication, or K partials;
- explicit reduction storage if K is split;
- physical multi-node correctness and performance validation.

Synthetic topology tests establish deterministic planning only, not physical
NUMA performance.

## Private planner inputs

A future deterministic GEVM planner should consume:

- K, N, matrix bytes, vector bytes, and output bytes;
- N-block versus row-stream versus caller-owned transformed mode;
- alignment and N-tail classes;
- expected cache-residency class supplied explicitly, not learned from timing
  history;
- output-block accumulator and ISA costs;
- N task count and cache-line-safe task boundaries;
- measured persistent-dispatch cost;
- measured bandwidth-saturation thread ceiling;
- capability, placement, SMT, and one-node evidence;
- controlled external-provider availability and thread count.

Selection must use fixed candidate order, saturating estimates, explicit
rejection reasons, and no runtime autotuning.

## External baseline

A coherent CBLAS baseline should use row-major SGEMV with transpose semantics:

```text
layout = CblasRowMajor
trans  = CblasTrans
M      = K
N      = N
lda    = N
incX   = 1
incY   = 1
alpha  = 1
beta   = 0
```

The adapter must authenticate the provider, control its thread count, reject
unsupported semantics before output mutation, and report setup versus
execution boundaries fairly. The mapping above is a proposed test target; it
does not register or expose an adapter today.

## Required implementation evidence

Before any automatic or public path, require:

- typed operation kind or orientation, rank-1 values, roles, shapes, effects,
  and deterministic JSON versioning;
- verifier rejection for every unsupported semantic;
- independent double-precision oracle;
- tiny, cache-resident, beyond-LLC, vector-width, cache-line, and page tails;
- alignments 4, 16, 32, and 64 bytes;
- alias/overlap, null, dimension, span-overflow, and unavailable-ISA failures;
- unchanged output on pre-execution failure;
- exact AVX2/AVX-512 symbol inspection with no out-of-bounds tail access;
- Release, Debug, ASan, UBSan, and TSan for parallel state;
- one-shot, repeated, hot, and cold measurements;
- effective matrix bandwidth as well as GFLOP/s;
- equal-thread and equal-placement CBLAS comparison;
- Linux and Windows build/package/consumer validation if public exposure is
  later approved.

## Open decisions before API freeze

- distinct public `gevm` spelling versus a generalized orientation model;
- real rank-1 `vector_view` versus generalized rank-aware view;
- overwrite-only versus alpha/beta;
- zero-size and NaN/Inf behavior;
- deterministic reduction-order guarantee;
- explicit batching and GEMM lowering;
- transformed-matrix ownership and invalidation;
- device-neutral plan/execution records.

Until those are resolved and the full compiler/runtime pipeline exists, GEVM
remains private design evidence only.
