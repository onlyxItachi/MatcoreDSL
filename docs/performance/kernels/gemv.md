# Private GEMV kernel design contract

Status: design-only input from Milestones 6 and 7.

MDSLC does not currently expose, capture, verify, plan, package, or execute a
public GEMV operation. This document defines a candidate private backend
contract for later investigation. It does not add `matcore::mdsl::gemv`, extend
Matcore IR, add a C ABI entry point, register a planner variant, or freeze an
API.

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
A: f32[M,K], rank 2, row-major contiguous, read-only, host
x: f32[K],   rank 1, contiguous,           read-only, host
y: f32[M],   rank 1, contiguous,           write-only, host

y[i] = sum(k = 0..K-1, A[i,k] * x[k])
```

The first private implementation should require:

- positive concrete M and K;
- F32 inputs, accumulation, and output;
- synchronous CPU execution;
- overwrite semantics (`alpha=1`, `beta=0`);
- no overlap between y and either input;
- no hidden allocation, matrix transform, copy, or migration;
- no strided/negative-stride vectors, column-major matrix, batching, transpose,
  mixed dtype, asynchronous execution, or non-host memory;
- explicit failure for every unsupported condition.

A and x are both read-only and may overlap semantically. A particular
optimized body must not assert a stronger no-alias relation unless legality
proves it.

Alpha/beta would change y from write-only to read-write and is not a private
implementation detail. Zero-size behavior, NaN/Inf policy, reduction order,
and rank-1 vector representation also require explicit versioned decisions
before any public exposure.

## Why this is not degenerate packed GEMM

**Derived:** GEMV performs `2*M*K` FLOPs. An optimistic unique-payload lower
bound is:

```text
4 * (M*K + K + M) bytes
```

Its large-problem arithmetic-intensity ceiling approaches 0.5 FLOP/byte.
Large cold GEMV is therefore normally cache- or memory-bandwidth oriented,
unlike a square GEMM whose reuse raises arithmetic intensity with size.

Transiently packing A adds a complete A read, packed write, and packed reread
to an operation that otherwise streams the matrix once. It must not be the
default one-shot strategy. GEMM's MC/NC/KC thresholds, padded four-row
microtiles, and compute-based thread floors are not valid GEMV defaults.

## Candidate direct kernel

Row-major A makes each matrix row and x traversal contiguous. Every y element
is an independent dot-product reduction.

**Proposed AVX2/AVX-512 design:**

1. Process several live A rows together.
2. Load one x vector.
3. Load the corresponding vector from each active row.
4. Update one or more independent accumulators per row.
5. Reduce each live row horizontally.
6. Store exactly the live y elements.

Sharing each x load across rows improves reuse and provides independent
reduction chains. Register tile and K unroll must be ISA-specific:

- AVX2 must balance live row accumulators against its 16 YMM registers;
- AVX-512 may use more rows or multiple accumulators per row, but wider vectors
  are not accepted without exact artifact and physical timing evidence.

K tails require bounded masked or scalar loads. M tails activate only live
rows; padded phantom rows are forbidden because they can multiply work for
small M. Output stores must never cross y.

The initial implementation sequence should be:

1. portable F32 reference with double-precision test oracle;
2. direct compiler-vectorized row dot;
3. exact-symbol AVX2/FMA multi-row dot;
4. physically gated AVX-512F/FMA multi-row dot;
5. parallel contiguous-row execution;
6. authenticated external CBLAS baseline.

These are private family descriptions, not stable IDs.

## Optional transformed-matrix mode

Direct no-pack execution comes first. A transformed-A path is credible only
for repeated use and must be caller-visible.

Any future transformed descriptor must record:

- GEMV orientation and format version;
- source address, dimensions, layout, stride, dtype, and alignment;
- ISA/vector/block parameters;
- transformed storage address, size, and alignment;
- caller-supplied content generation or an explicit immutability promise;
- lifetime, invalidation, concurrent-read, and storage-limit rules.

Pointer identity alone cannot prove the source contents are unchanged.
GEMM's packed-B descriptor cannot be reused because it authenticates a
different KC/NC/NR layout. A process-global mutable transform cache is
forbidden.

Repeated vectors may form an explicit batch that can legally lower to GEMM.
The compiler should evaluate that semantic opportunity instead of hiding a
loop of independent GEMV calls.

## Parallel tasking

Rows are independent. The first parallel strategy should assign contiguous row
ranges:

- disjoint y ranges;
- cache-line-aligned output boundaries where possible;
- shared read-only x;
- full contiguous A row streams;
- no arithmetic reduction;
- static deterministic assignment.

Task granularity must be finer than GEMM's fixed 128-row macro tile. Small
calls remain serial because current persistent-context submission alone costs
microseconds on the audited host.

Thread count should be capped by:

- live row chunks;
- matrix bytes;
- measured bandwidth saturation;
- requested/available physical cores;
- SMT and placement policy;
- one-node legality;
- explicit workspace, if any.

The current operation-independent persistent executor can host these tasks,
but GEMV needs its own plan and cost model. It must not inherit GEMM's
work-per-thread threshold.

## Topology and NUMA

Initial support should remain single-node. On multi-node systems, contiguous
row decomposition can match caller first-touch placement, while x may be
replicated or remotely read. Neither behavior may be implicit.

Cross-node support requires:

- explicit matrix-page placement evidence;
- declared first-touch/allocation responsibility;
- x replication or access policy;
- node-local worker selection;
- physical multi-node correctness and performance validation.

Synthetic topology tests establish deterministic planning only, not physical
NUMA performance.

## Private planner inputs

A future deterministic GEMV planner should consume:

- M, K, matrix bytes, vector bytes, and output bytes;
- direct versus caller-owned transformed mode;
- alignment and tail classes;
- expected cache-residency class supplied as an explicit mode, not inferred
  from timing history;
- horizontal-reduction and ISA costs;
- contiguous row-task count;
- measured persistent-dispatch cost;
- measured bandwidth-saturation thread ceiling;
- capability, placement, SMT, and one-node evidence;
- controlled external-provider availability and thread count.

Selection must use fixed candidate order, saturating estimates, explicit
rejection reasons, and no runtime autotuning.

## Required implementation evidence

Before any automatic or public path, require:

- typed operation kind, rank-1 values, roles, shapes, effects, and JSON
  versioning;
- verifier rejection for every unsupported semantic;
- independent double-precision oracle;
- tiny, cache-resident, beyond-LLC, vector-width, cache-line, and page tails;
- alignments 4, 16, 32, and 64 bytes;
- alias/overlap, null, dimension, span-overflow, and unavailable-ISA failures;
- unchanged output on pre-execution failure;
- exact AVX2/AVX-512 symbol inspection with no out-of-bounds tail load;
- Release, Debug, ASan, UBSan, and TSan for parallel state;
- one-shot, repeated, hot, and cold measurements;
- effective matrix bandwidth as well as GFLOP/s;
- equal-thread and equal-placement CBLAS comparison;
- Linux and Windows build/package/consumer validation if public exposure is
  later approved.

## Open decisions before API freeze

- real rank-1 `vector_view` versus generalized rank-aware view;
- public `gemv` declaration versus a more general orientation model;
- overwrite-only versus alpha/beta;
- zero-size and NaN/Inf behavior;
- deterministic reduction-order guarantee;
- explicit batching and GEMM lowering;
- transformed-matrix ownership and invalidation;
- device-neutral plan/execution records.

Until those are resolved and the full compiler/runtime pipeline exists, GEMV
remains private design evidence only.
