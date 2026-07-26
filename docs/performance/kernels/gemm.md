# Private CPU GEMM kernel contract

Status: Milestone 7 internal backend contract at
`e49260e68f7e43124591e6515bfeb1fe84c3ea74`.

This document describes the implemented and reviewed native CPU GEMM layers.
It is not a public API or ABI freeze, a promise that a private microkernel name
will remain stable, or a native-BLAS parity report. Final quantitative parity
results belong in `../cpu/native-blas-parity-v1.md` and are deliberately not
invented here.

Primary evidence:

- [HPC kernel engineering handbook](../HPC_KERNEL_ENGINEERING_HANDBOOK.md)
- [GEMM data-movement audit](../audits/gemm-data-movement.md)
- [GEMM microkernel audit](../audits/gemm-microkernel-analysis.md)
- [GEMM parallel-runtime audit](../audits/gemm-parallel-runtime.md)
- [AVX2 full-tile review](../../mdslc/agent-reports/m7-avx2-kernel.md)
- [AVX-512 4x32 review](../../mdslc/agent-reports/m7-avx512-kernel.md)
- [parallel task review](../../mdslc/agent-reports/m7-parallel-runtime.md)
- [cooperative B-packing review](../../mdslc/agent-reports/m7-parallel-performance.md)

## Semantic contract

The executable CPU operation is:

```text
A: f32[M,K], rank 2, row-major contiguous, read-only, host
B: f32[K,N], rank 2, row-major contiguous, read-only, host
C: f32[M,N], rank 2, row-major contiguous, write-only, host

C[i,j] = sum(k = 0..K-1, A[i,k] * B[k,j])
```

The current contract requires:

- positive concrete `M`, `N`, and `K`;
- F32 input, accumulation, and output;
- synchronous CPU execution;
- explicit `out(C)`;
- no output/input overlap, including partial overlap;
- no workspace/input/output overlap;
- no hidden allocation, input copy, output copy, or host/device migration;
- no alpha/beta scaling and no read-before-write requirement for C;
- explicit workspace and transformed-B storage owned by the caller;
- fail-closed variant and target selection.

Changing output initialization, alpha/beta, zero-size behavior, dtype,
strides, layout, memory space, or synchronization is a semantic change. It
must cross a versioned IR/runtime boundary rather than being inferred inside a
microkernel.

## Layer boundaries

The private execution pipeline is:

```text
typed Matcore IR v1 GEMM
  -> verified concrete CpuGemmProblemV1
  -> capability, topology, placement, and implementation resources
  -> deterministic planner-v3 candidate decisions
  -> workspace/prepacked-B legality
  -> serial or persistent-context executor
  -> packing and output-task decomposition
  -> checked edge or prevalidated full-tile microkernel
  -> explicit execution report
```

Each layer owns a different proof:

| Layer | Required proof |
|---|---|
| IR verifier | roles, ranks, shapes, dtype, layout, effects, policy |
| problem validation | positive concrete dimensions and overflow-safe spans |
| capability model | hardware, OS state, compiler, implementation, runtime validation |
| planner | candidate legality, resources, actual threads, stable explanation |
| runtime preflight | pointers, alignment, overlap, workspace, context, provider nesting |
| packing | exact format, padded extent, destination bounds, provenance |
| tasking | disjoint C rectangles, bounded packed panels, no K split |
| microkernel caller | full-versus-edge extent and ISA entry legality |
| execution report | implementation actually executed and resources actually used |

No lower layer may repair a missing upper-layer proof by silently changing the
problem.

## Registered implementation families

Planner-v3 currently evaluates these fixed diagnostic identities:

| Stable diagnostic ID | Execution family |
|---|---|
| `cpu.reference.f32.v1` | portable correctness/reference loop |
| `cpu.tiled.f32.v1` | portable 32x32x64 tiled loop |
| `cpu.compiler-vectorized.avx2-fma.f32.v1` | compiler-vectorized AVX2/FMA candidate |
| `cpu.external.openblas.f32.v1` | optional authenticated CBLAS SGEMM |
| `cpu.native-packed.avx2-fma.f32.v1` | serial packed AVX2/FMA |
| `cpu.native-packed.avx512-fma.f32.v1` | serial packed AVX-512F/FMA |
| `cpu.native-parallel.avx2-fma.f32.v1` | persistent-context packed AVX2/FMA |
| `cpu.native-parallel.avx512-fma.f32.v1` | persistent-context packed AVX-512F/FMA |

These IDs identify planner candidates, not individual microkernel symbols or
blocking profiles. A private tile body may change after correctness, artifact,
holdout, and planner evidence without forcing that body into the installed
headers. Whether forced implementation IDs should remain public enum values is
an open pre-freeze decision.

## Packed format and workspace

The v1 packed geometry is:

| Parameter | Value |
|---|---:|
| MR | 4 rows |
| NR | 16 columns |
| MC | 128 rows |
| NC | 256 columns |
| KC | 256 depth |
| workspace alignment | 64 bytes |

A is packed as `[4-row micro-panel][k][row lane]`. Missing M lanes are
zero-filled. B is packed as `[16-column micro-panel][k][column lane]`.
Missing N lanes are zero-filled. K is not padded.

The format contract records logical and padded extents, `KC/NC/NR`, storage
size, alignment, source and packed addresses, and provenance. Arithmetic for
every offset and byte extent must be checked before writing. The format is an
internal versioned asset; it is not a general transformed-matrix API.

Transient serial execution uses one MCxKC A region and one KCxNC B region.
Caller-owned prepacked-B execution retains only transient A workspace.
Parallel execution uses one shared read-only full-B image followed by
cache-line-separated per-worker A regions. Insufficient or misaligned storage
must fail before output mutation or worker submission.

The packed-B identity proves the descriptor and storage relationship, not that
source contents stayed immutable. The caller must keep the source and packed
storage alive and unmodified for the documented lifetime. A process-global
implicit packed-weight cache remains forbidden.

## Serial microkernel contract

### AVX2/FMA

The checked 4x16 entry accepts complete or partial M/N tiles. Partial output
uses a guarded 4x16 edge buffer and copies back only live lanes. The exact
Release body contains eight YMM accumulator chains and packed FMA
instructions.

A private prevalidated 4x16 symbol also exists for shared internal use. An
independent ABBA review rejected routing the serial executor through it as a
performance promotion, so serial AVX2 retains the checked entry. The private
symbol remains useful to the parallel executor after it proves a complete
tile. Both symbols are artifact-tested and hidden from the runtime dynamic
export surface.

### AVX-512F/FMA

Complete 4x32 tiles use a private body with eight ZMM accumulators and two
adjacent v1 B micro-panels. It reuses the v1 packed format without repacking.
Remaining 4x16 tiles and every M/N edge use the checked 4x16 entry.

The generic runtime is not globally compiled for AVX-512. Entry requires
x86-64, hardware AVX-512F/FMA, OS-enabled architectural state, compiler
support, built implementation, and physical runtime validation. Unsupported
hosts must not call the function, even when an object containing it exists.

### Common full-tile preconditions

An unchecked full-tile callback may execute only when the caller has proved:

- non-null packed A, packed B, and output;
- nonzero logical depth;
- complete register-tile rows and columns;
- packed-panel intervals inside authenticated storage;
- output rows inside C and a legal output stride;
- correct first-K-panel overwrite versus later-panel accumulation;
- non-overlap and usable ISA state.

Edge paths must never speculate beyond live source or output memory. Padded
packed lanes may be read only from storage whose padded extent was allocated
and initialized.

## Parallel execution contract

The persistent executor reuses workers and never creates a pool per GEMM. K is
not split, so no cross-worker reduction or change in per-output accumulation
order is introduced.

The deterministic task planner begins with M bands and may expose NC-sized N
panels. Row boundaries satisfy MR and output cache-line cycles. Column
partitioning requires at least 64-byte declared alignment, a cache-line-safe
N extent, more than one NC panel, and measured work floors:

- column-only: at least `2^23` FLOPs per task;
- true two-dimensional decomposition: at least `2^25` FLOPs per task.

Ties prefer more row tasks, limiting duplicated A packing. Row and column
ranges use quotient/remainder balancing. Every task owns one disjoint C
rectangle, reads immutable packed B, and uses its worker's isolated A
workspace. Planner and runtime call the same pure task-plan function so actual
threads and diagnostics cannot drift from execution.

## Cooperative B preparation

Ordinary parallel execution prepares B serially before dispatch. The accepted
Milestone 7 path cooperatively prepares disjoint final NC panels only in a
measured short-wide envelope:

- more than one worker;
- `M <= 64`, `N >= 4096`, and `K >= 1024`;
- at least two NC panels and at least 4 MiB packed B;
- additionally `K >= 4096`, or `M <= 32 && N >= 8192`.

Workers write disjoint final packed intervals. A release/acquire phase barrier
publishes the complete read-only image before compute. An abort state prevents
future packing-invariant failure from deadlocking peers. Packing and compute
share one context submission and no extra pool.

This rule is host-bounded measured calibration, not a universal constant.
Boundary shapes rejected by the calibration retain serial preparation. The
path is within-call cooperative preparation; it is not a cross-call
parallel-prepacked-B API.

## Planner and selection requirements

For every candidate, the planner must retain:

- legal or rejected state and actionable reason;
- required hardware, OS, compiler, implementation, and runtime evidence;
- requested ceiling and actual thread count;
- row tasks, column tasks, total tasks, and capacity limiting;
- shared and per-worker workspace;
- alignment and NUMA/placement facts;
- estimated cost, priority, selected state, and selection reason.

Selection remains deterministic. Costs use saturating arithmetic and fixed
tie-breaking. Runtime autotuning and implicit history-dependent dispatch are
outside this milestone. OpenBLAS remains selectable when it is faster; an
automatic plan that chooses OpenBLAS is not native parity.

Planner changes require frozen calibration and holdout evidence with every
legal candidate forced. A threshold that fixes one shape but creates
catastrophic regret elsewhere is rejected.

## Correctness and artifact gates

Every native path must cover:

- tiny, square, tall, wide, K-tail, M/N-tail, and randomized dimensions;
- 4-, 16-, 32-, and 64-byte declared/actual alignment where legal;
- transient and authenticated prepacked-B execution;
- repeated and concurrent independent contexts;
- null, zero, negative, overflow, malformed metadata, and unavailable ISA;
- exact and partial aliases among A, B, C, workspace, and packed storage;
- insufficient and misaligned workspace with unchanged output;
- context capacity, shutdown, nested-provider, and forced-variant failure;
- output guards and independent double-precision oracle;
- ASan, UBSan, and TSan for shared state.

Release artifact tests must isolate the production symbols, require the
claimed YMM/ZMM packed-FMA instructions, reject scalarization, and detect
spills in optimized full-tile bodies. Debug may be correctness-validated
without making a performance claim.

## Measurement boundary

Parity evidence must compare complete equivalent calls:

- same semantics, shape, dtype, layout, alignment, and correctness oracle;
- same actual thread count and placement class;
- same cache, allocation, packing, and repeated-execution mode;
- controlled OpenBLAS local threads with no nested native pool;
- stable forward/reverse process order, warmup, timer floor, and raw samples;
- exact clean source, runner, executable, plan, and raw-file identities.

Compute-only timing is a kernel diagnostic, not a complete CBLAS comparison.
Prepacked-B evidence must include one authenticated preparation and state the
amortization count. Raw data remains outside Git. No claim in this contract
substitutes for the final sanitized Milestone 7 report.

## Known boundaries

- One static v1 packed format and block tuple still serves all native shapes.
- AVX2 serial full-tile rerouting was measured and rejected.
- Cooperative B preparation is deliberately narrow.
- Parallel execution cannot consume caller-owned prepacked B across calls.
- Tail kernels still compute padded M/N lanes in the checked path.
- Hardware counters and workload frequency were unavailable in Milestone 6.
- Real multi-node NUMA performance is unvalidated.
- Public transformed-operand lifetime, stable variant-selection, and detailed
  topology contracts remain open for the later freeze milestone.
