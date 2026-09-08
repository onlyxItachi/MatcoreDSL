# Cache measurement review and next numerical boundary

## Decision and scope

**Keep cache tiling as a connected research branch, not a product schedule or
default.** Its correctness evidence survives, but the measured `[4,64,32]` choice
loses to row-contiguous M/K/N in every measured workload. Promote useful independent
boundary tests separately. A passing optional implementation is not by itself a
reason to maintain another product selector.

This is a read-only benchmark/compiler review plus this document. No harness,
runtime, issuer, frontend or authority was changed, and no new timings were run.
The reviewed implementation is `9dec69773bf9f7c12b27b27bf360952e173e0af4`;
independent source-boundary fixtures are
`902a4ef3bba940460d75cc474d879d56ffa42355`. The original research branch started
this review clean at `6ce4b3f6035f1a4ae8f5395753f9e0202eabdffc`.

## Raw measurement audit

**OBSERVED:** raw evidence is
`/home/hamza-usta/MatcoreDSL-wt-mlir-hpc-execution-evidence-v1/benchmark_reports/hpc-cache-comparison-v1/evidence.json`,
SHA256 `1e5d000f3731e91d86c7456336e1d7aa930859e057223be1ae4de268e62d1bef`,
started `2026-09-08T23:21:30.902695+00:00`. Harness/review checkout is clean
`3650fd4c31db6c8bf7d5eb2d4432799ca425e7fa`. The raw record independently identifies
the older scalar build, row head `58193878394c774c3365dbf746d1622e951b0813`, and
cache head `9dec697`; its top-level `driver_build_commit` is the scalar baseline,
not the cache source checkpoint.

Every recorded source, artifact and lane-executable hash was recomputed locally:
no mismatch. Relevant production object hashes are:

| Realization | SHA256 |
| --- | --- |
| Scalar | `8a5334297a2b680f2df2cea28945ce773b4b250abfa567b19f313c446b50b847` |
| Row | `26b5df728c64c3ae746da30373e821cba19c009ba6474f81ef056293282abc1e` |
| Cache | `02a8657e898705e156c593a9771e4c69886390895027a256062944798f596fd8` |

The cache driver is `18af6cab78d570980211ebcf671e146785a941625b845f58efe6120283c728ad`;
its candidate DSO is `dd45eb55a9ea2476086f0e7bb9d4b1b7216016165cad762e37f64784c094e50e`.
All three Runtime DSOs have the same recorded bytes. These are the actual connected
O2 artifacts, not the earlier raw-MLIR O3 research objects.

The 11 lanes contain **256 correctness cases**: ten strict lanes times 24 and one
reassociate/forced-reference lane with 16 ordinary cases. All 11 corrupt-output
controls exit 4 with the expected oracle diagnostic. The record also contains
24 bad-shape and 24 read-only-output checks, plus a 24-case strict/forced-reference
incompatibility run. Reassociation intentionally skips strict-sensitive bitwise
oracles; its 16 cases must not be reported as full numerical-family validation.

There are 198 unique `(lane, case, round)` timing batches, 594 positive batch-mean
samples, and nine samples per lane/shape. Independently sorting the raw samples
reproduced every recorded median exactly. The query was the direct JSON grouping
of `.timing[].samples[].ns_per_call` by lane/case, selecting sorted element 4;
there was no aggregation across unlike workloads or execution layers.

Primitive medians in microseconds, with source end-to-end ratios kept separate:

| M/N/K | Scalar | Row | Cache | Cache/row primitive | Cache/row source |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16/16/16 | 0.996 | 0.460 | 0.506 | 1.098 | 1.067 |
| 128/128/128 | 964.218 | 126.375 | 149.997 | 1.187 | 1.185 |
| 512/512/512 | 127482.433 | 7960.645 | 10003.580 | 1.257 | 1.236 |
| 128/256/64 | 877.659 | 127.406 | 151.590 | 1.190 | 1.184 |
| 16/512/256 | 1887.084 | 123.560 | 153.244 | 1.240 | 1.239 |
| 65/67/63 | 94.638 | 24.949 | 30.329 | 1.216 | 1.210 |

**Share with caveats:** the experiment supports rejecting this cache choice for
these workloads, not a theorem against tiling. CPU 2 was pinned on Ryzen AI 9 HX
370; lane order was shuffled for three rounds; buffers were warm; compiler activity
was checked and agents held a quiet window. Desktop activity, SMT sibling 14,
thermal state and frequency were not isolated. Samples are batch means, not nine
independent machines or process-level confidence intervals. The source lane includes
allocation/snapshot/publication overhead. Public Result does not expose actual
candidate identity; forced selection, reviewed registry and artifact binding supply
the bounded identification. No provider lane was requested: this is neither provider
performance evidence nor an OpenBLAS-OFF build claim. The cache Release build was ON.
No cache-miss or register-pressure counters identify the cause of its losses.

## Existing numerical permission, not new authority

The source already has authenticated `Numerics::reassociate_f32`:
[region.h](../../../compiler/include/matcore/region.h),
[MatcoreClosedRegion.cpp](../../../compiler/lib/mlir/MatcoreClosedRegion.cpp),
[runtime admission](../../../compiler/lib/runtime/closed_host_v1.cpp), and the
[candidate tests](../../../compiler/tests/closed_candidates/candidate_test.cpp)
agree on per-GEMM f32 reduction reordering and multiply/add contraction. The source
profile does **not** authorize ignoring signed zero, assuming finite inputs,
flushing subnormals, reducing precision, removing the f32 result boundary, or
reassociating across GEMMs. The runtime independently establishes nearest-even,
gradual-underflow, masked exceptions and complete caller-FP-state restoration.

**PROPOSED:** a generated relaxed candidate could use explicit, checked f32
FMA/reduction topology through upstream MLIR, with profile-specific issuance and
registry legality. Strict calls must reject that forced candidate before allocation
or invocation, including empty cases; existing publication/failure prefixes remain.
The strict candidate must keep its identity and behavior. Real FMA instructions also
require an isolated target artifact and validated hardware/OS capability admission;
an x86-64 baseline compile or successful intrinsic lowering is not hardware support.

Do not equate this permission with LLVM `fast`. LLVM distinguishes contraction,
reassociation and finite/zero assumptions; its `reassoc` is phrased as algebraically
equivalent rewriting, not only permutations of the source's product leaves.
A narrowly constructed reduction is easier to audit than granting an unexamined
bundle of flags to the entire module. This is a restriction on the proposed proof,
not a claim that a particular LLVM pass has already miscompiled it.
[Pinned LLVM 21.1.8 language reference](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/docs/LangRef.rst#fast-math-flags).

**Correction retained:** `fma(denorm_min, -0.5, +0)` can produce negative zero,
where appending a positive-zero accumulation changes that chosen realization.
This is **not** by itself a source-profile violation: the permitted unfused K1
expression already produces positive zero. Do not mistake a difference from one
allowed algorithm for a result outside the allowed family. Conversely, introducing
an extra product `0*Inf` or `0*NaN` through incomplete tail masking can contaminate
an otherwise finite reduction. Bilaterally zero-filled padding requires its own
family-level argument; the invalid signed-zero argument does not establish it.

## Prefer preserving a known private alias fact first

**OBSERVED:** current `Session::gemm` retains its immutable inputs and allocates a
fresh private output before issuing a value. A and B can be the same owning Value;
the new C allocation cannot overlap either live input under the trusted allocator
contract. Source Storage descriptors remain MAYalias. The row LLVM leaf carries
none of this data-pointer alias information.

The current upstream `memref.distinct_objects` spelling is absent from the exact
[21.1.8 MemRef operation set](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td)
and present in
[22.1.8](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td#L172-L200).
It is an assumption, not an overlap guard. A three-way distinctness assumption would
overstate this contract. Its availability is not a reason to migrate the product pin.

LLVM 21 already supplies argument `noalias` and alias-scope metadata; MLIR's LLVM
dialect exposes those mechanisms. A minimal experiment can encode noalias on the
exact flattened C **aligned data** pointer only, after checking the owned leaf ABI.
In the current fixed rank-2 three-memref ABI that is argument 15, not allocated
pointer 14 or the C-interface descriptor address. Root's separately reported 21
prototype found that putting `llvm.noalias` on the high-level memref argument only
annotated the wrapper descriptor pointer, not the leaf data pointer. This review did
not rerun that prototype.
[LLVM noalias contract](https://releases.llvm.org/21.1.0/docs/LangRef.html#parameter-attributes),
[MLIR 21 attributes](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.td),
[Func-to-LLVM descriptor wrapper](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp).

This would preserve a proved ownership fact, not implement a private optimizer.
Leave A/B unqualified and add no unproved nonnull, extent, alignment or independent
ownership claims. Its benefit remains **UNRESOLVED** pending actual execution and
measurement; a smaller object alone is not a speedup. Prefer this bounded inquiry
over further blind tile-size tweaks. If exhausted, the existing reassociation
permission makes an independently issued FMA candidate the next higher-leverage leaf
inquiry, not an excuse to expand all generated code's numerical authority.

## Three falsification families for the next inquiry

1. **Private alias precision:** A==B, partial overlapping input ranges at the direct
   leaf, and a Session result handle identical to both operands must remain correct;
   inspect that no C fact leaked to A/B or a descriptor address. Retain source
   publication followed by an aliased late read and earlier-output reuse. Compare
   original and annotated artifacts, with real generated-code ASan controls.
2. **Numerical-family legality:** a nonzero fused-versus-zero strict discriminator,
   exhaustive small-K product/permutation/tree/FMA membership, NaN/Inf/subnormal/
   signed-zero cases and masked tails. Preserve two dependent GEMMs with different
   profiles and their intermediate f32 boundary. A tolerance-only double oracle
   cannot replace these checks.
3. **Authority and effects:** forced relaxed candidate under strict source, including
   zero extents; malformed extents before leaf entry; late failure after a prior
   publication; full caller FP restoration and failed candidate private-write
   isolation. No flag, manifest or benchmark result may bypass these guards.
