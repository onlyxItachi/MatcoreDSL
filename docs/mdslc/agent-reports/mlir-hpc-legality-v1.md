# Independent MLIR/HPC legality review v1

Base: clean canonical `0e8c040809ae1fe6b0f1c01e41fc509b07212be7`.
Scope: independent architecture challenge and reproducible adversarial fixtures.
This lane changes this report, the independent
[`compiler/experiments/hpc_legality_v1`](../../../compiler/experiments/hpc_legality_v1/README.md)
numerical probe, and the newly promoted `schedule_*` source fixtures/runner in
`compiler/tests/generated_cpu/`. The integration owner explicitly requested
that promotion; this lane does not edit existing test registration or production.
Production integration, benchmarks and hosted gates belong to the owning lane.

## Architectural recommendation

**Use upstream schedule machinery, retaining the existing Matcore semantic and
candidate contracts.** The claim "MLIR needs no extra abstraction" is defensible
only if it means no second private implementation of tiling, vectorization,
bufferization or machine lowering. It cannot mean that successful generic passes
establish Matcore source provenance, ordered failure semantics, numerical
permissions or concrete storage facts.

The strongest bounded first integration is one compiler-issued realization of
the existing strict GEMM primitive:

1. Keep source admission, sealed frontend-neutral Program, exact untransformed
   source/MLIR pairing, static orchestration and checked host thunk unchanged.
2. Start from the issuer's already-verified positive-zero fill plus strict
   Linalg GEMM. Apply a closed, pinned upstream schedule to a derived payload.
3. Tile/vectorize independent M/N outputs while retaining increasing scalar K
   and one separate f32 multiply/add accumulator chain for each output lane.
   A K tile is legal only if its continuation preserves that exact chain; split
   partial sums are not a substitute. Avoid claiming K-vector reduction legality.
4. Check schedule-specific order, lane/index maps, tails, exact destination and
   resource postconditions before ordinary upstream LLVM lowering. Keep the
   generated symbol ABI and private locally-bound candidate DSO ownership.
5. Admit it through the existing compile-trusted generated candidate route;
   retain all runtime shape/capacity/FP/candidate guards and private-result
   issuance before publication. Identify the actual schedule in its manifest.

This is **transformed primitive execution through authenticated source**, not
transformed whole-region execution, fusion or two-GEMM storage reuse. The latter
frontier remains separate. A new generic region execution engine, second tensor
class, private schedule IR, bespoke bufferizer, optimizer JSON graph or plugin
authority mechanism is not necessary to establish this first result.

## Irreducible contracts already present

- [Candidate issuer](../../../compiler/lib/mlir/MatcoreCpuGemmCandidate.cpp)
  constructs its own primitive and derives canonical Linalg after exact pairing.
  It accepts no imported candidate graph. Its current scalar LLVM opcode counts
  are tailored to its current pipeline, not a sufficient vector/order verifier.
- [Candidate header](../../../compiler/lib/mlir/MatcoreCpuGemmCandidate.h)
  requires nonnegative representable extents, canonical host f32 descriptors,
  caller-provided capacity/lifetime/race freedom, compatible FP controls and a
  destination isolated from both inputs. It does **not** require inputs to be
  disjoint from each other. It promises no allocation or publication in the leaf.
- [Resource contract](../FOUNDATION_RESOURCE_DECISION_V1.md) distinguishes
  immutable contents from their realization and permits all-MAY-alias external
  descriptors. The selected host adapter has stronger failed-publication
  guarantees than the generic witness. It preserves earlier effects on failure.
- [Runtime adapter](../../../compiler/lib/runtime/closed_host_v1.cpp) retains
  legality before candidate selection, isolated private output, actual-path
  reporting, FP restoration and zero-output/zero-K realizations. These remain
  the boundary; a Transform script cannot discharge them.

The new derivation should be private compiler logic over existing MLIR, not a
claim that serializing a digest/certificate creates execution authority. Replay
and hashes bind exact inputs/stages; they do not replace semantic verification.

## Pinned upstream evidence and concrete traps

All upstream inspection here is at `llvmorg-21.1.8`, not moving trunk.

- [Transform execution model](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/mlir/docs/Dialects/Transform.md):
  matching can succeed with empty handles; containers can recover from silenceable
  failures. Therefore pass success alone does not show that the requested tile
  or vectorization happened. Assert target cardinality and achieved payload form.
- [Vector contraction lowering](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/mlir/lib/Dialect/Vector/Transforms/LowerVectorContract.cpp):
  `createContractArithOp` constructs `vector::FMAOp` for floating vector ADD with
  an accumulator. Absence of fast-math flags on the starting Linalg body does not
  make every contraction lowering strict. Avoid introducing fused semantics;
  indiscriminately splitting an already-semantic FMA is also not legal.
- [One-Shot Bufferization](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/mlir/docs/Bufferization.md):
  reuse follows tensor use/alias analysis, and `restrict` at tensor/buffer
  boundaries asserts a real alias condition. Distinct descriptors cannot justify
  it. Input/input identity remains possible even for the private leaf.

The executable numerical counterexamples establish:

| Incorrect transformation | Strict result | Incorrect result |
| --- | --- | --- |
| Fuse the second update for terms `-1`, `(1+2^-23)*(1-2^-23)` | `+0` | `-2^-46` |
| Split K terms `[2^25,1,-2^25,1]` into two independent pair sums | `1` | `+0` |
| Replace `(2^100 * 2^100) * 2^-100` across two f32 GEMMs with `2^100 * (2^100 * 2^-100)` | `+Inf` | `2^100` |

These are semantic counterexamples, not timing measurements or claims that the
proposed correct upstream schedule performs these transformations.

## Required adversarial postconditions

| Boundary | Positive obligation | Deliberate mutation that must not pass |
| --- | --- | --- |
| Closed issuance | Exact built-in source contract and fixed schedule inputs, no caller-supplied authority | Alter the primitive or substitute imported transformed IR with copied metadata |
| Schedule applied | Expected matched GEMM, achieved tile/vector form and no ignored failures | Empty match, wrong target, omitted vectorization, silently skipped transform |
| K arithmetic order | K begins at zero, increments by one, covers all K exactly, same f32 accumulator continues | Step two, reversed K, extra/missing iteration, two independent accumulators, reset per K tile |
| SIMD lane meaning | Each lane represents a distinct output `(i,j)`; reads use A `(i,k)` and B `(k,j)` | Swap operands/maps, broadcast wrong K/J, shuffle result lanes, use wrong row stride |
| Bounds/tails | Every actual input read and output write is in its guarded domain | All-true tail mask, false `in_bounds`, ceil-trip full load, wrong final lane/store mask |
| FP semantics | Positive-zero overwrite, separate f32 multiply/add, no unauthorized flags/reduction/FMA | Negative-zero fill, FMA, widened accumulator, `nnan`/`ninf`/`nsz`/`reassoc`/`contract`, tree reduction |
| Resource meaning | Only exact destination is written; inputs unchanged; no tensor allocation/copy/global/provider effects | Write an input, return foreign destination, append allocation/copy/unknown call, infer mutual input noalias |
| ABI/target/ownership | Exact private wrapper/leaf boundary, justified LLVM intrinsic use, admitted target features | Extra export/declaration, ABI drift, target-wide unsupported ISA, interposable helper ownership |
| Instrumentation | Generated memory operations are actually instrumented where claimed | A deliberate active-lane OOB must trigger ASan; successful sanitizer linkage alone is insufficient |

The postcondition must establish dataflow and control relations, not merely count
`fmul`/`fadd` instructions or inspect an operation's mnemonic. Machine disassembly
should separately establish actual SIMD instruction generation before claiming
that the issued object uses SIMD; MLIR vector syntax alone is not that evidence.
Baseline x64 SIMD does not imply AVX2/FMA availability or a performance advantage.

Tail masks can make inactive computed lanes unobservable without requiring their
arithmetic to disappear. Bounds safety, output isolation and absence of effect
leakage are the obligations; no blanket requirement to suppress harmless masked
arithmetic is needed under the existing masked-exception/restored-FP contract.

For the unchanged two-GEMM orchestration, retain these end-to-end falsifiers:

- Save `old = read(C)`, publish a new result into C, later republish old. Borrowing
  or reuse must not change old's immutable contents while it remains live.
- Read D after publishing C when their ranges partially overlap. Moving the D
  snapshot to region entry returns the wrong data.
- Fail the late D capacity check or the second publication's access check.
  The first publication and observation survive; later effects do not happen,
  and the failed publication's destination remains unchanged.
- Feed one immutable Value as both GEMM inputs. Destination isolation does not
  authorize adding a mutually-disjoint-input precondition.

## Observed independent baseline evidence

Small probes only; no production edits, rebuild or benchmark was performed here.
The first numerical-probe compile identified a missing `<initializer_list>`
include in the newly authored fixture. Adding that include fixed the fixture;
no production assertion or compiler contract changed.

The `-O2 -ffp-contract=off` Clang 21 numerical probe passed **8/8 checks**.
Both real-source fixtures were compiled using the existing scalar-generated
driver at unchanged implementation head
`b0e7ac9ac67c3c314a44e5db108be390485404eb`:

| Existing driver | SHA-256 | `source_contract` | `same_value_contract` |
| --- | --- | --- | --- |
| Release | `eee73557f8769569f45b51e95a269237d2863fff5fb3b5146a69daa57fdc69e6` | 4,140 PASS | 1,503 PASS |
| ASan+UBSan | `7e3c8091f9d6423e0328ceefa9ea3562ea734e907e148829edd564b55c02b214` | 4,140 PASS | 1,503 PASS |

An earlier smaller 2,024-check version of `source_contract` passed both existing
drivers. It was then strengthened to cover full 5x33 numerical tiles and tails;
its earlier result must not be substituted for the strengthened fixture.

These baseline runs demonstrate the independent fixtures execute through current
source admission and the real generated candidate. They do not validate a new
candidate or its stage rejection controls. The sources were subsequently moved,
without content changes, into durable test paths at the integration owner's
request. Their new CMake runner is separate from those direct baseline runs.
The runner rejects both a driver that exits nonzero (`/bin/false`) and one that
exits zero without creating an executable (`/bin/true`). After the owning lane's
benchmark quiet window ended, both positive runner invocations passed against
the actual row-selected Release driver (4,140 and 1,503 checks).

## First live production delta review and transformed-source result

The first owning-lane delta is narrower than general tiled/vectorized Linalg:
it uses upstream Transform `generalize` and `interchange [0,2,1]`, producing MKN
row-contiguous scalar Linalg traversal; LLVM may then vectorize independent N
updates. **Do not call it MLIR tiling or MLIR Vector lowering.** It is an actual
upstream scheduling transformation, not a hand-written substitute kernel.

Independent read-only review found no semantic or authority blocker in that
initial uncommitted delta. Its postcondition checks exact MKN indexing maps,
`[parallel,reduction,parallel]` iterators, unchanged f32 accumulator body and the
same private destination. It derives on a clone, verifies the original witness
unchanged, rejects unknown schedule enums and keeps the issuer closed. Inspection
of the issued LLVM confirms scalar K starts at zero, increments by one, covers
the full bound and uses separate f32 multiply/add. The fixed pinned loop lowering
is part of this argument; a reduction iterator label alone is not an order proof.

Both independent source fixtures passed the row-selected Release driver:

- Driver SHA-256: `f183efd71903d7a48b2c98c72bb7e722707db21a17cdea7ce3d2acdfe2518832`.
- Issued LLVM SHA-256: `8ef2b5b27e7f5fcd5fc58f6269927d341a8c31173a904c00263a1d2a1aa3e6b7`.
- Actual object SHA-256: `26b5df728c64c3ae746da30373e821cba19c009ba6474f81ef056293282abc1e`.
- `source_contract`: **4,140 PASS**; `same_value_contract`: **1,503 PASS**.

These are exact local artifact identities from an uncommitted production delta,
not an immutable implementation checkpoint or full integration acceptance.
The independent review requested durable row-specific iterator and accumulator/
add-fastmath mutations, plus a separate achieved-object SIMD assertion. These
requests and instrumented source execution were subsequently closed below; the
uncommitted-artifact chronology above is retained. A Release driver with an ASan
host option is not a substitute for an instrumented implementation.

## Generated SIMD-specific ASan negative control

At the integration owner's request, this lane also owns the bounded extension
of `compiler/tests/generated_cpu/execution_test.cpp` and `expect_asan.cmake`.
The existing N=1 `--oob` control remains unchanged. The new `--oob-simd` uses
M=1, K=1, N=16, one actual B element and a sufficiently large isolated C
allocation. It checks disjoint integer address ranges before calling the leaf,
so output/input alias versioning does not silently select the scalar path.
Only B's intentionally false capacity is outside the private leaf contract;
the caller does not itself perform an out-of-bounds load.

Linked to the actual row-selected `strict-asan.o`, the new control reports
`AddressSanitizer: heap-buffer-overflow`, **`READ of size 16`**, and generated
`_mlir_ciface___matcore_strict_gemm_f32_v1` as frame #0. Both the SIMD-specific
CMake classifier and the original scalar classifier pass their expected-failure
checks. Crucially, linking the same new source to the unchanged scalar candidate
produces only `READ of size 4`: the SIMD classifier **rejects** that result,
despite its otherwise genuine ASan error. Scalar instrumentation cannot stand
in for the new SIMD claim.

Register the SIMD mode only for the row-contiguous selected schedule; default
scalar builds retain their original control. The classifier requires a wide
read and generated top frame, not merely nonzero exit or an ASan library symbol.
This intentionally invalid leaf-only invocation does not weaken the public
adapter's capacity checks or grant source programs permission to bypass them.

## Committed implementation review and instrumented source closure

Read-only production review at owning commit
`9769dbafc18d4b8efd05c2e3519543eb57531604` confirms its implementation files match
the initial reviewed artifacts' source, with the additional requested regression
tests and object assertion. Their SHA-256 identities are:

| Production file | SHA-256 |
| --- | --- |
| `MatcoreCpuGemmCandidate.cpp` | `3f73e00ab25ce857825ee962894d02cefc462ff3e39e3a7f87fa14d789c49c56` |
| `MatcoreCpuGemmCandidate.h` | `394d90653073e9aae8b9680b7416eb0eee8941bbebeae9143576f9d95c15f263` |
| `compiler/lib/regions/CMakeLists.txt` | `f28b1e093f1ea1544288cae3311a03b0ee02e65b69cb9051bb9e4b25cb0ab54c` |
| `matcore-cpu-gemm-candidate/main.cpp` | `4ec15a8802ad291fb063a5e3053cb98dd5bfaab6da05e769f269e6feac49c7a3` |

No concrete numerical, storage, source-authority or malformed-attribute bypass
was found. The new fixed-schedule branch verifies exact maps, iterator kinds,
body and destination; required properties cannot be traded for an extra
`library_call` or other recognized optional property. Unknown discardable
attributes remain rejected. Stage inspection helpers are not execution issuers.
The CMake selector changes custom-command arguments, and the manifest records
distinct Transform, scheduled and LLVM hashes without changing semantic identity.
The added row Release lane verifies the selected manifest schedule, and the
instrumented lane selects the same row schedule explicitly. These workflow
definitions were inspected, not counted as successful hosted executions.

The actual row-selected Debug ASan+UBSan driver was then exercised with both
committed CMake source runners:

- Driver SHA-256: `22bc230dbaa5781c1bb07866d6a1925c8c5ced88875f0466ac595f108badfb97`.
- Production generated object SHA-256: `cb8cf8492e12b2b9afb95045be598a4bf579819821a882d08962730a8d3b43ed`.
- `schedule_source_contract`: **4,140 PASS**, exact output and no stderr.
- `schedule_same_value_contract`: **1,503 PASS**, exact output and no stderr.

The build cache identifies Debug, `-O1 -g -fsanitize=address,undefined
-fno-omit-frame-pointer`, and `row-contiguous`. This is actual instrumented
source-route execution, not passing an extra host flag to a Release driver.
Generated memory ASan is independently demonstrated above; raw LLVM computation
does not acquire source-level UBSan checks merely from sanitizer linkage.

The unchanged normal execution body of the leaf test also passed **174 checks**
when separately linked with both row and scalar instrumented objects. The new
SIMD expected-failure classifier deliberately rejects the scalar read report.

## Verdict at this stage

**Independent implementation ACCEPT at `9769dbaf...`, conditional on full
exact-head integration gates.**
No new generic optimization abstraction is required for this bounded leaf, but
Matcore's numerical/resource/source contracts and the new schedule-specific
derivation checks are irreducible. Integrating the candidate must not weaken
original pairing or falsely announce whole-region transformation/reuse.
Complete Release/ASan package suites, hosted checks and normal integration of
the final combined test/evidence head remain the owning lane's separate gates.
This review does not announce campaign completion or a performance/parity claim.

## Independent cache-tiled source execution and verifier challenge

This later, separate cache-schedule review does not change the row-only verdict
above. The cache implementation was reviewed read-only at base
`9769dbafc18d4b8efd05c2e3519543eb57531604` plus the owning lane's uncommitted
five-file delta. Production `MatcoreCpuGemmCandidate.cpp` SHA-256 was
`ac15b8fac8d21814879d06bf02b54e6e307fb2ec16820c51e50d8f645c432369`.

**The closed replay architecture is appropriate, with an explicit trust limit.**
Exact canonical prechecking, an isolated clone, the fixed upstream sequential
MNK tiles `[4,64,32]` and inner MKN order, and whole-module structural replay
are compiler-owned derivation/drift checks. They do not authenticate an arbitrary
imported program or independently prove upstream MLIR correct. The pinned
[OperationEquivalence implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/IR/OperationSupport.cpp#L651-L841)
retains properties, attributes, nested structure and SSA mapping with only
`IgnoreLocations`; it does not ignore arbitrary operand identity. The independent
operation whitelist, one outer fill, exact scalar body and LLVM numerical/call
postconditions remain valuable narrow assertions, not another optimizer IR.

The reviewed schedule accesses A at `(m0+i,k0+q)`, B at `(k0+q,n0+j)`, and the
same C at `(m0+i,n0+j)` across all increasing K chunks. Each tile dimension is
bounded by `min(tile, extent-origin)`; the original fill remains outside the
loops. Thus per-output arithmetic is the original rounded fold partitioned into
contiguous intervals, not independently accumulated partial sums. Upstream's
[bounded tile-size implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp#L216-L236)
and [Linalg tiling implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp#L109-L135)
were inspected; replaying those same passes would not itself detect a shared
upstream mistake. No public index-overflow counterexample was found: positive
generated calls retain pairwise byte-size bounds, and the adapter bypasses the
leaf for empty output or zero K. Allowing only the exact signed-i64 min intrinsic
for the three cache-tail minima adds pure integer arithmetic, not provider or
external execution authority.

The previous source fixtures had ordinary K at most 7 and N at most 33, so they
could not validate the new K32/N64 tile boundaries. The independently authored
[cache source fixture](../../../compiler/tests/generated_cpu/cache_schedule_source_contract.mdsl)
and [runner](../../../compiler/tests/generated_cpu/cache_schedule_source_contract.cmake)
close that specific gap with **225 cases / 563,296 checks**:

- All 216 combinations of M `{3,4,5,7,8,9}`, K `{31,32,33,63,64,65}` and
  N `{63,64,65,127,128,129}`, with lane-distinct dyadic data. A second strict GEMM
  has N=65 and K equal to the first GEMM's N.
- A K65 discriminator: ordered accumulation gives 3, while independently
  zeroed 32-term partial sums and per-chunk destination reset both give 2.
  The host counteroracles are themselves asserted before source execution.
  Separate FMA discriminators straddle K32 and K64; special-value and gradual
  underflow cases also reach the final K/N tails. Non-NaNs are compared bitwise;
  NaN payload is intentionally unspecified.
- Saved old values, three owning observations, late read and late publication
  failures with the exact earlier effect prefix, bounded destination canaries,
  and D=C+1 late overlap spanning multiple tiles. This preserves the existing
  two-GEMM semantics; it does not claim transformed whole-region execution.

All four actual driver lanes passed the exact count/output check with empty
stderr. Live SHA-256 identities were:

| Lane | Driver | Production generated object |
| --- | --- | --- |
| Scalar Release baseline | `eee73557f8769569f45b51e95a269237d2863fff5fb3b5146a69daa57fdc69e6` | Earlier scalar checkpoint, not rebuilt here |
| Row Release baseline | `f183efd71903d7a48b2c98c72bb7e722707db21a17cdea7ce3d2acdfe2518832` | Earlier row checkpoint, not rebuilt here |
| Cache Release | `18af6cab78d570980211ebcf671e146785a941625b845f58efe6120283c728ad` | `02a8657e898705e156c593a9771e4c69886390895027a256062944798f596fd8` |
| Cache ASan/UBSan build profile | `85a7bc7e7e4d02b65ba54bcfb828023d54abc344906c5eb9a61823aa8b382ffd` | `0fac7af5fb462fbfe3d38f0898f25b695d91e567cffdf63b99b196ede28bd30f` |

An initial parallel invocation shared a working directory: the compiler
correctly rejected the ASan invocation because captured directory metadata
changed. The runner was corrected to use a fresh per-invocation directory;
all four lanes then passed. An initially mistyped scalar build path was rejected
before compilation. `/bin/false` and success-without-artifact `/bin/true` are
both rejected by the final runner. Only a fresh successful executable is removed;
failed artifacts remain available. No benchmarks or production edits were made
in this independent lane. These are local source-route results, not final-head
hosted/package acceptance, a new UBSan claim for raw LLVM operations, or a
performance result. Full cache integration remains the owning lane's gate.
