# Independent bounded MLIR/HPC campaign terminal audit

## Verdict: conditional completion, not universal compiler sufficiency

**Yes:** the recorded [campaign objective](../MLIR_HPC_EXECUTION_CAMPAIGN_V1.md)
is satisfied once the surviving source-connected candidate and multi-source
work are normally integrated, their **combined** relevant gates pass, and the
canonical checkpoint is recorded. The result is a reusable, source-authenticated
Linux x86-64 f32 CPU compiler path with useful generated implementations,
source-visible mathematical libraries and ordinary multi-file host programs.
It is no longer only a standalone kernel demonstration. No counterexample found
in this audit requires a new machine-level IR or replacing MLIR's mechanics.

This is **not yet a merge/completion report**. On 2026-09-09, live GitHub main was
`b7e6cc03bd492ba9d0b2983dd1ce54611ef971fd`; H1's engineering merge was
`917b5ecf1d325e525bc24e3c94764d2958980a93` / PR #63. Reviewed candidates:

- PR #66: `76bcf62183712544b202ce58290f04fef7502e48`, still open with hosted
  checks in progress when inspected;
- multi-source: `ed8f7fed1d4a1969328e4f0c9626f0d98151169e`, including reviewed
  implementation `3ef0a78` and independent test/report `1876063`.

One concrete union obligation: multi-source moves policy parsing from `main.cpp`
to `DriverSupport.cpp`; PR #66 adds `generated-reassociate` in the former.
The merged parser must preserve that policy in **both** driver modes, with
strict-profile refusal and ordered failure tests. Separately green branches
are not evidence that this intersection already works. This audit ran no build,
benchmark or new production test and authored only this document.

## What the evidence proves

| Question | Bounded answer |
| --- | --- |
| Can upstream mechanics produce useful CPU code? | Yes: strict M/K/N scheduling and permitted register-retained FMA execute on real hardware through authenticated source, not merely verified IR. |
| Can mathematical code be reused? | Yes: H1 admits actual source-visible Value/Shape helper bodies and bounded selected templates, preserving inclusion FileID and source identity. |
| Can real host programs span files? | Yes within the reviewed 2–8 source-TU contract: original sources/closures, clean foreign ABI/effect witnesses, protected symbols and ordinary linking compose independent regions. |
| Does successful transformation imply a good schedule? | No: both explicit-vector and cache-tiled alternatives produced correct but losing implementations. |

PR #66's [integration record](../GENERATED_REASSOCIATE_CPU_V1.md) distinguishes
full 172/172 Release and 117/117 affected sanitizer runs at `73cdeb9` from later
two-test additions and still-pending final hosted gates. The independently
[audited comparison](generated-reassociate-source-measurement-independent-v1.md)
rechecks 540 samples/60 medians: the new candidate used 27.28–57.62% less time
than generated-strict on six **identically permissioned** source workloads;
OpenBLAS won five. This joint schedule/ISA/optimization intervention is not an
FMA-only effect or BLAS-parity envelope. The [multi-source review](authenticated-multi-source-independent-v1.md)
independently passed 39 source checks and 36 real-Clang ABI/effect checks in each
Release/sanitizer profile, including actual source counterexamples and corrections.

## What Matcore actually had to add

The necessary abstraction is a **legally selectable implementation of an
authoritative operation**, with explicit source/profile/resource/effect and
target requirements—not another loop/vector/backend language. Existing immutable
values, all-MAY-alias external resources, read frontiers, ordered publication/
observation/failure contracts remain indispensable. Candidate-private storage
and the precise fresh output-data pointer noalias fact are distinct from host
descriptor identity; no extra alignment is asserted. Numerical permission is per
GEMM; a fused implementation cannot lend
permission to strict calls or erase the intermediate f32 boundary between GEMMs.

`MatcoreCpuReassociateGemmCandidate.cpp` uses upstream tile/peel/vector/transfer
hoisting and lowering; its narrow independent envelope plus exact replay guards
one compiler-owned derivation. Replay is not a second independent proof that
upstream is correct, and cannot authorize arbitrary imported IR. Clang still
owns C++ Sema/code generation, while Matcore authenticates bodies, linkage and
observable contracts. The weakref call-only promise and injected `mdsl_probe`
counterexamples required source/call-interface corrections, not new mathematics.

[Pinned upstream Transform operations](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td)
already supply the selected structured mechanisms. Matcore chooses legal inputs
and candidate policy; LLVM owns instruction selection, registers and machine
scheduling. [Transform failure](https://mlir.llvm.org/docs/Dialects/Transform/#failure-propagation)
belongs to compiler candidate construction, not a source runtime failure or
permission to roll back/skip an earlier publication.

## Keep losing paths rejected

- The strict vector-contract/FMA route demonstrably changes strict results;
  `-ffp-contract=off` cannot undo an already introduced FMA intrinsic. Reuse it
  only under the separate authenticated numerical permission, as PR #66 does.
- Keep fixed cache `[4,64,32]` and the earlier masked-vector candidates as
  [research evidence](https://github.com/onlyxItachi/MatcoreDSL/blob/93e340a/docs/mdslc/agent-reports/cache-measurement-and-numerical-next-v1.md),
  not product/default selectors: passing correctness did not justify their
  maintenance cost, and cache lost all six measured source workloads to M/K/N.
  This does not reject tiling in general; retain their boundary tests.
- No blanket fast math, three-way A/B/C noalias, descriptor-only alias shortcut,
  x86-64-v3 execution from an AVX2/FMA-only guard, or unmeasured thresholds.
- Keep 22.1.8 a separately coherent compatibility experiment. It executed the
  research schedules but neither cured the strict-FMA counterexample nor
  authenticated a migrated product driver/package. No private replacement
  tiler, register allocator or opaque module format was justified.

## Still false; exactly one next boundary

The whole-region Matcore MLIR remains an **untransformed exact paired witness**;
static orchestration invokes individually issued GEMMs. This is not transformed
whole-region execution, fusion, automatic reuse, cross-region reordering, general
rank-N/view semantics, opaque mathematical imports, accelerator residency,
asynchrony or generated-region Windows support. The general contraction
inspection model is not an executable rank-N implementation. Nor is this a
sandbox, crash-atomic publication, arbitrary provider identity guarantee, stable
API/ABI, global optimum or Issue #15 completion. Snapshots are the current
realization, not the definition of a logical value.

**Next: checked read-after-publication forwarding inside one closed region.**
Keep the authoritative Program and exact MLIR witness unchanged; a separately
checked derived plan may reuse the successfully published immutable Value for
a later read of that same sealed resource/version. Retain the original read's
requested-extent, shape and descriptor guards in their original order, including
when its result is dead, its source failure location, sticky failure handling
and completed frontier. Do not replace an ordered publication or observation.

Final independent documentation review clarification: every derived realization
must preserve the selected adapter's normal-return guarantee that failed
publication leaves its own destination unchanged. The generic dialect's
partial-write allowance cannot justify weakening this stronger guarantee.

This recommendation **revises the initial serial structured two-GEMM precursor**.
That precursor was safe but not necessary: reproducing already serial
orchestration in another representation would not itself demonstrate additional
optimization freedom. Forwarding directly proves one eliminated snapshot/copy
under the existing semantics. The [resource contract](../FOUNDATION_RESOURCE_DECISION_V1.md)
items 1–3 and 8 define immutable read-at-frontier meaning, preserve source-required
checks, and explicitly permit removing an allocation opportunity. The private
adapter header likewise makes allocation-attempt counts implementation details,
not fixed mathematical trace events. This does not permit a speculative later
failure to retire before an earlier required effect.

The bounded derivation must establish a successful dominating publication on
every applicable taken path, exact source resource identity (not pointer equality
between unrelated descriptors), matching checked shape, no intervening
potentially aliasing publication, and a retained immutable Value lifetime.
An unproven branch join or potentially aliasing write must disable forwarding.
Required falsifiers include:

- `publish(v,C); read(C,m+1,n)`: fail at the late read, preserving publication,
  even if the result is unused. Do not erase a check as apparently redundant.
- `publish(v,C); publish(w,D); read(C,...)`, with `D=C+1`: do not forward the
  stale value. A branch-local publication also cannot justify an untaken path.
- Save an old value and an owning observation, then overwrite C: both must retain
  their original contents. No observable publication may be elided.
- Allocation-failure injection must retain allowed source-frontier/effect
  prefixes, not demand the old snapshot's removed nth allocation failure.
  Remaining failures cannot suppress earlier publication or observation.

This is Matcore-owned resource/effect fact consumption, not a generic liveness,
alias, bufferization or fusion engine. [One-Shot Bufferize](https://mlir.llvm.org/docs/Bufferization/)
already analyzes tensor uses and buffer conflicts; it cannot recover missing
host observation/failure obligations. Broader physical storage analysis should
use upstream machinery when those obligations are represented. No implementation
of forwarding, new public API, numerical change, cross-region optimization or
additional execution authority is supplied by this recommendation.
