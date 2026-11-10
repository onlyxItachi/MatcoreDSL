# Authenticated source-time full-N/tail GEMM request

Scope: [Issue #104](https://github.com/onlyxItachi/MatcoreDSL/issues/104), dependent
on the [private realization](PRIVATE_STRICT_OUTPUT_PEEL_V1.md).
Status: implementation and qualification in progress; no merge or performance
claim. This document defines the bounded acceptance contract.

## Request and ownership

The experimental driver accepts the separately explicit
`--candidate generated-strict --gemm-output-tiles-peel-n M,N`, with canonical
decimal integers in `[1,64]`. It is mutually exclusive with
`--gemm-output-tiles`; the old spelling continues to request the old unpeeled
recipe. Defaults, build-selected schedules, semantic IR, runtime dispatch and
provider ownership do not change.

Both routes use the same authenticated source/entry closure, fresh in-process
issuer, pinned PIC relink kit, isolated local leaf, canonical Runtime, immutable
DSO binding and typed final publication described in
[source-time v1](SOURCE_TIME_GEMM_PATTERN_V1.md). Neither accepts a supplied IR,
object, manifest, schedule callback or retained bundle as an executable input.

Only the new route uses inspection schema `matcore-source-output-peel-n-v1`,
`realization=output-tiled-peel-n-mkn` and binding domain
`matcore-source-output-peel-n-binding-v1`. The issuer manifest, all seven stages,
object and private DSO identities remain bound to source, target and toolchain.
For tile N=1, old/new computation bytes may be identical by construction;
requested realization, binding and private DSO identity must still differ.
These serialized fields describe provenance, never confer authority.

The primitive issuer remains source-unauthorized on its own. Its metadata says
that caller obligations are not discharged by primitive issuance; the separate
authenticated source invocation supplies that connection. No private artifact
can be promoted merely by changing the metadata.

## Guards, effects and numerics

On actual leaf entry the unchanged adapter has positive M,N,K and checked
`MN,MK,KN <= floor(PTRDIFF_MAX/sizeof(float))`. On the pinned LP64 targets this
bound is `2^61-1`, so every dimension plus the maximum tile increment 63 fits
signed i64. Dense indices and byte offsets are bounded by those products.
The peeled split `F=T*floor(N/T)` is at most N, with remainder less than T.
This implication is conditional on real leaf entry, not an unconditional new
source dimension limit.

In particular, valid empty values may have an extent `INT64_MAX` that violates
a raw tiled leaf's next-IV bound. Preserve the adapter's empty/zero-K bypass,
all earlier validation/capability checks, allocation and FP-state scope. K=0
with nonempty output still denotes positive-zero output. Fresh private C is
disjoint from live immutable inputs; A/B may overlap. Caller lifetime/race and
allocator FP-preservation preconditions remain explicit.

Full increasing K, separate f32 multiply/add, the original untransformed WHAT
witnesses and ordered failure/publication/observation frontiers remain intact.
No padding, K partition, reassociation permission, hidden fallback or source
storage noalias assertion is added.

## Qualification and exclusions

Required evidence: independent 3x5/5x3 source requests; old/new 5x1 identity;
actual Transform N-handle consumption and complete full/tail schedule; source
effect/alias/failure and end-of-K numerical oracles; stale/cross-realization
artifact and typed-publication refusals; actual generated-leaf sanitizer
detection in both domains; relocated producer-inaccessible installation; native
ARM, affected regressions and independent review.

The retained old inspection classifier and P1 performance harness remain
unchanged. The new outer schema is refused by the old complete manifest gate.
Identical unit-N graph bytes alone cannot distinguish provenance. P1 itself
requested only 3x5/5x3; this is not a claim of an old unit-N campaign bypass.

Support remains single-TU native-baseline Linux x64/AArch64 complete executables
with `none` or checked forwarding. No `-c`, program mode, per-operation selection,
other ISA candidate, fused-pair composition, GPU schedule, cache/import authority,
post-publication relocation, autotuning or performance claim.

After source correctness/ownership qualification, exactly one next boundary is
a separately preregistered source-call comparison against old and stock routes.
It must retain negative/inconclusive outcomes and account for compilation,
code-size/inlining, call/allocation interval and strict numerical equivalence.
It must not rewrite the frozen P1 plan or infer speed from register promotion.
