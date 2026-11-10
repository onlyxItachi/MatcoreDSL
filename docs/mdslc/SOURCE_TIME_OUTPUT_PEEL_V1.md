# Authenticated source-time full-N/tail GEMM request

Scope: [Issue #104](https://github.com/onlyxItachi/MatcoreDSL/issues/104), dependent
on the [private realization](PRIVATE_STRICT_OUTPUT_PEEL_V1.md).
Status: implementation and bounded qualification complete; no performance
claim. This document defines the bounded acceptance contract.
The [final qualification record](agent-reports/source-output-peel-qualification-v1.md)
separates actual local/hosted/installed execution from retained earlier attempts.

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

### Retained sanitizer integration counterexample

At source head `281a346d3cbb1aef2cb5bb96fb29e01d7abf3a31`, normal focused
qualification passed 4/4, but actual sanitized source issuance refused the new
3x5 request before linking/publication: the compiler-emitted peeled object
imports `__asan_report_store_n` for unaligned output-lane SLP, absent from the
old source object's allowlist. The same sanitized installation executed the old
3x5 request's 68 cases / 111292 checks successfully. No new executable or bundle
survived refusal. Normal correctness and private-leaf ASan execution therefore
did not establish sanitized source integration.

The correction permits only that exact fatal reporter for the sanitized
explicit peeled realization. The old common list, uninstrumented route, unknown
selectors, fixed-width `store8` and `_noabort` remain unchanged/refused. A
compile-time truth table defends this narrow gate; it does not create an object
import API. See the pinned [compiler-rt implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/compiler-rt/lib/asan/asan_rtl.cpp).

The counterexample is retained at
`/var/tmp/mdslc-source-peel-asan-refusal-VNisCYfH`; refusal stderr SHA-256
`83389b9d52f867b8cabc1ccc69bffd6baaeaa209a2a58baa1f8eb15bc3e9203c`.
The old source control stdout hashes to
`2a6ac559616ca1006b86c6bf4ae055ca4df6621f3009055df7cb8759faf76ecb`.
The initial normal full-build attempt was deliberately interrupted at 490/497
steps, before CTest, and is not a full regression pass.

After the independently reviewed correction, the local ASan+UBSan source scope
passed 4/4 CTests, zero skips, 171.96 seconds; its 158 mutation checks also passed.
The actual source-owned DSO full/tail-N input-capacity violations were detected.
Retained `build-source-peel-asan/store-report-fix-focus.log` SHA-256
`a33d938f92a71c5c82b43e7416b42bb464ead3bec9a8c59dcad2881aa2aa9106`;
JUnit `dc5e45650dbe4d5082902423f9207a583e93c96bd84db4cd7f7da67d1b679d65`.
That receipt remains focused historical evidence, not a substitute for the later
committed-head full/hosted/native-ARM/installed qualification linked above.

Support remains single-TU native-baseline Linux x64/AArch64 complete executables
with `none` or checked forwarding. No `-c`, program mode, per-operation selection,
other ISA candidate, fused-pair composition, GPU schedule, cache/import authority,
post-publication relocation, autotuning or performance claim.
Source-owned sanitizer qualification is x64-only; AArch64 source qualification
is normal execution, not inherited from private-leaf sanitizer results.

The separately preregistered [P5 source-call comparison](../performance/cpu/source-pattern-p5-2026-11-10.md)
has completed after source qualification. Correctness/identity passed, but no
comparison pair met the frozen timing rules: all ratios remain null. Exactly
one next boundary is [P6 / #108](https://github.com/onlyxItachi/MatcoreDSL/issues/108),
prospective measurement-method qualification, not a rerun or relaxation of P5.
Keep compilation, code-size/inlining, call/allocation/retained-ownership interval
and strict numerical equivalence explicit. No speed follows from register promotion.
