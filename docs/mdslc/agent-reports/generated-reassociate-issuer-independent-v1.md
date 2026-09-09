# Independent reassociate issuer review

**ACCEPT within the bounded primitive contract:** reviewed production issuer
`1c03306ee771f785378112866111288d8d7c5f1e` plus test-only classifier correction
`62379dd6323b2aa8d93d840b1f426b92a1c5d176` in
`/home/hamza-usta/MatcoreDSL-wt-generated-reassociate-gemm-v1`.
Both source and issued artifact identities remained unchanged through the final
independent executions. This is not acceptance of combined source/runtime,
installed SDK, hosted CI, automatic selection or performance.

The reviewer wrote only the independent fixtures and this report in
`/home/hamza-usta/MatcoreDSL-wt-reassociate-issuer-review-v1`, based on canonical
`c04215e787832b35bbd6a85a51d9285b053181bc`. No issuer/adapter edits, existing-build
modifications, full builds, benchmark runs or toolchain/package changes occurred.

## Semantic and lowering findings

The new builtin has its own source identity, contract digest and real per-GEMM
`reassociate_f32` profile. Cross-operation reassociation stays forbidden. This
is not strict semantic authority renamed after lowering. Structured GEMM begins
from the same valid separate-arithmetic form, with exact ordered A/B inputs,
positive-zero fill, caller C destination, and no tensor allocation/copy after
One-Shot Bufferization. A renamed strict *buffer form* deliberately passes the
new self-consistency check because that arithmetic form is legal under either
profile; it does not authenticate a source program. The closed issuer accepts
no imported semantic, buffer, scheduled or LLVM payload.

The full-tile schedule is serial, with floor-bounded M/4 and N/8 loops. Its
4x8 accumulator comes from already initialized private C, remains loop-carried
through complete increasing K, and is written after K. Input transfers remain
inside K; K0 touches no A/B, while M0/N0 execute no vector transfers. Scalar
remainders retain separately rounded multiply/add; no fictitious K padding,
hidden allocation/copy/publication, distribution or blanket fast math appears.
The narrow independent envelope checks these core facts, while exact upstream
replay additionally fixes all remainder indexing/bounds and remaining SSA and
attributes. Replay is dependency identity checking, not a generic independent
proof of upstream transformations or LLVM.

Only expanded aligned-C argument 15 is noalias. Input/input aliases remain
legal; allocated-C and descriptor pointers gain no such guarantee. The LLVM
replay checks the actual wrapper mapping as well as the parameter attribute.
Target attributes and isolated compilation are baseline `x86-64` with AVX2/FMA,
not full x86-64-v3. The normal object's K loop at offsets `0x4c0`–`0x4fb` has
four YMM fused accumulators, one B vector load and four A broadcasts, with no
C access or accumulator spill inside that loop; its only ordinary import is
`memset`. These are inspected structural facts, not timing evidence.

## Independently authored mutations

`compiler/tests/generated_cpu/reassociate_independent/mutations.cpp` passes
**47 checks** with the ordinary issuer library and again with the ASan/UBSan
issuer library. Each negative mutant first passes its underlying MLIR/LLVM
structural verifier. Wrong scalar-tail A-row and B-column changes specifically
reach the exact-replay diagnostic, exposing the intended limit of the narrow
envelope. Tests also refuse zero-trip initialization, a transposed full-tile
transfer, wrapper aligned-C redirected to A or allocated-C, invented alignment,
wrapper-only sanitizer marking, and changed LLVM source filename.

Changing even unused in-bounds transfer padding is rejected by exact replay;
this conservative identity boundary is not a claim that the change is itself
mathematically wrong. Conversely, changing MLIR diagnostic locations or LLVM's
diagnostic module identifier is accepted as documented. Both genuine builtin
profiles and the no-cross-GEMM permission are checked directly.

The test source compiled with Clang 21.1.8; link arguments came from read-only
Ninja command inspection, replacing only the test object/output and removing
the existing dependency-file output. Linked archive hashes were checked before
and after; generated review artifacts stay in
`/tmp/mdslc-reassociate-independent-mutations-v1`.

| Artifact | SHA256 |
| --- | --- |
| Ordinary issuer archive | `837018337f8756130e51caf42b648bce3e31ea0f1a9abeacf0ce603e65fefec4` |
| ASan/UBSan issuer archive | `7cae74d5719ce76fb35afde975f7df23686c272e4d7c9edef364d19058b0a120` |
| Independent `mutations` executable | `9fa07f3773021164f3559bd1f3770da34cc90f2bd25a1665908aee39e6b3a78f` |
| Independent `mutations-asan` executable | `1e71383465454b7b27fffa345957eab048236e118b10a0ccc43c5753c487d341` |

## Real leaf rerun and classifier counterexample

At the frozen issuer, the reviewer independently executed both existing numerical
harness binaries: each passed **1,417 cases / 109,632 checks**. Corruption and
strict-result counter-oracles in both binaries returned the required exit 1
and chosen-realization mismatch. These reuse the inspected committed numerical
oracle; the separate 47-check mutation suite is independently authored.

The actual ASan A adversary produced a primary **READ 4**, and the independent
undersized-B full-tile adversary a primary **READ 32**. Both returned exactly 1,
reported heap-buffer-overflow, and named
`__matcore_reassociate_gemm_f32_avx2_v1` as the first fault frame. The reviewer
inspected the primary traces, not just a symbol appearing elsewhere in a log.

A synthetic adversary then found a real test weakness at `1c03306`: the wrapper's
independent greps accepted an unrelated primary fault owner if a later
allocation stack contained frame `#0` with the generated name. The initial
13-control suite passed; adding this 14th control failed with actual wrapper
exit 0 where rejection 1 was required. This did not invalidate the inspected
real faults or demonstrate a numerical/code-generation bug.

Correction `62379dd` binds the first ASan error, its READ width and first following
stack frame in one ordered classifier. All **14 controls now pass**; wrong
statuses, scalar versus wide load, WRITE, non-top/wrong/secondary owners, wrong
error kind and missing error cannot prove the claimed read. Skip 77 remains a
skip, not a fabricated pass. The reviewer reran both real ASan faults with this
corrected script and unchanged binary. This is a test-only correction; no
production verification was relaxed.

Raw execution logs are under `/tmp/mdslc-reassociate-issued-review.96WP3z`.
The ordinary and ASan successful logs share SHA256
`6fbb6c554307ddde6029e114d6508b9ee04fc10bb4373057967eb21574c86808`.
The original independently inspected A/B primary error logs have hashes
`95d280efa06cb3eff6a6132db38a50de803115014a8ccb99139fbe20b3dbdedd` and
`2229ceb0130d8100c726d02c583c67bd9a4bfcd36229806498f4b85b415581e7`;
corrected-classifier reruns are separately preserved in `corrected-read-a/b`.

## Frozen identities and limits

| Item | SHA256 |
| --- | --- |
| Register issuer source | `26166ceb5e7a7169568284cf2c4ff6e7de90656d516830cde2a2a5358e251fee` |
| Shared stage/strict source | `50f381cd4d3c7f1411dec0256e13407c1f743b39485352a741613424d6abb073` |
| Normal issued object | `22285cebb5ea339caf3b65170a41b316de1110091b7dac7f5ca8f0580d71cfd3` |
| Actual-ASan issued object | `d030880dfd31d7c0a43bcb0143535b513e7664936f6a51058a745c58bc33dbd2` |
| Normal existing numerical harness | `2692572d12a372c58dcab41d6a7702b674704a0473dd4de736d748b45086730e` |
| ASan existing numerical harness | `2923ff055db6023717428e0fa003bb378cf133a5c828f7514e0dfdc35072bb4b` |

ASan mutation/success runs used empty `DEBUGINFOD_URLS`, leak/error/strict-string/
initialization checks and UBSan halt+stacktrace options. Fault controls use the
committed wrapper's exact exit-1 sanitizer configuration. No physical capability
skips occurred on this Linux x64 Ryzen AI 9 HX 370. This is not validation of
another CPU family, a generic hoister, arbitrary descriptor lifetimes, source
authority, public candidate telemetry, or performance/BLAS parity. Combined
runtime/source, failure-prefix/FP and installed-package gates remain separate.
