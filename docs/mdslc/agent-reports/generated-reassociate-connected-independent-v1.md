# Independent connected reassociate / installed owner review

2026-09-09. **ACCEPT within the selected synchronous Linux x86-64 contract.**
Independently authored full-source/storage/failure tests and a separate installed
private-adapter consumer pass against the actual integrated generated candidate.
No production correctness counterexample was found. This adds connected evidence
to the [primitive review](generated-reassociate-issuer-independent-v1.md), not a
general optimizer proof, performance result, or public candidate telemetry API.

## Exact reviewed inputs

Integration worktree:
`/home/hamza-usta/MatcoreDSL-wt-generated-reassociate-integration-v1`, clean
`73cdeb927c160c9eda691aeb7f666b9f22d3a6b3`. Production/tests were composed at
`55dd87b15b19121d5bdf7ef23abe807a00033310`; its two successors are documentation.
Canonical base is PR #63 merge `917b5ecf1d325e525bc24e3c94764d2958980a93`.
This report does not infer hosted or complete CTest results from focused runs.

Before the final test-only tightening below, the source fixture SHA256 was
`8ceb0545f091ab6d15acfdffad806744593785830c4cd808dbf8b86206c492d8`;
private consumer `b064c813f4d3b61e4594a4621372e35f6eacdb75ec0b235c7f1a99be0f719937`;
runner `874fa66c2d3c9f303e748c14ba0622981e0801cc38e157190393df03ec87ad15`.
They live under `compiler/tests/generated_cpu/reassociate_independent/`.
No runtime/issuer/driver changes were made in this review. Static composition
review retained the H1 mandatory SourceFile table with file id 1 bound to the
selected strict/reassociate builtin identity, per-site source diagnostics, and
the union of header-library and candidate tests. A numerical profile is not
inferred from a renamed strict buffer or a private implementation label.

| Build artifact | Release/provider ON SHA256 | ASan+UBSan/provider OFF SHA256 |
| --- | --- | --- |
| Actual mdslc-region | `6423db0db4ea0db91d4c4f747f4160de5508c432b496be08267745d690bf96b5` | `c71d5f8b6da590e65d116857db3120aa40e4d5145c2ad862c1f6546dc7fb3d40` |
| Private candidate DSO | `262cfc9c40bf23d2bfa326fb1a17d78afd91243cea03e3e1c0fbbf98060c9e4c` | `b8e8dcd74d00379761fdcb08ba545a1714c4ba221fabd75c2026378382ddd874` |
| Canonical Runtime | `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014` | `7db374591f0522be0a2088765eb375d74003da5b65559b90ea3f1d916b7f9488` |

Fresh `cmake --install` prefixes are
`/tmp/mdslc-reassociate-connected.opuCTy/sdk` and sibling `sdk-asan`.
Source, build, and system provider remained accessible: **installed resolution,
not source-inaccessible execution or arbitrary relocation**. CMake removed build
RPATHs during installation, so installed identities are separately recorded:

| Installed artifact | Release SHA256 | ASan+UBSan SHA256 |
| --- | --- | --- |
| mdslc-region | `b68a92c71e925d4cd6f218a0b67d608ce5f9c1a00f53463458211e82b438196f` | `61a1a26e07e8609b0e2c7c436e035163714a2e46e2d0bcd3d04bbaf4fc37e187` |
| Private candidate DSO | `06a2dcc1b220fd1e681f6a5de303617ba80cd76a525341789065bfd84ba93890` | `bc472c21809b4a0280a06782ccbad7ce38bf97e1ad99d59d2be4957061745709` |
| Canonical Runtime | `bbfbbd75b6c73aa7f1cd22227cdc2ab2fa6bdc21b3dee045f9b4346e6dc1851e` | `7db374591f0522be0a2088765eb375d74003da5b65559b90ea3f1d916b7f9488` |

Installed public `region.h` is `2cbff60e4c5685d1388604df9642126e31296e067bd471e6eae9a70a26c2a39b`;
`detail/region_storage.h` is `8803a9b31de0e72ffb1045acfede828473b08f9f7292e4fefd27e1dc9b1c0927`;
private `closed_host_v1.h` is `7de5029cc848437d1365614b76b37de0f3601c64eb09553ba98ed58de53eeff3`.
The configured provider spelling was `/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so`,
SHA256 `be2e7d119279836105e0361be92c78dbcd8a6e7357e74339bdbb32a5195ee35e`.
Ordinary consumer compiler: Ubuntu Clang 21.1.8 (6ubuntu1), x86_64-pc-linux-gnu.

## Independent oracles and actual results

The region reads an 8x8 immutable original, computes `gemm(original, original)`,
publishes at arena offset 5, observes, reads a late slice at offset 4/5/6, computes
`gemm(original, changed)`, publishes at offset 6, observes, and completes. Input
begins at offset 4. All external descriptors are mutable/MAY-alias. Every arena
element, including boundary sentinels, is compared bitwise to an independent
host state-transition oracle. Original-value rereading, premature late snapshots,
and shifted operand reversal each have independently unequal counteroracles.

The deliberate 63-element final capacity fails at publication frontier **7**,
completed frontier **6**, effect frontier **4**, original source **23:3**: exactly
one publication and observation survive, with the complete arena unchanged from
that first publication. Success completes frontier **9**, effect **8**, with two
publications/observations. The first owning observation survives Result destruction
and replacement of every external arena element. Both paths preserve errno,
x87 control/status words and full MXCSR including caller FTZ/DAZ/exception bits.

Public overlap exercises ordered snapshots, not overlapping private input buffers.
The first `gemm(v,v)` really supplies the same immutable input pointer twice;
the separate private consumer confirms result storage differs from that input.
Raw leaf partial-input-overlap conformance remains separate primitive evidence.

For full-tile arithmetic, sparse A has A00=-1, A01=`0x1.000002p0F`, A02=1,
A12=`0x1.fffffep-1F`; A*A02 distinguishes fused accumulation from separate f32
rounding. Every result is compared with the selected implementation's host
`std::fma` or volatile separately rounded oracle. Installed DSO disassembly has
four YMM `vfmadd231ps` operations in the leaf's increasing-K loop at 0x3d0b,
0x3d1d, 0x3d22, 0x3d2d. This bounded full-tile realization is not a claim that
all reassociate-permitted implementations must produce the same bits.

| Pre-tightening actual matrix | Positive executions | Assertion evaluations | Deliberate negative controls |
| --- | ---: | ---: | ---: |
| Release, six build + six installed source policies + private consumer | 13 | 30,023 | 5 passed |
| ASan+UBSan, five build + five installed source policies + private consumer | 11 | 27,045 | 5 passed |

Each generated-reassociate/generated-strict/automatic/native-strict source run
checks 2,968 assertions. Existing-native and optional OpenBLAS check 1,489 exact
integer assertions; their unspecified legal rounding is not forced to FMA.
The private consumer checks 323 assertions. No hardware skip occurred.
The four source rounding-label swaps each exit **1**, with exactly **21** numeric
oracle failures; swapping the two expected loaded owners exits **1**, with two
owner failures. Incorrect reporting cannot satisfy these independent controls.

Private CandidateReport identifies `closed.generated.reassociate_f32.avx2_fma.mlir21.v1`,
strict generated for both explicit strict and automatic, and native strict.
It asserts invocation/value/thread/numerical fields, verifies empty/K0 are marked
adapter-local with no invocation, and rejects strict after empty success without
claiming an actual implementation. `dlsym`/`dladdr` bind the private ABI gate to
the installed candidate DSO and runtime GEMM to the exact installed Runtime.
The issued leaf/wrapper are local ELF definitions, unavailable through dlsym.
Executables directly need the private DSO and canonical Runtime; the manual ON
consumer additionally directly needs the configured provider SONAME. Installed
Runtime needs `libopenblas.so.0` only in the ON build. Loader override variables
were removed; no separate Runtime/provider adapter was embedded into the consumer.
**These are private adapter/loader observations, not public Result telemetry.**

## Reproduction, corrections, and limits

Use the README runner invocation with the above build driver, fresh output path,
matching SDK, and ON provider. For OFF pass `--sanitized` and omit `--provider`.
Source sanitization is installation-owned; only the ordinary private final link
adds `-fsanitize=address,undefined`. The runner verifies source `__asan_init` and
uses halt-on-error/leak/strict-string/initialization-order ASan checks and
halt-on-error/stack-trace UBSan. Actual generated-load fault controls remain the
separately reviewed READ4/READ32 primitive gates, not inferred from no-error runs.

Two harness-construction failures are retained honestly. The first source run
had two bad noncommutation counterassertions: unshifted changed=A*A makes
A*(A*A)==(A*A)*A for exact integers. The correction restricts that discriminator
to shifted slices; no output oracle or production code was changed. The first
ASan launch passed a user sanitizer flag and was correctly refused with
`sanitizer profile belongs to this compiler installation`. The runner now uses
the actual installation profile and verifies it; this was not a compiler failure.

These pre-tightening raw commands, diagnostics, artifact identities and output hashes are under
`/tmp/mdslc-reassociate-connected.opuCTy/{release-final,asan-final}/evidence.json`.
Their respective SHA256 values are
`296542fc64cf19acb61517a11b1dcc7a80e29bcdb6267f0e08e626cac6f17df0` and
`1041d91f2687948c28216f9732c4a6af6d43c42c5f7e8e6a28e1deb6a4245aae`.
All recorded source/compiler/DSO/Runtime inputs and generated executables were
hash-checked unchanged through the complete run, including negative controls.
No timings, source-hiding, archive-consumer authority, universal numerics theorem,
default policy change, generated GPU/Windows, or generic sandbox claim follows.
Static review preserves per-GEMM profile checks and the original adapter's
stronger normal-return publication guarantee; future transformed derivations
must preserve that selected guarantee, not merely generic partial-write latitude.

## Final independent test tightening

Root and the separate issuer owner accepted the bounded source/private math,
alias and ownership distinctions in read-only peer review. Both requested
stronger harness evidence controls, not a production correction. The final
source and private consumer explicitly check their requested nondefault FP
setup succeeded. The runner pins every input and every previously produced
executable before and after **each** subprocess, registering each fresh output
immediately on successful compilation. Positive commands require empty stderr;
negative source execution requires exact count/identity and math-only diagnostic
lines, while wrong-owner execution requires exactly its two owner diagnostics.
Thus an intended exit 1 cannot hide an additional sanitizer/crash diagnostic.

An exact known hardware-unavailable stdout with exit 77 and no stderr propagates
as whole-suite SKIP, not failure or success. Other exits or malformed/diagnostic
skips fail. CTest integration must retain `SKIP_RETURN_CODE 77`. These synthetic
controls do not pretend that hardware was unavailable on this machine.
`runner_test.py` passes **20/20** classifier and changed/missing-file controls.

Final source SHA256:
`3690e9d5b81b34f4c951fc13bda8de82b4d80490f39d4fcd629dcc4b9a2d4969`;
private consumer `4a70f8ce65cfef7443fdb499525263ab845e4bd763342ebab43396a229ed196d`;
runner `9c5add4738d6066e4b1ab5d426ab3bc8f8c71fdf92ced89c441226cd8ce99e9f`;
synthetic controls `998e86340a15a926e63f529cb693b9666b8a7cc61325982f1b88ee918af2fed0`.
The preceding source/runner hashes and results remain historical, not relabeled.

After the coordinated quiet window was released, both complete strengthened
matrices passed against the same unchanged build/installed artifact identities:

| Final matrix | Positive executions | Assertion evaluations | Negative controls |
| --- | ---: | ---: | ---: |
| Release/provider ON, build + installed | 13 | **30,147** | **5/5 passed** |
| ASan+UBSan/provider OFF, build + installed | 11 | **27,157** | **5/5 passed** |

Source counts are now 2,980 for each generated/automatic/native-strict run and
1,495 for each existing-native/provider integer run; the private consumer checks
327. The four rounding negatives still each exit 1 with exactly 21 mathematical
failures, and the private owner negative exits 1 with exactly two owner failures.
There were no positive diagnostics, sanitizer diagnostics, or hardware skips.
The final raw evidence files are sibling
`release-strengthened/evidence.json` (SHA256
`485126a607ee777758446819d9c408f8198a091e90d83b9ef2a8f4c220828f01`) and
`asan-strengthened/evidence.json` (SHA256
`232d7aad110651257c76e87fc26b21b8fbd9394af4f6e5279fc92cec051b03d3`).
Root separately reported full Release 172/172 and affected ASan/UBSan 117/117
at the frozen integration checkpoint; those are root-owned regression results,
not an independent rerun or future hosted-CI claim. Root owns subsequent permanent
CTest registration and inventory reconciliation for these new test-only fixtures.
