# Authenticated source output-peel: qualification checkpoint v1

Status: **ALL 26 HOSTED CHECKS SUCCESSFUL; normally merged 2026-11-11** in
[PR #105](https://github.com/onlyxItachi/MatcoreDSL/pull/105), implementing
[Issue #104](https://github.com/onlyxItachi/MatcoreDSL/issues/104).
Canonical source merge: `6d30a204aa6f7224b736c2035de15ce20b0e6b3d`, parents
`c903351f7009848a00c83498e50183135e9c6404` (PR95) and qualified head
`cf154deacea2bddb3240660c1af27ff4ffbc4042`. Its tree
`2aab2289ccfa246c5f921b2e741263fc96217e48` exactly matches the independent merge
projection. Production/tests/headers/workflows match that qualified head; the
whole compiler tree differs only by P1 README navigation from PR95. This is
source/tree equivalence, not a new binary build or a relabeling of P5's SDK.
The qualification and historical attempt receipts below retain their actual
pre-merge identities and executed scopes.

Frozen source is `1ad7fe56b5fcb10efe37e523206228e9ef303f6a`, tree
`4e784afa700c447fe28b19e55443ed55b2ce140c`. The source worktree was inspected
clean. Completed evidence now includes the frozen local 356-test regression,
producer-inaccessible installation, committed-head x64 source ASan scope,
hosted x64 Release/Debug/ASan scopes, and normal-source native ARM under three
schedules. The original head's read-only PR query recorded **24 successful
checks and two Release deadline cancellations**. The first CI-only follow-up,
`d3460226d7f9358ac815e6f49c73a7a9cff8bbea`, subsequently completed with
**24 successful checks and two Debug deadline cancellations**. Its completed
Release/Windows/TSan/ASan/ARM scopes and cancelled Debug prefixes remain distinct
historical evidence below; no cancellation is counted as a pass.

At qualification, PR head **`cf154deacea2bddb3240660c1af27ff4ffbc4042`** had all 26
fresh checks successful while still open/unmerged. Canonical main then was
`1f7c491f230251d04d42b5d4a0af5eaf29bcb685`. That qualified head includes the reviewed Release and
Debug 60-to-90-minute job budgets and their reports/clarifier only. Tracked
compiler/runtime/test/header and source-contract files remain byte-identical to
`1ad7fe5...`; the producer and installed SDK remain frozen there. Completed
final-head PR Release/Debug/ASan/ARM/Windows/TSan scopes are recounted separately
below; other successful checks receive no invented execution totals. Old greens
are not carried over as current-head passes. Retained unrelated ISA skips remain
explicit. PR103's private qualification is a prerequisite, not substitute source
evidence.

This writer inspected existing source, logs/XML and the preservation receipt;
no configure, build, compiler/generated program, sanitizer or benchmark was
run to prepare this report. Earlier lane reports describe their historical
authored/initially pending status. The results below are retained root-run
local/hosted evidence; the subsequent canonical merge is separately identified above.

## Additive request and immutable ownership

The [frozen contract][contract] admits a separately explicit
`--candidate generated-strict --gemm-output-tiles-peel-n M,N`, canonical decimal
integers in `[1,64]`. A closed two-value selector defaults to the old unpeeled
realization for existing calls. Both flag orders conflict, duplicates/unknown
selectors refuse, and exactly one explicit generated-strict candidate is
required. Program mode, `-c`, fused-pair and other candidate/ISA/import requests
are excluded. There is no new runtime candidate, default/build-selected
schedule, source shape limit, numerical permission or automatic policy.

The request reuses `SourcePatternInstance` and its existing authenticated
source/entry closure, in-process issuer, pinned native PIC compilation/relink
kit, local private DSO leaf, canonical Runtime, typed source/compiler pairing
and no-clobber publication. The original Program/WHAT witness is unchanged.
No second compiler pipeline, callback, artifact cache/importer or source hash
schema supplies authority. Concrete tool/header/kit/export/Runtime identities
and source input rechecks retain their existing trust scope, not hostile
build-origin attestation or post-publication executable self-authentication.

Only the new route uses `schema=matcore-source-output-peel-n-v1`, additional
`realization=output-tiled-peel-n-mkn`, and binding domain
`matcore-source-output-peel-n-binding-v1`. The exact old schema/domain/field
construction is retained. Stage/object/issuer-manifest hashes feed the existing
request-bound SONAME and DSO binding. Unit N deliberately keeps equal old/new
Transform, scheduled, LLVM and object computation while preserving distinct
source schema, issuer manifest, binding, SONAME and DSO identity. Metadata is
inspection-only; none can promote a submitted artifact into source authority.

## Guard implication, not a new global extent restriction

The [independent guard/mutation report][guards] checks the unchanged adapter.
On actual production generated-leaf entry, M,N,K are positive and the owning
input/output allocations have checked `MK,KN,MN <= B`, where
`B=min(SIZE_MAX,PTRDIFF_MAX)/sizeof(float)=2^61-1` on the pinned LP64 tuple.
Thus each dimension plus at most 63 fits signed i64. With
`F=tile_n*floor(N/tile_n)`, the verified full/tail coverage and dense accesses
retain representable products, byte offsets and next-IV increments. No MKN
work bound is inferred.

This implication is conditional. Valid zero-area values can contain an extent
`INT64_MAX`, beyond a raw tiled leaf's stepping margin, yet bypass the leaf.
Positive output with K=0 is still extent-checked and allocated before the
positive-zero/zero-reduction bypass. Forced candidate and FP checks remain in
their original order. Huge-empty examples in the guard report are proof
counterexamples to a universal dimension claim, not new executed stress tests.

External source Storage remains all-MAY-alias. Reads snapshot immutable owned
values or use separately proved forwarding; successful output allocation is
fresh and disjoint from live inputs under the conforming allocator contract.
A/B may be the same Value or overlap; no public input-disjointness assertion is
added. Caller object lifetime/capacity/race obligations and allocator/deallocator
FP/effect preservation remain explicit preconditions. Existing source guard,
failure, publication, owning-observation and FP-restoration obligations are not
discharged by the primitive's metadata or by a passing raw-leaf test.

## Retained sanitizer integration counterexample and exact fix

At predecessor `281a346d3cbb1aef2cb5bb96fb29e01d7abf3a31`, the normal source
scope passed 4/4 but sanitized new 3x5 issuance refused before linking and
publication. Pinned Clang emitted the exact leaf-object undefined symbol
`__asan_report_store_n` for unaligned output-lane SLP, absent from the existing
allowlist. The same sanitized installation compiled/executed the old 3x5
request's 68 cases / 111,292 checks. Refusal left no new executable or bundle.
Normal math and private raw-leaf ASan passes had not qualified this source path.

The fix at the frozen head changes only this additional import predicate:
sanitized installation **and** explicit peeled realization **and** exactly
`__asan_report_store_n`. The common old allowlist is unchanged. Compile-time
truth-table assertions reject uninstrumented/old/unknown realizations, fixed
`store8`, `_noabort` and unrelated imports for this exception. The compiler-owned
object must still pass the original ELF/symbol/effect checks; no object import
interface or wildcard sanitizer permission was added. Linked DSO imports alone
would not identify this seam, because unchanged relink-kit code has its own
sanitizer imports; the audit separately records actual leaf-object imports.

The failed issuance and old positive control are preserved at
`/var/tmp/mdslc-source-peel-asan-refusal-VNisCYfH`. The earlier normal full-build
attempt stopped deliberately at 490/497 steps before CTest and is incomplete,
not a full regression pass. The correction's focused receipt preceded final
freeze. The later frozen-head completion used a reconfigure and 16 incremental
build steps, then the complete passing regression below; it is not a pristine
509-step build claim. Historical failure/interruption receipts remain retained.

## Completed local evidence

| Root-run scope | Result | Limit |
| --- | --- | --- |
| Initial old/new mutation tests | 2/2 CTests; old mode 74 checks, peeled mode 158; zero failures/skips; 104.16 seconds | Pre-final-freeze focused controls, not full regression |
| Initial normal new source scope | 4/4 CTests; zero failures/skips; 156.56 seconds | Before the sanitizer-specific correction |
| Initial corrected ASan+UBSan new source scope | 4/4 CTests; zero failures/skips; 171.96 seconds | Before final commit freeze; retained historical focused evidence |
| Frozen local Release regression | 356/356 CTests, every testcase `status=run`; zero failures/disabled/skips; 1453.14 seconds | Complete local x64 suite, not final hosted/merge disposition |
| Frozen producer-inaccessible installation | Test #18 passed in 194.393 seconds | Included in 356, not an extra test; normal x64 installed driver |
| Committed-head ASan+UBSan source scope | 4/4 CTests, every testcase `status=run`; zero failures/disabled/skips; 172.89 seconds | Actual sanitized source-owned DSO evidence on x64 only |

The final ASan reconfigure was followed by `ninja: no work to do` and a new
four-test execution at clean committed head `1ad7fe5...`; it is not a new
from-scratch sanitizer build. The earlier reporter-fix build remains retained.
The four new source tests are also included in the normal 356-test suite;
do not add repeated focused executions to that test count.

Each four-test scope comprises connected execution, refusals, synthetic runner
and typed phase mutations. The committed-head receipt contains 15 synthetic
unittest methods, 39 refused CLI invocations and 158 mutation checks. Synthetic
canonical-emission controls are not upstream-valid semantic IR mutants. The
connected receipt records eleven issued source instances and 79 commands;
successful output alone is not used to infer the requested implementation.

The initial corrected and committed-head sanitized connected receipts contain
the same eleven positive source-oracle summaries:

| Actual source instance(s) | Per-instance cases / checks | Failures |
| --- | ---: | ---: |
| New connected 3x5, 5x3, 5x1; old unit 5x1 reference | 68 / 111,292 each | 0 |
| Same-Value 3x5 and 5x3 | 5 / 480 each | 0 |
| RHS-carry 3x5 and 5x3 | 8 / 256 each | 0 |
| Source-observable end-K 3x5 | 792 / 206,372 | 0 |
| Source-observable end-K 5x3 | 792 / 196,420 | 0 |
| Eligible forwarding 5x3 | 8 / 1,456 | 0 |

Each end-K receipt has overlapping participation full=728, tail-width-one=364,
tail-width-two=364 and 40 explicitly inapplicable short-K witnesses. Those are
not disjoint-case totals, executed passes or skipped CTests. Frozen analytic
strict-order/non-FMA/rounding/subnormal/overflow/Inf/NaN-class/zero-seed witnesses
and lane-distinct last-K-only products are checked through publication and
retained owning observations. Forwarding additionally requires the actual
`readForwarded` call discriminator. Corruption controls must fail for the
intended oracle reason.

The new discriminator independently checks the actual Transform N result handle,
definition-before-use scoped values, full split, tail start/bounds and consumed
A/B/C slices. It is a bounded canonical-emission classifier, not a general MLIR
parser/verifier/importer; the production envelope/replay owns arithmetic legality.
The old complete manifest gate rejects the new schema even when unit-N graph
bytes match. The frozen P1 plan has no unit-N role; no historical P1 bypass is
claimed and its inspector/plan/results remain unchanged.

Typed negative controls use valid closed-issuer donors and exact source-seal,
candidate-byte or compiled-pairing diagnostics in both directions, including
same-source/same-tile old/new and unit-N identities. Precreated stages and
positive rechecks prevent incidental directory-metadata failures from posing
as binding evidence. Manifest prepopulation tests no-clobber ownership, not
serialized-manifest authorization.

## Actual source-owned DSO sanitizer observations

Five recorded deliberate short-B calls in the committed-head receipt report
heap-buffer-overflow in the request-bound DSO: full-N and tail-N for each
nonunit 3x5/5x3, and one unit 5x1
identity call. The nonunit first frames name `__matcore_strict_gemm_f32_v1`;
the unit first frame is **`_mlir_ciface___matcore_strict_gemm_f32_v1`**, the
generated wrapper, not a standalone-leaf-frame observation. Unit N has no tail;
it is not counted as two distinct full/tail detections.

These noninstalled probes invoke the actual local wrapper in the validated
source-owned DSO using real descriptors with M=1, K=2 and B one float short.
They deliberately violate raw private capacity preconditions. Their scope is
full/tail **N**, not both M domains, universal memory safety, normal source
failure behavior or a new runtime callback/import permission.

The normal x64 producer-inaccessible test separately builds from exact frozen
source, relocates the installation, removes disposable producer source/build,
then executes old/new 3x5/5x3 requests, end-K/full/tail and old/new unit-N
identity controls, copied-header forgery refusal, ordered failures/owning
observations and checked forwarding. The pinned test source maintains and
rechecks SDK/issued-bundle hash dictionaries and command logs **in memory** inside
`TemporaryDirectory(prefix="mdslc-closed-installed-")`. It prints its command
JSON on failure, but emits no persistent installed receipt JSON on success;
successful exit removes that temporary installation and its artifacts.

The genuine durable success record is the frozen CTest log/XML: testcase
`package.experimental_regions.closed_source_inaccessible`, `status=run`,
194.393 seconds, and its PASS summary. The summary's wording that hashes were
"retained" describes test-lifetime checking, **not a persisted complete installed
SDK/bundle inventory**. The XML does not retain that hash map or the full
installed command log. Its reported installed driver SHA256 is
`f784e38a967b5d7de16db954570809076f8b9bfc5ddea3984ec183546092685b`.
The durable acceptance interpretation uses the pinned test source and this
actual passing CTest/XML; it cannot reconstruct a standalone installed-artifact
inventory that was not persisted. This is normal installed-source execution
evidence, not sanitized installation acceptance.

The frozen normal connected test separately names the retained
`tests/closed_driver/mdslc-source-peel-vn5wup7v/evidence.json` receipt. It is
`status=pass`, with 73 commands and eleven issued instances, and retains normal
source/artifact bindings. It uses the build-tree driver (`dc326289...`), not the
disposable installed driver above, so it must not substitute for the
producer-inaccessible installed run or imply that run's full inventory survives.

## Completed hosted native ARM scopes

All three downloaded ARM logs were independently recounted from actual passed
records and final summaries, with no skipped/not-run markers:

| Native ARM build schedule | Existing native scope | Private peeled scope | New normal-source scope | Skips |
| --- | ---: | ---: | ---: | ---: |
| [Scalar][arm-scalar] | 130 | 44 | 4 | 0 |
| [Row-contiguous][arm-row] | 132 | 44 | 4 | 0 |
| [Output-tiled][arm-output] | 130 | 44 | 4 | 0 |

The actual passed-record totals are 178, 180 and 178, respectively. Their
checkout is synthetic merge `4d6eb0b3e726d670cd48821eee4e5f92cd1f9214`, whose
parents `[dfc35f9..., 1ad7fe5...]` and tree
`4e784afa700c447fe28b19e55443ed55b2ce140c` were independently verified by the
integration owner through the repository API. This is exactly the frozen local
source tree, not a canonical merge claim.

The four-test source driver is **normal/unsanitized** in these ARM jobs. The
44-test private suite includes generated-ASan raw leaves; that does **not**
qualify a sanitized AArch64 source driver or source-owned ASan DSO. Sanitized
source evidence above is x64-only. The exact store-reporter gate remains
fail-closed for unobserved imports; no all-sanitized-target codegen guarantee
follows from normal ARM execution. These are original-head scopes only; final
current-head whole-check completion is separately recorded below.

## Additional completed original-head hosted x64 scopes

The downloaded Debug PR log has exactly 350 actual passed records, zero skips
and zero failures. Both Release logs have 333 executed passes and 19 retained
AVX512 skips out of 352 selected tests, with zero failures. The Release PR,
Debug PR and ASan PR logs use the same synthetic `4d6eb0b...` checkout/tree as
ARM. Push logs check out exact frozen source `1ad7fe5...`. Their independently
recounted scopes are:

| Hosted scope | Selected tests | Executed passes | Skips | Failures |
| --- | ---: | ---: | ---: | ---: |
| [Release PR][release-pr] full suite | 352 | 333 | 19 | 0 |
| [Release push][release-push] full suite | 352 | 333 | 19 | 0 |
| [Debug PR][debug-pr] full suite | 350 | 350 | 0 | 0 |
| [ASan push][asan-push], retained baseline | 236 | 217 | 19 | 0 |
| ASan push, separate SPIR-V | 2 | 2 | 0 | 0 |
| ASan push, separate private peeled family | 44 | 44 | 0 | 0 |
| ASan push, separate baseline source-peel gate | 4 | 4 | 0 | 0 |
| [ASan PR][asan-pr], retained baseline | 236 | 217 | 19 | 0 |
| ASan PR, separate SPIR-V | 2 | 2 | 0 | 0 |
| ASan PR, separate private peeled family | 44 | 44 | 0 | 0 |
| ASan PR, separate baseline source-peel gate | 4 | 4 | 0 | 0 |

Each OpenBLAS-required Release/ASan scope listed above has 19 skipped hardware
controls: one packed-AVX512 test,
13 strict-ISA AVX512F tests and five old AVX512F output-pattern tests. They are
not executed passes or physical AVX512 qualification. The ASan retained-baseline
CTest summary says zero failures out of 236 selected tests, **not 236 executed
passes**. Across
the four ASan scopes in each job, 267 tests executed successfully and 19 were
skipped. The new baseline source gate independently passed 4/4 with zero skips:
369.03 seconds on push and 388.72 seconds on PR. Unrelated retained ISA skips
neither qualify it nor count as its skips. There is no blanket zero-hosted-skips
claim, and the Release summaries are not 352 executed passes.

### Preserved cancellations and CI-only budget correction

The original two PR-native jobs remain `CANCELLED`, not `SUCCESS`:

- [OpenBLAS OFF, Matcore MLIR scalar][cancelled-scalar];
- [OpenBLAS OFF, parameterized output pattern][cancelled-output].

The retained job JSON and annotations identify the explicit one-hour limit and
the cancelled step `Prove opt-in production build without test targets`. Their
full CTest scopes had finished, but final production/package proof had not:

| Original job scope | Selected full-suite tests | Executed passes | Hardware skips | Full-test seconds | Whole job |
| --- | ---: | ---: | ---: | ---: | --- |
| PR OpenBLAS OFF / scalar | 353 | 334 | 19 | 2976.49 | Cancelled |
| PR OpenBLAS OFF / output pattern | 348 | 329 | 19 | 2966.27 | Cancelled |
| Matching push OpenBLAS OFF / scalar | 353 | 353 | 0 | 2164.62 | Success |
| Matching push OpenBLAS OFF / output pattern | 348 | 329 | 19 | 2942.32 | Success |

The scalar logs also contain a separate one-test preflight, not an extra case
in the full-suite counts above. The differing skip counts are retained runner
hardware observations, not normalized into a blanket hosted zero-skip claim.
Receipts are under `build-source-peel-release/ci-pr105-timeout-review.AC8jnP`.

The [reviewed budget-only correction][budget-fix] raises Release jobs from 60
to 90 minutes; matrix, two-job build limit, commands, CTest selection, individual
test timeouts and the final production proof are unchanged. This first correction
left Debug/sanitizer/ARM/Windows budgets unchanged; the separately reviewed
Debug correction below follows it. Git comparison confirms its only changed paths
are `.github/workflows/mdslc-native.yml` and that new CI report, with parent
`1ad7fe5...`. This is not a compiler repair, benchmark result or proof that the
new budget will suffice. The [integration comment][budget-comment] preserves
the original cancellations and requires fresh exact-head hosted evidence.

Independent Git/API inspection confirms synthetic merge
`411335b028c6d6b2f59d7234c991f006b4419555` has parents
`1f7c491f230251d04d42b5d4a0af5eaf29bcb685` and
`d3460226d7f9358ac815e6f49c73a7a9cff8bbea`. Its whole tree
`6544096242a8e69be94c2364ae4225814ebb0244` equals the CI-only head's tree.
Its `compiler/` subtree `07c0b72c91d837777f933484f3ea0a905ea563ad` equals
both local `1ad7fe5...:compiler` and `d346022...:compiler`. The whole-tree
`1ad7fe5...` to `d346022...` comparison changes only the workflow and added
budget report named above. This is tracked source identity, **not binary
equality, reproducible-build proof or new local artifact qualification**.

No cancelled job is relabeled passing, silently omitted, or used to weaken the
scope. These were not source-code/contract counterexamples, but their original
acceptance gates were incomplete. The subsequent `d346022...` result was 24
successful checks and two cancelled Debug jobs, not complete qualification.
The later `cf154de...` CI passed all 26 checks before the normal merge recorded
above. Earlier cancellations remain cancelled, not retroactively passing.

### Completed d346 Windows and TSan regression scopes

Representative fresh PR logs are retained under `build-source-peel-release/`.
Both check out synthetic merge `411335b028c6d6b2f59d7234c991f006b4419555`,
whose checkout description merges `d346022...` into main `1f7c491...`; the job
API binds head `d346022...`. The local 356-test, installation and x64 source-ASan
evidence above remains attributed to frozen product source `1ad7fe5...`, not
relabeled as a new local build.

| Representative completed PR scope | Selected CTests | Executed passes | Skips | Failures | Test seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Windows x64][windows-d346-pr], Release | 52 | 52 | 0 | 0 | 159.27 |
| Same Windows job, focused Debug runtime | 33 | 33 | 0 | 0 | 21.95 |
| Same Windows job, focused AddressSanitizer runtime | 1 | 1 | 0 | 0 | 0.56 |
| [Linux x64 ThreadSanitizer][tsan-d346-pr], focused runtime | 4 | 4 | 0 | 0 | 11.40 |

The counts come from actual `Passed` records and the matching CTest summaries,
not just green job conclusions. The Windows job is run `42751626459`, job
`129616336758`, using clang-cl 21.1.8. Its sanitizer is **AddressSanitizer**, not
UBSan or TSan: `/fsanitize=address`, the selected x64 dynamic ASan runtime and
its import/thunk libraries. The separate ASan native-pipeline script also
reports `Windows native pipeline PASS` outside the one-test CTest scope.
That path instruments runtime/generated translation units while retaining
uninstrumented Release compiler tools for orchestration. These are preserved
Windows regression/package checks, **not Windows source-peel support** or
sanitization of the complete compiler. The source-peel contract remains Linux
x64/AArch64.

The TSan job is run `42751626689`, job `129616340060`. Its Debug configuration
uses `-fsanitize=thread` and `TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1`.
Exactly `runtime.cpu.execution_context.v1`, `runtime.cpu.parallel_packed.v1`,
`runtime.cpu.planner_v3_resources` and `runtime.c_abi.public_context_v1` execute.
Native frontend, Matcore MLIR and OpenBLAS are disabled. This is bounded runtime
ThreadSanitizer evidence, not source-peel/leaf TSan coverage or a universal
race-freedom claim. The matching [Windows push][windows-d346-push] and
[TSan push][tsan-d346-push] job APIs also report success at `d346022...`; their
logs were not independently recounted here and do not supply the table's counts.

An initial inspection of [new-head ARM run `42751626656`][arm-d346-run] found
scalar/row successful and output-tiled in progress. After all three completed,
the separately authorized download/recount below replaced that pending-run
disposition. That intermediate observation did not clear a whole-PR gate; final
current-head completion is separately recorded below.

### Completed d346 native ARM normal-source scopes

All three fresh jobs in run `42751626656` now report success at `d346022...`.
Their retained logs independently show actual native `aarch64` execution,
the same synthetic checkout `411335b028c6d6b2f59d7234c991f006b4419555`, and:

| Fresh ARM schedule | Retained baseline passes | Separate private family passes | Separate source-peel passes | Executed total | CTest skips/failures | Source-gate seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| [scalar, job `129616337245`][arm-d346-scalar] | 130 | 44 | 4 | 178 | 0 / 0 | 202.38 |
| [row-contiguous, job `129616337107`][arm-d346-row] | 132 | 44 | 4 | 180 | 0 / 0 | 201.84 |
| [output-tiled, job `129616336923`][arm-d346-output] | 130 | 44 | 4 | 178 | 0 / 0 | 202.52 |

Each four-test source scope is exactly `connected`, `refusals`, `runner` and
`phase_mutations`. These new-head scopes do not add cases to or relabel the
original-head ARM receipts. The 44 private tests include 28 explicitly named
ASan controls, but the source driver is normal/unsanitized. Thus this is normal
source-peel AArch64 qualification under three producer schedules plus separate
private generated-leaf sanitizer coverage, **not sanitized AArch64 source-driver
or source-owned-DSO qualification**. The exact import gate still refuses
unobserved sanitized target imports. All source-ASan evidence in this record
remains x64-only. These are completed `d346022...` scopes, not final-head ARM
passes; completed qualified-head scopes appear separately below. These receipts
predate the normal merge recorded above.

### Completed d346 x64 ASan+UBSan scope

The [fresh ASan PR job][asan-d346-pr], run `42751626689` / job `129616339679`,
checks out the same synthetic merge `411335b...` and binds API head `d346022...`.
Independent actual-record recount of `hosted105-d346-asan-pr.log` gives:

| Fresh ASan PR scope | Selected CTests | Executed passes | Skips | Failures | Test seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained baseline | 236 | 236 | 0 | 0 | 1341.54 |
| Separate SPIR-V | 2 | 2 | 0 | 0 | 0.37 |
| Separate private peeled family | 44 | 44 | 0 | 0 | 4.36 |
| Separate source-peel gate | 4 | 4 | 0 | 0 | 227.70 |

All 286 selected CTests execute and pass here. This fresh baseline has no skip
records; it does **not** inherit the 19 AVX512 hardware skips observed in the
original-head ASan jobs, nor retroactively execute those earlier skipped tests.
The differing runner observations are retained, not normalized into a global
zero-hosted-skips claim.

The native host configuration uses `-fsanitize=address,undefined` for C/C++ and
linking, and the job authenticates ASan+UBSan linkage. Execution uses
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`
and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. The four source tests are
`connected`, `refusals`, `runner` and `phase_mutations`; private generated-leaf
sanitizer controls remain their separate scope. This x64 host/native sanitizer
configuration is not TSan, universal generated-IR UBSan coverage or sanitized
AArch64 source qualification. The `d346022...` Debug cancellations below leave
that old head incomplete; its green sanitizer job does not qualify current
`cf154de...` by inheritance.

### d346 Debug deadlines: reviewed diagnosis, incomplete gates retained

An intermediate query at `d346022...` recorded 22 successful checks, two
in-progress checks and two cancellations; the completed rollup is 24 successful
checks and those same two cancellations. Both [Debug PR][debug-d346-pr]
job `129616340141` and [Debug push][debug-d346-push] job `129616322713`
cancelled in `Test exact-MLIR Debug validation`, rather than a production-proof step.
The previously adopted Release-only budget increase did not change Debug's
60-minute job limit. Both retained annotations explicitly identify the global
one-hour deadline.

The retained root-downloaded `hosted105-d346-debug-pr-timeout.log` contains
**346 actual passed records out of 350 selected tests**, zero recorded skips or
completed failures. Its last started test is
`driver.source_pattern_peel.phase_mutations`; that test has no completion record
before `The operation was canceled`. Four selected tests therefore have no
recorded passing completion. This is a partial prefix, **not 350/350, a complete
four-test source gate or a successful Debug suite**. Independent raw-log recount
of the push finds 343 completed records: 324 passes plus 19 AVX512 skips, with
peeled `connected` still running. Original-head Debug's complete 350/350 receipt
remains historical evidence, not a replacement new-head pass.

The two independently reviewed diagnoses are retained in
`ci-debug-review.EAEHLq` and `ci-debug-diagnosis-QAwoGf`, under
`build-source-peel-release/`, and summarized in the [committed Debug budget
report][debug-budget-fix]. The PR cancellation was 75.452 seconds into
`phase_mutations`, whose earlier passing durations were 76.46/80.68 seconds.
The push cancellation was 24.684 seconds into `connected`, whose earlier passing
durations were 108.18/112.53 seconds; the current PR had completed it in 185.13
seconds. Both diagnoses found distributed completed-test slowdown and no observed
individual-test timeout or failed-test diagnostic. This supports global budget
exhaustion rather than an observed hung-test failure; it does **not** prove an
unfinished test would terminate or identify a hardware cause.

Reviewed commit `ee4585cbee981a5cd1c47eb4ae1874eb563397a9` changes only the
Debug job's timeout/comment to 90 minutes and adds its report. Current
`cf154de...` adds only the historical Release-report clarifier. Configure/build
commands, test selections, individual timeouts and two-job build/serial CTest
limits remain unchanged. These corrections do not clear an old cancellation;
fresh current-head completion was still required and is recorded below, not
inferred from this diagnosis.

### Completed d346 required Release PR scope

The earlier in-progress required Release PR job `129616340143` subsequently
completed successfully in run `42751626689`. Its uniquely retained
`hosted105-d346-release-pr-129616340143.log` records **352 selected tests:
333 executed passes, 19 explicit AVX512 skips, zero failures**, in 3017.12
seconds. The recount includes the no-whitespace `***Skipped` spelling. The
skipped controls are the same one packed-AVX512, 13 strict-ISA AVX512F and five
old AVX512F output-pattern tests, not executed passes.

This OpenBLAS-required role intentionally skips the conditional production-proof
step; it does not supply a producer-without-tests proof from this job. All
Release jobs completed under their unchanged role-specific selections at the
90-minute budget, but that historical completion neither promotes the Debug
prefixes nor substitutes for current-head CI. The representative log is bound
to synthetic `411335b...` and old head `d346022...` only.

### Qualified cf154de: all hosted checks successful before normal merge

At qualification, PR head `cf154deacea2bddb3240660c1af27ff4ffbc4042` had synthetic checkout
`9d30589bbdd9624cdfef63c6c2107488e35537c8`, with parents `1f7c491...` and
that exact head. Both complete trees equal
`0330fb5748277b3fe8f6b89f13986173c3d10674`; local Git and retained commit JSON
agree. Its compiler subtree remains `07c0b72c91d837777f933484f3ea0a905ea563ad`,
identical to frozen local `1ad7fe5...`. The entire `1ad7fe5...` to current-head
diff is the workflow and two budget reports, not compiler/test/runtime/header
changes or evidence of binary equality.

The independent receipt
`hosted105-final-pr-scopes-P4TyCK/scope-recount.json` and actual retained raw
logs were rechecked separately from all older-head evidence:

| Current-head PR scope | Selected CTests | Executed passes | Skips/failures | Test seconds |
| --- | ---: | ---: | --- | ---: |
| [Windows `129638646575`][windows-cf-pr], Release | 52 | 52 | 0 / 0 | 144.58 |
| Same job, focused Debug runtime | 33 | 33 | 0 / 0 | 21.82 |
| Same job, focused AddressSanitizer runtime | 1 | 1 | 0 / 0 | 0.03 |
| [TSan `129638645432`][tsan-cf-pr], focused runtime | 4 | 4 | 0 / 0 | 11.29 |

Windows uses clang-cl x64 ASan for its focused sanitizer scope; TSan executes
the same four exact runtime tests listed above, not source-peel code. These
are the final-head **PR-side** scopes only, not unreviewed push-side results.
Windows package/relocation/artifact success is separate from CTest counts and
does not add Windows source-peel support.

The final retained `hosted105-cf-final-status.json` binds exactly this head and
contains **26 `COMPLETED` / `SUCCESS` check records**, without a pending, failed,
cancelled or skipped check. That snapshot was captured just before marking the
PR ready (`isDraft=true` at capture); a subsequent pre-merge query confirmed the
same 26 successful checks, `isDraft=false`, state `OPEN` and no merge commit. These
whole-check conclusions are distinct from CTest hardware skips below.

The separate `selected-final.json` receipt joins six additional successful PR
jobs to their actual raw log hashes, `9d30589...` checkout and exact complete
tree `0330fb57...`. Independent raw-record recount gives:

| Final selected PR scope | Selected CTests | Executed passes | CTest skips | Failures |
| --- | ---: | ---: | ---: | ---: |
| [Required Release `129638645574`][release-cf-pr] | 352 | 333 | 19 | 0 |
| [Debug `129638645152`][debug-cf-pr] | 350 | 331 | 19 | 0 |
| [ASan+UBSan `129638645570`][asan-cf-pr], retained baseline | 236 | 217 | 19 | 0 |
| Same ASan job, separate SPIR-V | 2 | 2 | 0 | 0 |
| Same ASan job, separate private peeled family | 44 | 44 | 0 | 0 |
| Same ASan job, separate source-peel gate | 4 | 4 | 0 | 0 |
| [ARM scalar `129638646125`][arm-cf-scalar], 130 + 44 + 4 scopes | 178 | 178 | 0 | 0 |
| [ARM row `129638645848`][arm-cf-row], 132 + 44 + 4 scopes | 180 | 180 | 0 | 0 |
| [ARM output `129638646080`][arm-cf-output], 130 + 44 + 4 scopes | 178 | 178 | 0 | 0 |

Release/Debug full-suite times are 3009.70/3508.32 seconds. ASan's four scopes
take 2211.64/0.53/6.53/382.49 seconds. All four source-peel tests execute and
pass in the Release and Debug totals already shown, not as additional tests.
The same four source tests pass separately under ASan and each ARM schedule.
ARM source-gate times are 202.83/199.89/203.70 seconds. This completes the
previously missing Debug gate at the actual current head, rather than promoting
either cancelled prefix.

The 19 skips in each listed x64 baseline are the retained packed/strict-ISA/old
output-pattern AVX512 controls, not new baseline source-peel skips or executed
passes. ARM's private 44-test family includes generated-ASan controls; its
source driver remains normal/unsanitized. Source-owned sanitized DSO evidence
remains x64-only; no sanitized AArch64 source-driver, Windows source-peel,
universal ISA or performance qualification follows.

The required Release role's conditional production-proof step is explicitly
`SKIPPED`, as its unchanged role selection requires. That job is not claimed as
the producer-without-tests proof. Other Release jobs have whole-job `SUCCESS`
records in the final rollup; their per-scope execution counts were not recounted
here. Likewise, no CTest execution/skip totals are invented for the other
unrecounted successful checks. Completed qualification was separate from merging:
main was still the private PR103 checkpoint at that collection-time snapshot;
the subsequent actual canonical source merge is recorded at the top.

## Independent preservation receipt

`/var/tmp/mdslc-source-peel-preservation-53XNFY/audit.json` binds clean frozen
source `1ad7fe5...`, stable regular-file reads and retained root-run evidence:

- All 84 canonical CPU and 84 old unpeeled-pattern LLVM/companion stage/manifest
  files match the private checkpoint byte-for-byte.
- Of 98 private peeled files, the 84 nonmanifest LLVM/MLIR/Transform files are
  unchanged; 14 manifests change only `pattern_source_guard` and
  `pattern_source_selection` to primitive-issuer/non-source-authority wording.
- Actual private normal 5x3 LLVM still equals frozen P3, SHA256
  `7864ea0110a3567ad29c0f27d8a64369f05fb4f440625d457a4712b8021d3496`.
- All retained connected source instances still match recorded manifests and
  file hashes; actual nonunit leaf objects contain the new exact reporter,
  while the old unit reference object does not.

The audit's source-ASan artifact join covers the initial corrected connected
receipt `mdslc-source-peel-jce6ewzu`; the later committed-head rerun is separate
evidence, with its own connected/refusal receipt hashes below.

No cross-build object equality, recursive scratch identity, source/runtime
qualification, native ARM support or performance conclusion follows from
preservation alone. The audit reran no compiler, test or generated executable.

## Compact qualification receipt archive

Canonical local archive, under `/home/hamza-usta/MatcoreDSL`:
`benchmark_reports/source-output-peel-qualification-2026-11-10-v1/receipts.tar.gz`,
SHA256 `4c522bcb6a356e02aae7abe257ed10b7406bac3e3a800eb63be334a505a8786c`.
It is 959,776 compressed bytes / 8,519,680 uncompressed tar bytes: 77 receipt
files, 8,363,050 payload bytes, plus the manifest (78 regular members total).
The embedded manifest exactly matches v2, SHA256
`eac900c87187b2faadd1511594897019ac411a8aa5015c77bba741b2c2b1f4d1`.
An independent streamed read verified the archive digest, every receipt size/hash
and exact manifest bytes without filesystem extraction. Creation rechecked and
retained the originals; no originals were deleted.

The selection preserves frozen local/normal connected and committed-ASan records,
the installed acceptance test source, preservation audit, final eight selected
PR logs/receipts and rollup, sanitizer refusal/control, and CI-budget negative
evidence/diagnoses. It excludes binaries, SDK/generated bundle trees and duplicate
job-log copies. This preserves existing recorded evidence, not the installed
test's unpersisted temporary inventory, a re-executable build, new authority or
P5 collection results.

## Evidence identities and remaining boundary

The normal receipts below are under `build-source-peel-release`; corrected
sanitizer receipts are under sibling `build-source-peel-asan`, both in
`/home/hamza-usta/mdslc-work/correctness-campaign-v1`.

| Receipt | SHA256 |
| --- | --- |
| Normal `first-mutations.log` | `5e21c49b910bb66982249fbca8e1d011c94da80dff1bf4db927cfa6f99d2287c` |
| Normal `first-mutations.xml` | `5d38fbf581182046a93560d88d7621e548c598300c1c5ae3f5382ab2e6a40762` |
| Normal `first-connected-focus.log` | `3e7465aa1e39c9a4de08180c9ce9a997f120ff6534a48256bf619cae25e95c5c` |
| Normal `first-connected-focus.xml` | `a713ae3e9d4ba1b21e4550ca069acb706dfaf52032c69209d06975e888ec8130` |
| Normal frozen `frozen-configure.log` | `b93adfa485918f01dbf231483c3dc83223b86bf151d3d299a0189013bcadf3ca` |
| Normal incremental completion `frozen-full-build.log` | `28fab70f15b3b13a4f2d287e5e34944da340b535c1bab18753b2a9486db4311e` |
| Normal frozen `frozen-full-ctest.log` | `29ab4daefc2ac3497940d74f338976ea93846418fca611dcfd4b3320a009c243` |
| Normal frozen `frozen-full-ctest.xml` | `481b101a2f6474cf355621d75fa0a6efef4a30384d543d12dedc9e6a3b2bc154` |
| Normal frozen connected `tests/closed_driver/mdslc-source-peel-vn5wup7v/evidence.json` | `01f3f5e777a46dba0142becd0f4379d58bb357a2afa95b08f7947a6c4c640c92` |
| Frozen acceptance test `source-peel/compiler/tests/closed_driver/installed_source_inaccessible.py` | `735ac83cc870420cfec187ab63b4fb0a4342d0ee300f75800425461c3086d226` |
| ASan `store-report-fix-build.log` | `6e5023f1198d3c963ed7987dbb73e2b0bf96d5d060ce2282978e36e5807e58f5` |
| ASan `store-report-fix-focus.log` | `a33d938f92a71c5c82b43e7416b42bb464ead3bec9a8c59dcad2881aa2aa9106` |
| ASan `store-report-fix-focus.xml` | `dc5e45650dbe4d5082902423f9207a583e93c96bd84db4cd7f7da67d1b679d65` |
| ASan connected `tests/closed_driver/mdslc-source-peel-jce6ewzu/evidence.json` | `25e8e1239f533a00cd58c619a9a32741a3a420e3a5d3fe0bd5694a5ed49fe91f` |
| Committed-head ASan `committed-configure.log` | `a18b9e89767219c663d2d6dd6359ab45dba96bc2520b64a5bd389e79c3aca928` |
| Committed-head ASan `committed-focused-build.log` | `8c35b1682f73592d30612d7b0fabdb384fcfff35cb5a076cb48b85b1368aa794` |
| Committed-head ASan `committed-focused-ctest.log` | `876037c06a3d36fc7aa2be6b674e27b860a8b91515ad835aa47055141d53f519` |
| Committed-head ASan `committed-focused-ctest.xml` | `4f6bdaa585f78a8c4144bfbb1532cb0fe8d7297f526274516d6c0bf0d4ef940a` |
| Committed-head ASan connected `tests/closed_driver/mdslc-source-peel-_cprdr0o/evidence.json` | `084d8c9a27cf4e64f90a085406f01b611a410a1b4e074a12976526e788fc4f98` |
| Committed-head ASan refusals `tests/closed_driver/mdslc-source-peel-njx4eiam/evidence.json` | `fa07eb80c4a7efe9b2a9a698b6225469141709afc6a0696ba7b0f5eceb1bc057` |
| Normal ARM `hosted105-arm-scalar.log` | `7f0a8994791acc1d972332c765f4b33d294a83144ee56aa73e7d8577137f96fb` |
| Normal ARM `hosted105-arm-row.log` | `171e1d33a81a8e6b32dfb895b9673aa5b45b763912cb73348ce15ed5462923eb` |
| Normal ARM `hosted105-arm-output.log` | `4f0aadf38fff821b78e09edbc375b7dce38a3239a8833608a4ac94483783963d` |
| Hosted Debug PR `hosted105-debug-pr.log` | `be6f4899017b3a8345fb4ef7bb73c0232097b584f9f17503684c103be3cc507e` |
| Hosted ASan push `hosted105-asan-push.log` | `bbf8b01f133d51044699e0d08e3e03f4c9403c7184c2fa56a68809a1bc4fccb5` |
| Hosted Release PR `hosted105-release-pr.log` | `45a957a407028ea27d6339c3c29d4ae5f830481fa7e5b41b1275e52f0ab4de95` |
| Hosted Release push `hosted105-release-push.log` | `b735162a857ca25a646be602b397fe889d917558b109f14ed3fb12d83b011adc` |
| Hosted ASan PR `hosted105-asan-pr.log` | `5e6e7dd248753f81090dbeb7998e95b46ae18ff8212cac4d3aa12824b75e947d` |
| CI-only-head Windows PR `hosted105-d346-windows-pr.log` | `eb8349d12d105776093303155cc02eda8b0dc3db22ffcf798c2045f98ea6977c` |
| CI-only-head TSan PR `hosted105-d346-tsan-pr.log` | `d65be39c6d99769cd71ac6e4c01d7fcb66a574012e48f7147fd8aed25f1f4249` |
| CI-only-head ASan PR `hosted105-d346-asan-pr.log` | `aefc511e09f2b318995a07d177902214fab43b5952bef1ae56cf3f7569695a49` |
| d346 required Release PR `hosted105-d346-release-pr-129616340143.log` | `db3f199183e41cba3e20219d8d7d78808092fe923f5ab0488b574a5d6a57f17f` |
| CI-only-head Debug PR partial `hosted105-d346-debug-pr-timeout.log` | `c607178a04da11590acbe0d7f174ab31063e7305c4fbd31fc8ab3f899a4ec9b0` |
| d346 Debug push partial `ci-debug-review.EAEHLq/push-debug.log` | `e9f7d6b0cf4f03dbf1baca7632740570e86e1ee496c41dba42fc0af7e4a69f8b` |
| Debug independent eleven-file `ci-debug-review.EAEHLq/SHA256SUMS` | `585fe5d9f4ba57f301205a6defc214b9c80fe1402af66ba02a4b5b7e01e99982` |
| Opposing Debug diagnosis `ci-debug-diagnosis-QAwoGf/diagnosis.md` | `13afc309bdd56361771deade7375cd0dff36f3ddbadb8340a23376b5a97842ad` |
| Current-head scopes `hosted105-final-pr-scopes-P4TyCK/scope-recount.json` | `f7d41ed4e305d7163a474972d0e8e7f220ca851635879ff76c757f7fd58605eb` |
| Final 26-check rollup `hosted105-cf-final-status.json` | `7f3b25e0a956c4ff906235844a24055d34c19e55caa0e4395f5c9e0c8b5f1b4f` |
| Final selected six-job receipt `hosted105-final-pr-scopes-P4TyCK/selected-final.json` | `a3979ccea4884334b721bde782423e7c40bc6490d18a9768013d08f317a594b9` |
| Current-head Release PR `hosted105-final-pr-scopes-P4TyCK/129638645574.log` | `24233540fb7d417c2d8069e9bc9a8e83dbf7a015e43c6f3a54e70a87b2214707` |
| Current-head Debug PR `hosted105-final-pr-scopes-P4TyCK/129638645152.log` | `2cf51a4b9c1e0c82a789e6e09d3ce4a9fc63db6f752b0f78f1761a18fb49d043` |
| Current-head ASan PR `hosted105-final-pr-scopes-P4TyCK/129638645570.log` | `9c579b19b362d833ee20e6c7ec0d973c86c0e41af5bb7708a9e43d8261e55ed9` |
| Current-head ARM scalar `hosted105-final-pr-scopes-P4TyCK/129638646125.log` | `20952d633cfc53b9a5de9996091307d5aba419e17d2dacf7f3c4b35dce8d1e37` |
| Current-head ARM row `hosted105-final-pr-scopes-P4TyCK/129638645848.log` | `419eee1a49706d8220770acd0b6ac0f75b1443872cc93b2531c6e59f55347385` |
| Current-head ARM output `hosted105-final-pr-scopes-P4TyCK/129638646080.log` | `9f3386c916d9ecd90c95a2990f7ea58d000f128dbbd19ca8c45d8ccf1146a11c` |
| Current-head Windows PR `hosted105-final-pr-scopes-P4TyCK/129638646575.log` | `7132a5d7032c29cbac9e8319288388be3e2eec891c6e320a59dfab0edb01461c` |
| Current-head TSan PR `hosted105-final-pr-scopes-P4TyCK/129638645432.log` | `c9ded55e6989e11a97af22de5e7f2a2deb823ec2a6e90cbf536be5c1828176e3` |
| CI-only-head ARM scalar `hosted105-d346-arm-scalar.log` | `9af366de508f0a54922f1d0a4d3d4233fa6e81107b8f9badf21e16f241779b42` |
| CI-only-head ARM row `hosted105-d346-arm-row.log` | `24b18588f5ada4f873adbc9493516a60e39367646840f00e8d324d23799816c0` |
| CI-only-head ARM output `hosted105-d346-arm-output.log` | `e41ec8d0984ceefd0e20262afd8c4ad01b71ff2e47785a0f7c52497a1a23d6e7` |
| Cancelled PR scalar `ci-pr105-timeout-review.AC8jnP/job-129591398741.log` | `86e328c7bb5518484e7001756ff1c4c32a95dab03a81dce33c4eae6a07433eb5` |
| Cancelled PR output `ci-pr105-timeout-review.AC8jnP/job-129591399005.log` | `f92e070edea510ebb25430e5071346f79c44431f6719ab3efde9df9978886e2c` |
| Successful push scalar `ci-pr105-timeout-review.AC8jnP/job-129591367297.log` | `627f1e91ac44272f7b090b7dac2e760535ae3326bd41a96509a93bb270cb2fb9` |
| Successful push output `ci-pr105-timeout-review.AC8jnP/job-129591367448.log` | `bd16e567b1ff3917aac7ae0e8a6f931e60cebe08cdbdea483da4df07f25dd59c` |
| `/var/tmp/mdslc-source-peel-preservation-53XNFY/audit.json` | `4ad629ca34bbe7e5255750452da01942109b10d5ed201c4ca2dcfd8ac677c735` |
| `/var/tmp/mdslc-source-peel-preservation-53XNFY/audit.md` | `9271d7e101de1106fe41f874392629e15cb6070a432f953168a590ae24e1eabf` |
| Retained refusal `new-compile.stderr` in `/var/tmp/mdslc-source-peel-asan-refusal-VNisCYfH` | `83389b9d52f867b8cabc1ccc69bffd6baaeaa209a2a58baa1f8eb15bc3e9203c` |
| Retained old control `old-execute.stdout` in the same directory | `2a6ac559616ca1006b86c6bf4ae055ca4df6621f3009055df7cb8759faf76ecb` |

The intended source contract remains single-TU native-baseline Linux x64/AArch64
complete executables with `none` or separately checked forwarding. Local frozen
regression/normal installation, local x64 source ASan and native ARM normal-source
scopes and current-head hosted qualification are complete as bounded above.
The subsequent normal merge is recorded above; it was not a P5 collection
prerequisite. No sanitized-source
AArch64 qualification is claimed.
Other ISAs, GPU mappings, multi-TU,
post-publication relocation, automatic selection and general executable imports
remain excluded. Private PR103 counts are not added to these source counts.

## Source/contract review disposition, separate from CI clearance

No unresolved code or contract blocker was found within the stated frozen
envelope. The rationale is the combined structural, ownership and execution
evidence, not a finite numerical example alone:

- The same independently checked strict full/tail primitive is selected through
  a closed source request; original WHAT and all old/default routes stay intact.
- Existing source/entry, artifact bytes, local DSO ownership and compiler-produced
  publication seals are reused; exact-diagnostic typed swaps exercise those
  boundaries rather than incidental metadata failures.
- The guard implication is conditional at real leaf entry, retains caller and
  allocator preconditions, and does not impose false huge-empty limits.
- The sanitizer reporter correction is exact and realization/profile-scoped;
  observed source-owned full/tail detections, strict/effect oracles, normal
  producer-inaccessible installation and normal native ARM complement the proof.

This disposition is not arbitrary-IR verification, all-parameter execution,
sanitized-source ARM qualification or a source grammar/runtime expansion.
The preserved cancellations remain historical failures to complete those heads'
gates, not passes. The independently reviewed budget corrections are now followed
by actual complete current-head hosted checks, including the final Debug scope.
This source/contract and CI disposition does not perform or authorize a normal
merge or a timing campaign. The unchanged [P5 protocol][p5-protocol] requires
completed source/native/installed/relevant-hosted qualification at an immutable
checkpoint and separate explicit coordination, **not normal merge**. Source
`cf154de...` and harness `ea7accfe94fbb05a79c5135f074e25390b0e147d` had completed
their 26-check gates and independent reviews before the owner-authorized fixed
collection. Conducting that separately authorized collection while unmerged
therefore did not violate the protocol. This corrects this report's previously
introduced extra merge-before-collection gate; the frozen P5 protocol did not
change. No old prefix, whole-check metadata or benchmark preparation replaces an
executed scoped qualification.

The separately preregistered old/peeled/stock comparison accounts for strict
numerics, code size/inlining and call/allocation lifetime. Its collection and
results are outside this qualification report's audit scope; no P5 timing result
is assessed here. P1 remains `INCONCLUSIVE_TIMING` with
its bounded loss retained; no thresholds, automatic winner policy or revised P1
plan are authorized by this record.

[contract]: https://github.com/onlyxItachi/MatcoreDSL/blob/1ad7fe56b5fcb10efe37e523206228e9ef303f6a/docs/mdslc/SOURCE_TIME_OUTPUT_PEEL_V1.md
[guards]: https://github.com/onlyxItachi/MatcoreDSL/blob/1ad7fe56b5fcb10efe37e523206228e9ef303f6a/docs/mdslc/agent-reports/source-output-peel-guards-mutations-v1.md
[arm-scalar]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932194/job/129591398127
[arm-row]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932194/job/129591398007
[arm-output]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932194/job/129591397658
[asan-push]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743923523/job/129591367354
[asan-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932236/job/129591398609
[release-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932236/job/129591398625
[release-push]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743923523/job/129591367199
[debug-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932236/job/129591398314
[cancelled-scalar]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932236/job/129591398741
[cancelled-output]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932236/job/129591399005
[budget-fix]: https://github.com/onlyxItachi/MatcoreDSL/blob/d3460226d7f9358ac815e6f49c73a7a9cff8bbea/docs/mdslc/agent-reports/source-peel-release-ci-budget-v1.md
[budget-comment]: https://github.com/onlyxItachi/MatcoreDSL/pull/105#issuecomment-6758400143
[windows-d346-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626459/job/129616336758
[windows-d346-push]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751620853/job/129616318172
[tsan-d346-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626689/job/129616340060
[tsan-d346-push]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751621276/job/129616322385
[arm-d346-run]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626656
[arm-d346-scalar]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626656/job/129616337245
[arm-d346-row]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626656/job/129616337107
[arm-d346-output]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626656/job/129616336923
[asan-d346-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626689/job/129616339679
[debug-d346-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751626689/job/129616340141
[debug-d346-push]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42751621276/job/129616322713
[debug-budget-fix]: https://github.com/onlyxItachi/MatcoreDSL/blob/cf154deacea2bddb3240660c1af27ff4ffbc4042/docs/mdslc/agent-reports/source-peel-debug-ci-budget-v1.md
[windows-cf-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42758581654/job/129638646575
[tsan-cf-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42758581664/job/129638645432
[release-cf-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42758581664/job/129638645574
[debug-cf-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42758581664/job/129638645152
[asan-cf-pr]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42758581664/job/129638645570
[arm-cf-scalar]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42758581844/job/129638646125
[arm-cf-row]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42758581844/job/129638645848
[arm-cf-output]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42758581844/job/129638646080
[p5-protocol]: https://github.com/onlyxItachi/MatcoreDSL/blob/ea7accfe94fbb05a79c5135f074e25390b0e147d/compiler/benchmarks/source_pattern_peel/README.md
