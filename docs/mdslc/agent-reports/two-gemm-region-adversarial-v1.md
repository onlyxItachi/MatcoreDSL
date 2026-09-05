# Independent adversarial review: ordered two-GEMM region

Date: 2026-09-05. Canonical base:
`5f455bacde0959983b2b888f15fd5dabd4b1ceaa`.
Reviewed production checkpoint:
`de204b5bc288fe90865284b30661aeffadd132d4`.
Final reviewed test checkpoint:
`f8f15eb3fad50af81e07e2deecb25b024975c572`.
The latter adds only the seven-line fresh-context round-trip test.

The reviewer worked independently of both implementation lanes, without
production edits or duplicate builds. No unresolved production correctness
blocker was found **within the explicitly inspection-only admission envelope**.
This is not approval for generated execution, region bufferization, generic
dispatch, or merging without the integration workflow's remaining gates.

## Coverage

The review covered:

- Native admission, declaration/reference identities, reserved-intrinsic
  authentication, physical dependency sealing and immutable frontend evidence.
- Registered descriptor/order types, every region boundary operation's effects
  and verifier, the source-to-region builder, and complete standalone/paired
  verification in `MatcoreTwoGemmRegion.cpp`.
- CLI mode/output isolation, build and CI wiring, direct frontend and MLIR
  tests, upstream storage controls, and unchanged-route executable test source.
- The complete F1-F14 ledger and claim boundaries in `TWO_GEMM_REGION_V1.md`.

Runtime/provider/codegen implementation is unchanged. This was not a new
repository-wide runtime audit or an executable-candidate implementation review.

## Concrete corrections and prudential tightening

Concrete counterexamples caught during design/implementation and covered by
the final rejection/protection tests include:

1. A persistent `static auto &alias = C` can refer to a previous invocation's
   parameter rather than the current one. Only automatic transparent reference
   initializers are chased.
2. Repeated macro-expanded declarations can share a spelling location while
   denoting different bindings. Ambiguous macro identities are rejected.
3. Memory-effect freedom alone admits an unused integer division by zero.
   The final tests require these mutants to pass upstream IR verification and
   then reject them through the Matcore verifier, both outside and inside the
   scalar contraction body.
4. An inspection output can alias a parsed header/dependency, not just the
   main source. Publication now protects the complete captured dependency set.

These are concrete code-level counterexamples; the review does not claim that
every pre-fix implementation was separately built and executed. The source
observer/mutation barriers, pure-tensor DCE loss, and first-success/second-failure
`C=6, E=-9` were motivating architectural falsifiers, not a demonstrated
wrong-result bug in the unchanged per-call runtime.

Prudential hardening is distinct: individual attribute-origin checks protect
against declaration attribute merging; the modeled incidental-operation
allowlist is a conservative certification scope, not proof that upstream
speculatability is unsound. Fresh-context serialization/parse/source pairing
exercises the new type parser without introducing another format or an
authorization route.

## Direct invariants and unopened authority boundaries

F1-F10 directly challenge the admitted region. The first committed postvalue
feeds the second contraction; each read is paired with its stage guard, and D
remains late under MAYalias storage. Distinct declarations are not disjoint
allocation proofs, and A/B aliasing remains legal. Ordered opaque failure
frontiers cannot cross commits. Effects preserve unused output writes through
actual upstream canonicalization, CSE and SymbolDCE. Generalized Linalg is
accepted by maps, iterators and scalar meaning; upstream-valid semantic changes
are rejected. Native source pairing is mandatory: coordinated editable contract
changes can be self-consistent while failing the sealed-source check.

F11-F14 deliberately test unopened authority boundaries. The actual upstream
materialization control exposes allocation/copy and unresolved ownership; it
does not establish source-paired region storage or zero-copy. The actual CPU
lowerer rejects that control and region artifacts even after producer, target,
capability and retry labels are forged.

Those negative tests do **not** implement or prove generic candidate admission,
safe retry after partial writes, loaded-binary authentication, target-artifact
reuse, or hardware/numerical candidate compatibility. The failure obligation is
not a captured C++ handler/noexcept/diagnostic contract. Unchanged host C++ still
owns those behaviors. A retained requirement is not a discharged runtime fact.

## Validation provenance

Independently executed at `de204b5`:

- `frontend.native.two_gemm_region`: PASS, 1/1 CTest, 0.14 seconds.
- Direct frontend test binary: PASS, 28 real native extractions.
- Direct MLIR region test binary: PASS, 83/83 checks.

Independently executed at `f8f15eb`:

- Direct MLIR region test binary: PASS, 85/85 checks, including fresh-context
  text parsing and paired source verification.
- The worktree was clean before this report; `git diff --check` passed at the
  reviewed production checkpoint.

The integration owner's full Release run at `de204b5` was independently
inspected, **not independently rerun**: 73/73 tests passed in 201.33 seconds,
with 73 individual PASS records and no skipped, failed or not-run entries.
Local log: `/tmp/mdslc-region-release-ctest.log`, SHA-256
`dd6d0b7818d9c77ffe95d18ed5ddb4433815d8e02e4574a9e6816fce0fdc485c`.
The owner's focused round-trip log records 85/85 checks, one CTest, 0.04 seconds;
local log `/tmp/mdslc-region-roundtrip-tests.log`, SHA-256
`16d0ab669b468d35b519df2c96132a9186d6305a31a533c3dc2651fa2bcbe414`.
Temporary log paths are local evidence pointers, not portable retained assets.

Wider Debug, OpenBLAS, sanitizer, compatibility and hosted outcomes belong to
the integration record. They must not be inferred from these focused runs or
from the Release configuration, which has OpenBLAS disabled.

## Hosted sanitizer follow-up: allocator protocol, not a suppressed failure

Subsequent hosted ASan+UBSan validation failed four existing MLIR tests during
the first builtin context/type registration, before Matcore's dialect loaded.
The reviewer inspected `/tmp/mdslc-region-ci-asan-job.log`, SHA-256
`f4b2e3987c5e4fd586ab95cccb1322730b6e9bb6a418b299aba6ae7716ca6f0b`.
The first allocation called an instrumented weak LLVM allocator slow path;
the next builtin type accessed a still-user-poisoned byte in the same live
slab. This was not a freed-storage trace or execution of a region operation.

Exact-version installed header inspection explains the protocol mismatch:
the ASan slow path poisons a slab and unpoisons the first allocation, while an
unsanitized inline fast path lacks the unpoisoning required for later valid
allocations. The same mechanism is visible in the
[LLVM 21.1.8 allocator source](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/include/llvm/Support/Allocator.h).
Independent symbol inspection and execution of the MLIR lane's LLVM-only
two-translation-unit reproducer confirmed that link-time selection of the weak
template is sufficient even when the registration function does not execute:
the mixed protocol exits 1 with `use-after-poison`; the compatible registration
object exits 0 and reads/writes its second allocation successfully.

The reviewer separately built an ASan+UBSan caller linked to the compatible
registration object. Valid allocation passed; deliberate manual poisoning,
heap overflow and heap use-after-free each exited 1 with the corresponding
ASan diagnostic. Scratch source:
`/tmp/mdslc-asan-review-controls.kqc2f3/controls.cpp`, SHA-256
`0b12aac0c659d0ed593b25da63c75901c059253d3ecf5d5288388f94deec6a2e`.
Only diagnostic symbolization was disabled after local symbolizer stalls.
Poisoning, leak checks and failure-on-error remained enabled. Scratch processes
were allowed to finish or explicitly stopped; no system/toolchain changes or
duplicate repository builds were performed by the reviewer.

The correction at `3c81bfb8509bef54c798bebe6b5a59cf0be802a5`, plus the integration
owner's source-specific CMake option, was independently reviewed. Only
`MatcoreRegionTypeRegistration.cpp`, containing one `addTypes` call, is compiled
with trailing `-fno-sanitize=address`; UBSan remains enabled. Actual Ninja
commands confirmed that the parser, boundary operations, region builder and
verifier, and tests keep ASan+UBSan. No semantic or execution logic was moved
into the shim.

After the correction, the reviewer independently executed the five MLIR
sanitizer binaries under leak detection, halt-on-error, strict string checking,
initialization-order checking and UBSan halt-on-error:

| Binary scope | Independent outcome |
| --- | --- |
| Semantic core | 204 checks, zero failures |
| Map/domain | 343 checks, zero failures |
| CPU runtime-dispatch lowering | 18 checks, zero failures |
| Recovered GEMM bridge | 78/78 checks passed |
| Ordered two-GEMM region | 85/85 checks passed |

The integration owner's six-test rerun, including the new frontend test, was
also independently inspected: 6/6 passed in 0.62 seconds, including all four
original failures. Log `/tmp/mdslc-region-asan-fixed-tests.log`, SHA-256
`c5c7f111a8667b49961c3d6917957e45f50448c0203140e7ea5f0a25467a6542`.
The integration record owns the complete sanitizer and subsequent hosted
outcomes; the individual reruns above do not substitute for those gates.

The durable real-MLIR controls were additionally reviewed. The checker retains
caller sanitizer policy; a caller disabling manual poisoning must fail it.
The reviewer requested explicit runtime checks rather than Python `assert`,
which optimization can erase. The implementation lane made that correction and
reported an optimized-Python positive run plus a poison-disabled negative run.
Those two additional harness executions were not independently rerun here.

Rejected alternatives included disabling user poisoning globally, suppressing
the `SmallVector` diagnostic, excluding entire MLIR/semantic libraries from
ASan, and removing the failing tests. This scoped correction restores the
pinned non-ASan prebuilt dependency's allocator protocol; it does not claim
sanitizer instrumentation inside that prebuilt library. A coherently
ASan-instrumented upstream tuple must revisit this explicit boundary. There is
no remaining blocker in the reviewed shim; exact final hosted checks remain
required before integration approval is expanded.

## Exactly one next engineering boundary

Build an inspection-only per-call guard/discharge ledger for this same admitted
region. Map retained requirements to exact source proofs or still-required
predicates, using existing runtime validation as the oracle and preserving both
failure frontiers. Raw-pointer shape/range validation must not be mislabeled
proof of backing allocation capacity or lifetime. Do not bundle fusion,
generated execution or generic candidate selection into that next boundary.
