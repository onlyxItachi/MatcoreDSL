# Closed-region host-context independent adversarial review

Date: 2026-09-06. Lane: `closed_region_adversary`.
Canonical engineering base: `e53f2c6302430f574473ba1a644cc42f1169384a`
([PR #35](https://github.com/onlyxItachi/MatcoreDSL/pull/35)).
Worktree: `MatcoreDSL-wt-closed-region-host-context-v1`.

## Scoped verdict

**ACCEPT the bounded host-TU admission implementation**, conditional on the
integration owner's complete affected regression, sanitizer and exact-head
hosted gates. The tested real-header context preserves the previously admitted
value/resource/effect grammar and inspection-only authority. No counterexample
found requires changing B into a standalone language or adding a shadow C++
interpreter. This is not approval of public syntax or generated execution.

This lane independently reviewed the complete new input recorder/replay layer,
host frontend, shared AST-admission changes, runtime-identity extraction, CMake,
workflow and integration harness. The input-layer author is an implementation
owner in this milestone; that author's earlier design advice is not counted as
independent review of the new implementation. This lane owns only the hostile
source catalogue and this report.

## Source counterexamples

`compiler/tests/closed_region/host_adversarial_sources.h` contains 38 source
specimens: seven expected admissions and 31 expected rejections. All 38 separately
passed ordinary `/usr/bin/clang++-21 -std=c++20 -fsyntax-only` with the tool-owned
declarations prepended and their real auxiliary files available. The temporary
checker and sources are retained under
`/tmp/mdslc-host-adversarial-syntax.pIkyIA1N/`. No source program was executed.

Positive cases use actual standard-library IO/RAII/macros outside the selected
region; local/transitive headers; a header-owned exact canonical type alias;
reused main-source template helpers; absent and present optional includes; and
a successful `__has_include` lookup whose file is deliberately invalid C++ and
never read as source. An explicit main-source prototype marker inherited by its
later definition remains admitted rather than shrinking the original grammar.
Useful dependent mathematical computation remains inside
the unchanged region grammar. Additional integration positives explicitly check
rectangular RHS-carried GEMM and symbolic mathematical branches with reusable
helpers, late reads and operation-local numerical permissions alongside real
`iostream` host code.

Negative cases exercise real IO/network/cleanup declarations used inside the
region; header-defined helpers and templates; empty/erasing macros; macros in
types, parameters, conditions, callee and numerical-profile spelling; skipped
conditional source; body includes; forged line coordinates; copied or forged
primitive declarations; inherited function attributes; inherited floating-point
policy; macro-owned region markers; and pragma-generated exact private markers.
Date/time/timestamp builtins outside the body are three separate context-policy
rejections, not evidence that these macros execute hidden region effects.
All cases reach successful Clang/Sema in the integrated host test; policy/source
admission failure is distinguished from C++ syntax failure.

Two targeted ordinary-Clang AST checks strengthened the falsifiers:

- `#pragma STDC FENV_ACCESS ON` in a header leaves `RoundingMath=1` and
  `AllowFEnvAccess=1` on the region compound/call AST nodes; `#pragma clang fp
  contract(fast)` leaves `FPContractMode=2`. These are retained compiler state,
  not ignored pragmas or presumed numerical permissions.
- A continued `#pragma clang attribute push` can attach the exact private
  marker as a plain `AnnotateAttr` on a later physical line, with neither
  `Implicit` nor `Inherited` in its AST dump. Those flags alone therefore do
  not prove declaration-owned source spelling.

## Authentication and replay review

The captured context binds the selected compiler tuple, effective argument
order, explicit working directory, immutable main bytes, dependency contents,
physical file identities/path chains, and observed filesystem lookup outcomes.
The recorder preserves unresolved lookup spelling instead of lexically folding
`symlink/../path`. Open-file status must agree with the previously captured
identity and bounded file contents. Final pre-issuance checks reject changed
bytes, replaced identities, symlink retargeting and changed negative lookups.

Replay has no physical filesystem member or fallback. Captured negative results
remain negative results; a previously unobserved query sets an authority error,
even if Clang might otherwise tolerate that lookup failure. Positive-but-unread
file status remains available without pretending the file's contents were read.
Directory/realpath/locality queries have distinct captured outcomes. The final
source/semantic comparison also binds normalized preprocessing-event evidence.

The fixed declaration buffer is authenticated through its actual FileID and
exact bytes. Canonical primitive redeclarations remain fully checked. Empty
macro expansion and skipped-source events cannot evade admission merely by
disappearing from the AST. Full declaration/helper/type/attribute source ranges
are fenced, while unrelated host preprocessing stays outside those ranges.
Clang's lexer supplies logical pragma-line extent; Matcore does not implement a
replacement preprocessor or execute helper callbacks.

The runtime-version/loaded-library query extraction is behavior-preserving;
the supported native consumer now links the same functions from a shared
private target. The new host experiment remains test-only and Linux-scoped.
No installed API, source driver option or CPU/provider execution connection was
introduced. The actual existing CPU lowerer rejects every admitted host specimen.

## Prototype defects and corrections retained

1. The integration owner observed an initial 157-check run with five failures.
   The initial context-fingerprint helper named `bind` could resolve via
   argument-dependent lookup to `std::bind` for `std::string` fields, discarding
   bound functors instead of adding identity fields. The helper was renamed to
   `appendBoundField`. Header-only byte changes, unused compiler definitions,
   working-directory changes and include-search order now have distinct-context
   falsifiers. Hash syntax alone would not have caught this defect.
2. The initial paired verifier revisited physical source files before/after
   replay. That incorrectly turned historical source authentication into a live
   file-stability requirement. Physical stability checks now occur before seal
   issuance only. Old seals replay their old bytes and negative/positive lookup
   outcomes; fresh captures see changed physical files.
3. Independent review identified that removing physical rechecks must not allow
   live driver-environment injection. Replay now checks the forbidden-input
   environment and configured in-process runtime version before ToolInvocation,
   without reopening source files. Old-seal replay rejects newly injected
   `CPATH` and `SOURCE_DATE_EPOCH`.
4. The first pragma fence covered one physical line. Independent AST evidence
   showed how a continued pragma could place a plain exact-marker attribute on
   the next line. The fence now uses upstream raw lexing through the logical
   directive. Both single-line and continued forged-marker programs reject.
5. Frontend self-review identified that rejecting every inherited annotation
   would also reject an explicit main-source prototype marker inherited by its
   later definition, previously allowed by the hermetic grammar. Independent
   review agreed to remove only those flags, retaining the full canonical and
   source/preprocessing range checks. A host and a hermetic positive control
   prevent accidental grammar shrinkage; pragma-origin markers remain rejected
   by their actual source provenance, not irrelevant attribute flags.

These are corrected prototype/verification defects, not reasons to invent a
new source language or widen execution authority. The initial failed run is not
counted as passing evidence.

## Independent validation

On the coherent Linux x64 Clang/LLVM/MLIR 21.1.8 Release build, before the
explicit-prototype preservation follow-up:

```text
ctest --test-dir build-host-release --output-on-failure -j1 \
  -R '(closed_region|frontend.native.two_gemm_region)'
5/5 PASS, 14.64 seconds
host-context executable: 173 checks, 0 failures
existing hermetic admission: 486 checks, 0 failures
existing semantic model: 71 checks, 0 failures
```

This independently rerun selection also covers the old authenticated two-GEMM
frontend and ordinary-compiler rejection of the private fixture header. Root's
separate integration harness covers filesystem mutation/replay, environment
poisoning, compiler/resource rejection, forbidden flags, context sensitivity,
semantic equality with the hermetic control, and actual CPU-consumer rejection.
The working-directory identity comparison refreshes both captures after creating
the alternate directory, isolating the cwd input from directory mutation.
`git diff --check` passed. Complete regression, new sanitizer and hosted results
must be supplied by the integration owner; they are not inferred here.

The lane also independently executed the temporary compatibility probe
`/tmp/mdsl-prototype-control-z2Mqnf/probe`: **506 admission checks, 0 failures**.
It links the original pre-host admission archive (SHA-256
`2e4b1d937f99cdf21dc544937e28e57b4942ecc62703a159f89658cbc466ca19`)
with the added prototype source test. This mechanically confirms that the
restored case was part of the original grammar. The follow-up
`9f696147fcf9bc3613cd2919432cce46f0e2ebf3` removes only the redundant attribute
flags; independent diff review confirms that full source/PP range checks remain.
The integrated host positive and both pragma negatives require the subsequent
fresh regression run, not inference from the old binary.

Reviewed SHA-256 identities at the 173-check focused checkpoint:

| File | SHA-256 |
| --- | --- |
| `compiler/lib/frontend/ClosedRegionAdmission.cpp` | `079b5509cf6ad0d0d9a6693046a64ff84b6c43b1570380a164cddd2141fdeeba` |
| `compiler/lib/frontend/ClosedRegionAdmissionInternal.h` | `91177744230a3011d45fbe0e0d128d5cc5ae4d1cf572564e4e939a68f57cf8fa` |
| `compiler/lib/frontend/ClosedRegionHostAdmission.cpp` | `ac6e262fbc69f42a0183cdc75fd0878633445cc811e301247c299a2cbfb407bb` |
| `compiler/lib/frontend/ClosedRegionHostInputs.cpp` | `4b4afc4c665ee0ec20d18fb5b06eaa90c6544cf171b5bca1a61133736d30990d` |
| `compiler/lib/frontend/ClosedRegionHostInputs.h` | `553b174a12052767f5710638becb8981dbc778c2c7197b70b7d03f8212b5ae4a` |
| `compiler/lib/frontend/native_frontend_runtime_identity.cpp` | `e06459cc44c2c2a0d3098c84d8df486bd72b1d157db27cab0bf5cf4e5bedcff5` |
| `compiler/tests/closed_region/host_admission_test.cpp` | `0295169d6b9f597bacf57ab7b0bec69277e00ea27f3dc0dde2dddf6392909e56` |
| `compiler/tests/closed_region/host_adversarial_sources.h` | `1400e358c0075d64026eb0bc83ab1e82e4ed20c951dcf5bfabecc492af800a02` |

## Claim boundaries

This establishes one useful closed subset inside authenticated real C++ host
context. It does not admit header-defined mathematical helpers, arbitrary host
effects in a region, public tensor/view types, public region syntax, arbitrary
compiler flags or targets, generated execution, storage realization, provider
failure adaptation, fusion or accelerator support. Historical immutable replay
does not authorize compiling or executing against currently changed files.
Issues #15 and #20 remain outside this milestone.

## Exactly one next boundary

**Design-only storage/provider publication adapter contract.** The bounded real
C++ host feasibility gap is closed; arbitrary source-grammar expansion is not
justified by that result. The next question is what an adapter must prove about
imported immutable values, storage versions, old-value preservation across
possible aliasing, publication/observation, partial provider failure and machine
FP/completion behavior before any physical consumer may exist.

Borrowed versus snapshot realization and failure-visible publication choices
still have consequential tradeoffs. This review does not choose them. Obtain
owner approval before committing to such semantics or implementing a consumer;
the proven frontend does not expand generated execution authority. No buffer
lowering, fusion, public tensor/view API or new target follows automatically.
