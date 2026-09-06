# Closed-region host-context frontend lane v1

## Scope and starting point

This lane starts at canonical `e53f2c6302430f574473ba1a644cc42f1169384a`
after PR #35, in the isolated `mdslc/closed-region-host-context-v1` worktree.
It authenticates the **same private, inspection-only grammar** described in
[the first feasibility proof](../CLOSED_REGION_ADMISSION_V1.md) while Clang
parses real host headers and unrelated host code. It adds no source syntax,
public API, installed feature, semantic operation, execution route, or storage
realization. The frontend-neutral `closed_region::Program` and its semantic
verifier remain unchanged.

Owned implementation: `ClosedRegionAdmission.{h,cpp}`,
`ClosedRegionAdmissionInternal.h`, and `ClosedRegionHostAdmission.cpp`.
Input recording/replay is a separately owned `ClosedRegionHostInputs` layer;
root owns build integration, physical-mutation tests, and the canonical handoff.

## Authority and source closure

The original hermetic source API remains a control. The new private host API
accepts options, a working directory and a selected region. Options themselves
are not authority: the input layer validates the configured compiler tuple and
bounded flags/environment, records physical input/query outcomes, and supplies
the **complete effective driver argv**. `ToolInvocation` receives that vector
unchanged and uses the recording or immutable-replay filesystem directly.
There is no caller VFS overlay, argument adjuster, generated execution, or
live-header fallback.

The host action uses ordinary Clang parsing/Sema/template instantiation. It
authenticates the resolved main `FileEntry` against the main `FileID` and exact
captured source digest. The injected private declaration file is authenticated
by its resolved `FileEntry`, parsed `FileID`, and exact compiler-owned bytes,
not by a presumed filename. Any macro/directive event inside that fixed header
rejects admission; command-line macro substitution cannot preserve trust merely
by leaving similar canonical names or annotation strings.

Both entry paths call one AST whitelist. Its complete TU discovery and all
canonical redeclaration checks remain intact. The host path adds Clang
preprocessor observations over declarations, parameter/type spellings,
attributes and every admitted main-source helper. Intersections reject erased
macros and inactive source which leave no AST node. Clang's raw lexer rejects
directives, pragma operators and escaped tokens in those closed source ranges.
Both shape-if arms remain semantic records; no C++ helper or host callback is
executed during admission.

Pragma source ownership requires the **logical directive extent**, not a
physical-line approximation. The observer uses Clang's raw lexer to include
backslash-continued tokens. Ordinary declaration attributes and effective
floating-point options are checked independently: a header can otherwise carry
`FENV_ACCESS` or contraction policy into an apparently harmless handle-only
body. Nondefault effective FP state rejects admission rather than changing the
meaning of explicit per-GEMM numerical permissions.

Unrelated host IO, RAII, templates and macros can remain outside the selected
closure. Header-defined mathematical helpers are still not admitted. Ordinary
header aliases can stage through Clang only to the same exact canonical types.
Date/time/timestamp macro expansion anywhere in the TU is rejected, because
this experiment does not freeze clock inputs. Preprocessor evidence is bounded
to 250,000 recorded events, with overflow rejected after Sema.

## Replay and historical evidence

Successful first admission is followed by input freeze/rechecks, a fresh parse
on immutable replay inputs, exact semantic comparison, and a final physical
recheck before seal issuance. The exact existing semantic verifier compares
the replayed program against a module built from the original program; it is
not a new optimizer-equivalence checker. A normalized closure-observation
transcript binds callback descriptions, actual file names, immutable buffer
digests and checked byte offsets. Its digest and the complete host-context
digest participate in the existing compiler-identity provenance field. Raw
Clang source encodings and presumed `#line` coordinates are not identities.

The in-process evidence payload retains immutable input records, not just
editable digest strings. Later pairing replays those **historical inputs**:
changing or deleting live source/header files cannot replace them. A fresh
capture of changed inputs receives a different context and cannot authenticate
the old specimen. Replay first rejects poisoned process environment or a
mismatched loaded compiler version; unrecorded filesystem queries also fail
closed even when Clang would otherwise treat them as missing files.

Physical checks before issuance detect sampled mutation/retargeting; they do
not claim an OS-atomic filesystem transaction against arbitrary concurrent
writers. No seal provides physical pointer capacity, alias separation,
allocation, publication completion, failure rollback, or execution authority.

## Adversarial development observations

- The first integrated run retained five failures across 157 checks. Three
  exposed an overly strict replay policy: later pairing reread live paths even
  though the approved seal denotes historical captured inputs. Live rechecks
  now occur before issuance, never as a replay source. Two separate input-layer
  identity failures were corrected by that lane and remain in its evidence.
- Independent review constructed a main-source continued pragma that injects
  the exact private marker on a later physical line. Clang retains a plain
  annotation, neither implicit nor inherited. A physical-line observer was
  therefore insufficient; the logical-directive fence is the focused repair.
- The prior fully instrumented source-query fix is retained: upstream
  out-of-line buffer/TU queries and generic declaration redeclaration iteration
  avoid defining incompatible allocator slow paths in admission code. No
  sanitizer exclusion was added by this lane.
- Self-review caught an unnecessary grammar shrink: rejecting all inherited
  attributes would reject an explicit main-source prototype marker inherited
  by its later definition. The original pre-host admission archive mechanically
  accepts this control (506 checks, zero failures with the added positive).
  The redundant attribute-flag rejection was removed; every annotation's full
  source/PP range is still fenced. The actual continued-pragma witness has
  neither flag, so the removal does not weaken its logical-directive defense.

## Validation checkpoint

At this lane checkpoint, the frontend sources pass direct coherent Clang
21.1.8 `-fsyntax-only` checks and `git diff --check`. Root's rebuilt focused
selection passed 5/5 in 11.35 seconds: 169 host-context checks, 486 hermetic
admission checks and 71 semantic checks report zero failures, with ordinary
fixture rejection and the affected existing ordered-region control passing.
The 37-source catalogue includes the continued-pragma rejection. The frontend
agent independently repeated `ctest -R 'closed_region|frontend.native_two_gemm'
-j1`: that regex selected four closed-region tests, all passing in 11.51
seconds. This was a Release test, not a sanitizer result.

The latest Release host-admission object has no `BumpPtrAllocator`,
`LazyGenerational` or `PagedVector` symbols. Root's final sanitizer, clean-build,
hosted-CI and full integration evidence remains separate; these focused results
do not claim completion of those gates.

The prototype-marker compatibility experiment compiled the updated hermetic
test source against the preserved pre-host header/archive, writing only to
`/tmp/mdsl-prototype-control-z2Mqnf`. The old admission archive SHA-256 is
`2e4b1d937f99cdf21dc544937e28e57b4942ecc62703a159f89658cbc466ca19`;
the old worktree remained clean. This proves preservation of existing admitted
behavior rather than introducing a new annotation grammar. Current-build
reruns of the new positive and all negative fences remain integration gates.

Real-header positives already demonstrate rectangular dependent GEMMs, late
reads and ordered publication/observation, reused main-source template helpers,
canonical aliases, and positive/negative optional-header staging. Integration
also carries the existing proof's useful static rhs-carried and symbolic
shape-if controls into real host files: their purpose is to verify that host
context changes provenance only, not operand orientation, ordered effects or
both-arm shape representation. Those additional controls are root-owned and
are not included in the 169-check count above.

## Architecture disposition

The bounded design preserves the separation demonstrated by the original
proof: C++ supplies the host and its normal Sema/type staging, while one closed
admission grammar produces frontend-neutral mathematical/resource/effect
semantics. Real host headers do not require a proxy-template language, C++
interpreter, or a second semantic graph. This lane does not establish arbitrary
host-TU support, public region syntax, device execution or storage planning.
The existing CPU/provider execution route remains the only execution authority.
