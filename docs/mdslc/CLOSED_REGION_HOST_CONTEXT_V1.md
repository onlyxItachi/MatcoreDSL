# Closed-region host-context admission v1

This is the bounded continuation of [closed-region admission](CLOSED_REGION_ADMISSION_V1.md),
not another language design or an execution consumer. Engineering began from
canonical `e53f2c6302430f574473ba1a644cc42f1169384a` (PR #35) in the isolated
`mdslc/closed-region-host-context-v1` worktree. PR #36's operator checkpoint is
integrated normally before the final candidate; no history is rewritten.

## Question and result

Can the **same** closed mathematical/value/resource/effect grammar coexist with
real C++ host headers, preprocessing, declarations and unrelated effects while
retaining reproducible source authentication? The focused proof says yes on the
pinned Linux x64 Clang/LLVM/MLIR 21.1.8 tuple. Public syntax, installed driver
behavior, header-defined mathematical helpers and generated execution remain
outside this proof. The previous frontend-neutral semantic records and private
MLIR model are unchanged.

## Implemented seam

```text
configured Clang tuple + bounded argv + explicit working directory
  -> recording LLVM VFS: source/header bytes and filesystem lookup outcomes
  -> real Clang preprocessing / Sema
  -> shared closed AST grammar + declaration/source/FP-policy fences
  -> physical-input recheck, immutable snapshot, frozen Clang replay
  -> exact comparison using the existing semantic verifier
  -> final physical-input recheck and sealed inspection evidence
  -> later pairing against historical frozen inputs, never live source fallback
```

Clang remains the parser, preprocessor and template instantiator. Matcore observes
preprocessor events and admits an explicit AST grammar; it neither executes
helpers nor evaluates C++ callbacks. The same annotated private free function
is the falsification instrument. No new spelling, lambda/proxy mechanism,
public header or `mdslc++` option was introduced.

The context captures effective compiler arguments and order, working directory,
configured executable/loaded Clang library identity, resource-directory identity,
main bytes, header bytes and physical identities/path chains. Positive,
negative and positive-but-unopened lookups all matter. Status, open, realpath,
directory and locality observations are recorded separately. Unknown replay
queries fail the audit rather than masquerading as ordinary `ENOENT`.
Normalized Clang preprocessor observations are additionally bound to the
semantic specimen. Length-bound fields and collection counts are internal
fingerprint implementation details, not a serialized authority format.

Physical checks before issuance catch tested byte/identity replacements,
symlink retargeting and changed lookup outcomes. They are **not an OS-atomic
filesystem snapshot** against an arbitrarily hostile concurrent writer. Once
issued, evidence authenticates its historical inputs. Editing or removing a
physical header does not alter that history; a fresh capture must see the new
context. Replay still rejects newly poisoned compiler environment variables
before invoking Clang. Historical evidence never authorizes compilation or
execution against changed live files.

The recorder is deliberately bounded: 1 MiB main source, 16 MiB per opened
file, 64 MiB total parsed bytes, 32,768 filesystem queries, 250,000 preprocessing
events, and an allowlist of C++20/include/define/undefine options. Arbitrary
compiler flags, modules, PCH, overlays, plugins, target overrides, fast-math
and volatile date/time/timestamp preprocessing are rejected.

## Expressiveness and closure

Useful controls include actual iostream/string/RAII/macros **outside** the
selected region; local/transitive headers; header-staged exact canonical type
aliases; reused main-source template helpers; static rectangular RHS-carried
GEMMs; symbolic shapes with both shape-conditional branches; explicit per-GEMM
numerical profiles; late reads and ordered publication/observation. No source
program is executed by admission.

Host calls, IO/network effects, hidden lifetimes, header-defined helpers,
forged/copied primitive declarations, inherited FP policy, macros within
admitted declarations and helpers, skipped unsafe source, body includes,
forged source coordinates, and pragma-generated region markers reject.
Canonical AST identity alone is insufficient: empty macro expansions and
continued pragma annotations can disappear from, or look ordinary within,
the surviving AST. Clang callbacks and upstream raw lexing fence those cases.

The original 486-check admission and 71-check semantic suites remain the
closure/value/effect oracle. The host catalogue adds 38 independently authored,
ordinary-C++20-valid sources (seven admissions, 31 policy rejections). Integration
adds context mutation/replay tests and two further mathematical controls.
Every catalogue admission is also presented to the actual CPU lowerer, which
must reject it and clear any stale dispatch records.

## Retained failed prototypes

1. Initial host run: **157 checks, five failures**, not passing evidence.
   An unqualified fingerprint helper named `bind` selected `std::bind` through
   argument-dependent lookup for some string fields. Header changes could leave
   the context digest unchanged. The helper is now `appendBoundField`; tests
   independently perturb header bytes, unused definitions, cwd and include order.
2. Historical pairing initially reread live files. Three failures showed this
   contradicted sealed-snapshot replay. Physical checks now occur before issue;
   historical pairing uses frozen inputs while rejecting ambient driver poison.
3. Independent adversarial AST inspection showed a continued pragma can inject
   an exact marker with neither implicit nor inherited attribute flags. A
   one-physical-line fence was inadequate. Upstream Clang raw lexing now supplies
   the whole logical directive extent; both forged-marker forms reject.
4. Rejecting all inherited attributes also rejected an already-supported explicit
   main-source prototype marker inherited by its definition. A new positive
   control passed against the preserved pre-host archive (506 checks, zero
   failures). The redundant attribute flags were removed; complete declaration,
   attribute and preprocessor source fences remain the actual forgery defense.

These were falsified implementation drafts, not failures of the closed-region
language model. They are retained as mechanical regressions, not hidden by
weaker assertions or execution-policy changes.

## Integration and validation

The only modification shared with the existing native route is extracting its
two runtime version/library-path query bodies **verbatim** into
`native_frontend_runtime_identity.cpp`. This avoids importing the entire legacy
AST consumer and allocator template instantiations merely to reuse its identity
queries in a Clang+MLIR inspection binary. No new sanitizer exclusion is used.
Windows keeps its existing behavior; the new host proof is Linux-scoped.

Final local validation at clean source checkpoint
`00f269503da2d16a1e11d9cc0d21add214d35604`, after normally integrating canonical
`7e64f20237329d6e129ef798615f85bb19dd5cee`:

| Check | Observed result |
| --- | --- |
| Focused Release CTest | 5/5 passed, 14.51 seconds |
| Complete Release standalone CTest | 78/78 passed, 234.99 seconds |
| Focused Debug ASan+UBSan, exact CI selection | 27/27 passed, 23.72 seconds |
| Final host / hermetic admission / semantic checks | 179 / 506 / 71; zero failures |
| Independently syntax-checked host source catalogue | 38/38 ordinary Clang C++20 valid |
| Repository hygiene and `git diff --check` | Passed |

Release uses the exact tuple in `AGENTS.md`, MLIR and experimental vector
readiness enabled, OpenBLAS disabled. Reconfigure from the clean source commit
before the complete suite, then `cmake --build build-host-release -- -j2` and
`ctest --test-dir build-host-release --output-on-failure -j1`. The suite includes
frontend contracts, semantic/structured/buffer/vector verification, original CPU
execution, planner/runtime/ABI, installed consumers, source-inaccessible package
installation and benchmark-evidence contract tests. It does not produce new
performance or provider-parity evidence.

The separate Debug build uses `-O1 -g -fsanitize=address,undefined
-fno-omit-frame-pointer` in both language flags and the sanitizer link flags;
vector readiness and OpenBLAS are disabled. Run with `DEBUGINFOD_URLS=` and the
strict ASAN/UBSAN options and exact 27-test regex in
`.github/workflows/mdslc-native.yml`. The allocator protocol's intentional-error
controls still detect errors. The new HostAdmission and shared runtime-identity
objects retain sanitizer instrumentation and define none of the checked
`BumpPtrAllocator`, `LazyGenerational` or `PagedVector` allocator templates.

Independent final integration review ACCEPTed the complete diff against
canonical `7e64f20237329d6e129ef798615f85bb19dd5cee` at source commit
`00f269503da2d16a1e11d9cc0d21add214d35604`, including the compatibility repair,
all source/context counterexamples, build integration and unchanged execution
boundary. Approval remained conditional on the local and exact-head hosted
gates. The subsequent validation-record commit changes documentation only.
Hosted evidence and the normal-merge checkpoint are recorded after those gates
complete, not inferred from local success.

Independent evidence:
[adversarial review](agent-reports/closed-region-host-adversarial-v1.md),
[frontend implementation report](agent-reports/closed-region-host-frontend-v1.md),
[input-layer implementation report](agent-reports/closed-region-host-context-review-v1.md).
The input-layer author's implementation report is not independent approval.

## Unchanged architectural limits

Logical values are not physical allocations. Resource IDs and distinct
descriptors prove no physical disjointness. Publication, observation, checked
failure and completion remain separate obligations; a publication token does
not prove rollback, atomicity, synchronization or external storage lifetime.
The f32 output-rounding and within-reduction permission boundaries are unchanged.
No storage adapter, snapshot/donation policy, materialization, fusion,
buffer/vector lowering or generated region execution is added. Existing
authenticated CPU runtime/provider execution remains the only authority.
Issues #15 and #20 remain open; this produces no parity or accelerator evidence.

## Next decision boundary

The next meaningful boundary is **a closed-region external-storage/publication
adapter contract review**, before implementing its consumer. Admission now
survives real host context, but does not establish when an immutable read can
borrow, when prior values need preservation, or what a fallible publication
exposes. Snapshot versus borrow/reuse and failure-visible update contracts have
consequential semantics. The review must recommend one bounded contract using
alias/late-read/old-value/provider-partial-write counterexamples; owner approval
is required before choosing and implementing that contract. It is not permission
to add public types, physical materialization or generated execution.
