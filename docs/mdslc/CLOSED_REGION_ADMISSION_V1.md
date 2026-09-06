# Closed-region admission feasibility v1

This is a private, inspection-only architecture experiment. It is not public
MDSL syntax, an installed frontend feature, a tensor/view API, or an execution
consumer. The owner approved the restricted-region architecture and this
feasibility milestone after the review at local commit
`cd12496b054963876d4b22f07850ca0f07323fe8` on
`review/mdsl-language-boundary-v1`. The canonical starting point was
`a94b01067d390f0b3f997cd09692dba38cdba455` (clean local/main/origin and GitHub).

## Question and falsification boundary

Can closed C++ admission preserve frontend-neutral mathematical/resource/effect
meaning while admitting useful composition, reusable helpers and symbolic shape
control? Rejecting every helper/control construct would not answer this question.
Executing a graph-building callback or inventing a second evaluator for C++ would
also fail it. This experiment uses Clang parsing, Sema and normal template
instantiation; it lowers an explicitly admitted AST grammar, never executes it.

## Private source instrument

The instrument selects an annotated free-function definition, not a capturing
lambda. Its declarations are fixed bytes embedded in the test-only frontend.
Input snapshots and source identities are supplied to a private C++ test API;
there is no new `mdslc++` option, source rewrite or runtime entry point. Ordinary
inclusion of the fixture header fails with an inspection-only diagnostic.

Source types distinguish opaque, trivial logical value handles from external
storage handles. Shape scalars use an ordinary unsigned C++ integer type.
Canonical operations read a specified shape from a storage state, compute an
f32 GEMM value, publish a value to storage, observe storage, or query value
dimensions. Explicit numerical-profile arguments are per operation. This does
not reinterpret the existing mutating `matcore::mdsl::gemm` operation.

The demonstrated grammar is one selected, annotated, non-overloaded free
function; Storage/unsigned-64 Shape parameters; immutable local bindings;
canonical direct calls; recursively checked nonrecursive Value/Shape helpers;
and region-body `if/else` using a builtin shape comparison. Clang performs
ordinary template instantiation for reusable helpers. Both branches retain
their own computation and effects; branch-local values do not escape.

This is not arbitrary C++ staging. Expression-bearing `decltype`/`typeof` in
region declarations is rejected; host type aliases formed outside the region
may resolve through Clang to an exact canonical type. No helper is executed by
admission. The fixed source fixture rejects preprocessing/inclusion through
Clang's lexer and an isolated in-memory filesystem; full host-header integration
is not demonstrated by this instrument. Capturing lambdas, loops, mutable local
bindings, general scalar arithmetic, branch-result joins, recursion, low-precision
values and arbitrary views remain unsupported.

The declarations are falsification instruments, not a proposed spelling for a
product language. No ownership or noalias fact follows from passing a handle.
No raw pointer capacity, lifetime, alignment, initialization, or concurrency
claim is established by successful source admission.

## Semantic contract

The transient admission records contain only typed IDs, source spans and
mathematical/resource operations, never Clang AST pointers. The internal MLIR
specimen is built from these frontend-neutral records. No overlapping JSON
optimizer schema is introduced. Its private dialect is test-only and does not
change the legacy semantic dialect or executable GEMM lowering.

- Logical values are immutable contents, not allocations or live pointer views.
- Every external resource pair remains MAYalias, including distinct parameters.
- The bounded read/publication request is an f32 dense row-major storage view.
  Compatibility is a required adapter predicate, not a property proved about an
  opaque handle. This does not assign a physical layout to intermediate values.
- Reads bind the current conservative resource/effect frontier. Publication
  changes that frontier for subsequent reads of every potentially aliased
  resource. This state token is not physical storage identity.
- Old values retain their meaning across publication. Borrowing, snapshots,
  copy-on-write and allocation strategy are deliberately not chosen here.
- Dynamic shape/storage obligations remain explicit and unproven. Ordered
  checks are not erased merely because a computed value is unused.
- Publication and observation retain their source order and failure frontier.
  They are semantic effects, not generated physical writes or synchronization.
- A fallible publication is not an atomic transaction; partial mutation remains
  possible. Tokens supply neither rollback nor precise asynchronous retirement.
- Each GEMM retains its f32 output-rounding boundary. Permission to reassociate
  one reduction is not permission to reassociate across GEMMs.
- Shape scalars remain explicitly 64-bit with unsigned comparison; a target's
  MLIR index width cannot silently narrow ordinary C++ Shape semantics.
- Both branches of admitted shape control remain represented. Admission does
  not choose a branch by running source code or sample dimensions.

The model intentionally uses conservative global ordering rather than claiming
precise physical alias analysis. Removing or splitting that ordering requires
new, independently justified effect/resource proofs.

## Authority

Structural verification checks an internal specimen. Only pairing against the
immutable native admission evidence establishes its relationship to the input
snapshot and injected declaration contract. Editable hashes and authority
attributes alone do not authenticate a source. Exact paired comparison applies
to this untransformed admission seam; it is not an optimizer-equivalence theorem.

Mathematical failure checks, resource publication, guaranteed observation and
completion requirements remain distinct operations/obligations even though this
conservative proof uses one ordering carrier. The fixture's `noexcept`
declarations do not select a future exception/status ABI. Machine FP-status and
trap adaptation remains explicitly unresolved and execution-forbidden; an
abstract value operation is not a claim that a provider or machine instruction
cannot fail or mutate FP state.

Every specimen remains inspection-only. No consumer connects it to v1-to-v0
projection, runtime lowering, bufferization, fusion, vectorization or generated
execution. Existing CPU/provider execution remains the sole authority.

## Validation and disposition

The source-positive cases include rectangular lhs/rhs dependencies, a template
helper reused twice with dynamic shapes, shape-conditional second GEMMs in both
orientations, late reads after publication/observation, old values used after
their possible backing storage is published, explicit numerical profiles and
dead-result checked obligations. Unsupported low-precision/literal conversions
are rejection probes, not new operation support.

An independently constructed SSA-ancestry oracle checks the prefix before the
second GEMM check: read A, read B, first check, publish C, observe C, late read D.
It also moves the late read earlier while repairing tokens/epochs: the result is
structurally self-consistent but must fail source pairing. This tests actual
source-order authority, not merely malformed IR rejection. No C++ interpreter,
GEMM execution, provider failure injection or physical write is involved.

The final catalogue contains 65 independently authored, ordinary-C++20-valid
hostile sources: 58 pass the fixture's Clang/Sema phase and fail admission; seven
fail the explicit no-preprocessing preflight. The independent review also checks
all 65 with ordinary Clang and the embedded declarations. One proposed late
attribute witness was discarded because Clang ignored that attribute; the
replacement is confirmed in Clang's AST before being counted as a rejection.

Focused integration results on the coherent Linux x64 LLVM/Clang/MLIR 21.1.8
tuple, with OpenBLAS disabled:

| Check | Observed result |
| --- | --- |
| Release admission executable | 486 checks, zero failures |
| Release semantic executable | 71 checks, zero failures |
| Release `ctest -R closed_region -j1` | 3/3 passed, 0.15 seconds |
| Debug ASan+UBSan admission/model/ordinary-header/allocator protocol | 4/4 passed; independently repeated in 0.49 seconds |
| Complete Release standalone suite at `e7d7140af842b0c6e367d116525a4b0a71c7f5a4` | 77/77 passed, 231.64 seconds |
| Established focused Debug ASan+UBSan scope plus new admission/model | 26/26 passed, 6.15 seconds |
| Repository hygiene and `git diff --check` | Passed |

The sanitizer run initially failed at the pinned prebuilt Clang/MLIR allocator
boundary. Inline upstream source/redeclaration queries instantiated instrumented
allocator slow paths that could interpose on unsanitized package fast paths.
The repair uses upstream out-of-line queries for the same complete TU,
redeclaration chains and source buffers; no new sanitizer exclusion was added.
Independent byte-span/line checks, a retained later-redeclaration probe, and the
existing allocator protocol's positive and deliberate-error controls all pass.
The frontend object retains ASan/UBSan instrumentation and no longer defines
those upstream allocator templates. This is toolchain integration evidence, not
a language-architecture counterexample.

The complete suite includes existing native/frontend contracts, per-call and
two-GEMM semantics, structured/buffer/vector specimens, explicit CPU execution,
runtime/planner and ABI tests, installed consumers, C17 ABI, source-inaccessible
package installation and benchmark-evidence contract tests. It creates no new
parity evidence. Issues #15 and #20 remain open.

Reproduction uses the exact tuple in `AGENTS.md`, adding
`-DMDSLC_ENABLE_MATCORE_MLIR=ON`,
`-DMLIR_DIR=/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir`,
`-DMDSLC_ENABLE_OPENBLAS=OFF`, `-DCMAKE_BUILD_TYPE=Release`, and
`-DMDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS=ON`.
The source was committed and clean before reconfiguration, including the
source-inaccessible package-test checkpoint:

```sh
cmake -S compiler -B build-closed-release
cmake --build build-closed-release -- -j2
ctest --test-dir build-closed-release --output-on-failure -j1
```

The separate Debug sanitizer build uses `-O1 -g -fsanitize=address,undefined
-fno-omit-frame-pointer` in both language flags and `-fsanitize=address,undefined`
at link, with vector-readiness disabled. Its exact 26-test selection and strict
ASan/UBSan environment are in `.github/workflows/mdslc-native.yml`; local runs
also set `DEBUGINFOD_URLS=` to avoid external symbolizer delays. No system
toolchain packages were changed. OpenBLAS-enabled and native Windows regressions
are hosted-CI gates, not locally executed claims; the private proof is Linux-only.

At that local integration checkpoint hosted gates were pending. They subsequently
passed as recorded below. Detailed evidence:
[frontend lane](agent-reports/closed-region-admission-frontend-v1.md),
[semantic lane](agent-reports/closed-region-semantic-model-v1.md), and
[independent adversarial review](agent-reports/closed-region-adversarial-review-v1.md).
The tests live in `compiler/tests/closed_region/` and
`compiler/tests/mlir/matcore_closed_region_test.cpp`.

## Canonical integration

[PR #35](https://github.com/onlyxItachi/MatcoreDSL/pull/35) merged normally on
2026-09-06 at **`e53f2c6302430f574473ba1a644cc42f1169384a`**. Its parents preserve
canonical base `a94b01067d390f0b3f997cd09692dba38cdba455` and reviewed pre-merge
head `638bedd1ee1d0339a6c6aef6845742cd9b6175e8`; the merge tree is identical to
that reviewed head. No history rewrite, branch deletion or execution change.

All **19/19** exact-head hosted checks succeeded before merge: Linux Release
with OpenBLAS enabled/disabled and MLIR enabled/disabled, MLIR Debug, focused
ASan/UBSan, TSan runtime, native Windows including its sanitizer scope, legacy
CI and repository hygiene. Push and PR event checks were both retained:
[native PR run](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34043556782),
[native push run](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34043554440),
[Windows PR run](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34043556897),
[Windows push run](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34043554518).
Windows validates the unchanged supported route; this private admission proof is
not enabled there. Independent agent review accepted the exact implementation
and integration diff. Copilot's quota-limited non-review was not counted as
approval. The [merge gate](https://github.com/onlyxItachi/MatcoreDSL/pull/35#issuecomment-5560443581)
records those distinctions. Issues #15 and #20 remain open.

## Next justified boundary

Authenticate this same inspection-only subset in a real host translation-unit
context, reusing existing native compiler/options/source/dependency identity.
The fixed empty-filesystem experiment proves useful closed AST admission, not
coexistence with a host application's real headers, macros and dependency
closure. That gap must close before selecting storage/publication realization.
No public syntax, generated execution or new storage strategy follows from this
next boundary. The operator checkpoint changes only after an engineering merge.
