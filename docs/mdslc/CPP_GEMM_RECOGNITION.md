# Conservative ordinary-C++ GEMM recognition v1

- Status: Milestone D design contract; no implicit source replacement is
  enabled by this document
- Scope: one canonical, row-major, rank-two F32 GEMM loop in a `.mdsl`
  translation unit parsed by the native Clang 21 frontend
- Architecture source of truth: ADR-0009

## Purpose

MDSLC already recognizes an explicit `matcore::mdsl::gemm` call by authenticating
its canonical declaration after Clang Sema. This document defines the first
conservative experiment for recovering the same mathematical operation from an
ordinary C++ loop.

The experiment has two goals:

1. prove that explicit and recovered mathematical intent can enter one Matcore
   semantic operation; and
2. prove that recognizing an algebraic pattern does not silently authorize a
   source-semantic change.

It is not a general polyhedral recognizer, alias analyzer, loop optimizer, or
new source language. It does not add a public operation, mutate Matcore IR v1,
or authorize a default production rewrite.

## Recognition and permission are separate

Every candidate has exactly one of four internal states:

| State | Meaning | Host compilation |
| --- | --- | --- |
| `not_recognized` | The source does not have the canonical algebraic skeleton. | Compile ordinary C++ unchanged. |
| `recognized_rejected` | The algebraic skeleton is present, but one or more legality proofs failed. | Compile ordinary C++ unchanged. |
| `recognized_guard_required` | Static proofs succeeded and every unresolved precondition has a representable, pre-mutation runtime guard and exact ordinary-loop fallback. | Until guarded replacement is validated, compile ordinary C++ unchanged and expose the candidate only for inspection. |
| `raised` | All static proofs and required dominating guards are present, and a verified Matcore semantic operation was created. | Replacement is allowed only within the proven guarded path. |

`recognized` is therefore never synonymous with `raised`. A normal build does
not fail because recovery failed. An explicit Matcore call retains its existing
fail-closed diagnostics because it is a user request; an ordinary loop remains
ordinary C++ whenever implicit acceleration cannot be proven legal.

Milestone D may construct and inspect a guarded semantic candidate. Default
source replacement remains disabled until Milestone E validates the guard,
fallback, runtime, artifact, and numerical-environment path end to end.

## Canonical accepted loop

The first recognizer accepts the following semantic and AST shape only:

```cpp
#include <cstddef>

void ordinary_gemm(float* c,
                   const float* a,
                   const float* b,
                   std::size_t m,
                   std::size_t n,
                   std::size_t k) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0F;
      for (std::size_t p = 0; p < k; ++p) {
        acc += a[i * k + p] * b[p * n + j];
      }
      c[i * n + j] = acc;
    }
  }
}
```

Identifiers and redundant parentheses may differ. Comments and whitespace are
not semantic. The accepted v1 form otherwise requires all of the following:

- a non-template free-function definition written in the main `.mdsl` file;
- `void`, non-variadic, with exactly six parameters;
- one mutable `float *`, two `const float *`, then three canonical
  `std::size_t` values;
- a function body containing exactly the outer loop;
- three perfectly nested `ForStmt` nodes with compound-statement bodies;
- fresh `std::size_t` induction variables initialized by zero;
- exact conditions `i < m`, `j < n`, and `p < k`;
- exact prefix increments `++i`, `++j`, and `++p`;
- an F32 accumulator initialized by the positive F32 literal `0.0F`;
- one built-in F32 `+=` update whose right side is one built-in F32 multiply;
- direct row-major indices `i*k+p`, `p*n+j`, and `i*n+j` tied by declaration
  identity to the induction and dimension variables;
- one final built-in assignment of the accumulator to the output element;
- no output read, input write, alpha/beta scaling, transpose, padded stride,
  conversion, overloaded operator, or additional statement.

The recognizer compares Clang declarations and canonical types, not identifier
spelling or source text. It strips only parentheses and implicit casts that
Sema introduced. An explicit cast is outside the v1 form.

The recovered semantic types are:

```text
lhs    tensor<?x?xf32>  shape [m, k]  strides [k, 1]
rhs    tensor<?x?xf32>  shape [k, n]  strides [n, 1]
output tensor<?x?xf32>  shape [m, n]  strides [n, 1]
accumulation f32
```

Dynamic symbols are scoped to this operation. Equal names in another recovered
site do not imply a cross-site dimension relationship.

## Two-phase Clang analysis

The native frontend should register a shallow matcher and perform exact typed
validation in C++. A monolithic matcher would be difficult to review and could
silently broaden as Clang AST details evolve.

A suitable candidate matcher is equivalent to:

```cpp
finder.addMatcher(
    forStmt(
        isExpansionInMainFile(),
        unless(hasAncestor(forStmt())),
        hasAncestor(functionDecl(isDefinition()).bind("enclosing-function")))
      .bind("outer-loop"),
    callback);
```

Phase one identifies the triple-loop, scalar-reduction, two-load, one-store
algebraic skeleton. Phase two independently verifies types, declarations,
indices, control flow, effects, aliases, compiler state, floating-point
semantics, source ownership, and guard availability.

Relevant Clang 21 APIs include:

- `clang::ast_matchers::forStmt`, `hasAncestor`, `functionDecl`,
  `isExpansionInMainFile`, `hasLoopInit`, `hasCondition`, `hasIncrement`, and
  `hasBody`;
- `declStmt`, `hasSingleDecl`, `varDecl`, `binaryOperator`, `unaryOperator`,
  `arraySubscriptExpr`, `hasOperatorName`, `ignoringParenImpCasts`, and
  `equalsBoundNode` where useful;
- `ASTContext::getParents`, `ASTContext::getSizeType`, and
  `ASTContext::hasSameType`;
- `FunctionDecl::getTemplatedKind`, `FunctionDecl::redecls`, declaration and
  parameter attribute iteration, and `CXXRecordDecl::isLambda`; permission
  checks the complete redeclaration chain because Clang need not inherit an
  optimization/offload attribute from a prototype onto its definition;
- `AttributedStmt::getAttrs`, `OptimizeNoneAttr`, and `LoopHintAttr`;
- `Expr::HasSideEffects`, `Expr::getFPFeaturesInEffect`,
  `QualType::isVolatileQualified`, `Type::isAtomicType`, and `AtomicExpr`;
- `SourceManager::isWrittenInMainFile`, `getSpellingLoc`, `getFileID`, and
  `getFileOffset`;
- `Lexer::makeFileCharRange` and `Lexer::getLocForEndOfToken`; and
- `PPCallbacks::MacroExpands` and `PPCallbacks::PragmaDirective`.

A small `RecursiveASTVisitor` performs the barrier and observable-effect scan.
`clang::CFG::buildCFG` may be used as an additional single-entry/single-exit
check, but does not replace the exact structural dependence proof.

## Structural dependence proof

For the accepted form, the proof is closed-form rather than a claim of general
dependence analysis:

- each `(i,j)` output element is written exactly once after its K loop;
- `acc` is local to one `(i,j)` iteration and does not escape;
- only K terms update `acc`;
- A and B are read-only;
- C is never read;
- base pointers and M/N/K values are invariant within the region;
- induction variables are modified only by their canonical increments; and
- distinct `(i,j)` iterations address distinct output elements when checked
  size and layout preconditions hold.

A and B may overlap because both are read-only. C must not overlap either
input. Ordinary C++ pointer parameters do not prove this, so the semantic
operation records output/input no-alias as a required precondition rather than
an unconditional optimizer fact.

## Runtime guard contract

The canonical pointer form normally reaches `recognized_guard_required`.
Before an optimized replacement can execute, a dominating guard must check:

- positive M, N, and K for the current positive-dimension semantic contract;
- overflow-safe `M*N`, `M*K`, `K*N`, element-count, and byte-count arithmetic;
- non-null pointers when a positive dimension makes them observable;
- natural F32 pointer alignment;
- non-overlapping output/input byte intervals;
- the required floating-point environment; and
- availability of a legal selected implementation.

The guard must complete before packing, speculative vector access, destination
mutation, or another observable effect. Address-range construction must reject
integer overflow.

The guarded source behavior is conceptually:

```cpp
if (all_recovered_gemm_preconditions_hold()) {
  execute_verified_semantic_gemm();
} else {
  execute_the_original_loop_exactly();
}
```

Guard failure is not an explicit-target error. It preserves the ordinary C++
path. This is distinct from an explicit `fallback::error` request and must be
visible in recovery diagnostics. The fallback must use the same frozen source
snapshot and must not be reconstructed from a pretty-printer.

## Numerical-semantics proof

The default evaluation order of a handwritten accumulator loop is not the
`explicit-gemm-f32-v1` contract. In particular, a strict increasing-K loop
cannot be replaced by an implementation-defined K reduction merely because
its indices resemble GEMM.

A recovered loop may use the same expanded numerical fields only after its
effective Clang semantics prove them. The implementation must inspect the
update expression rather than authenticate literal command-line spelling:

```cpp
clang::FPOptions fp =
    update_expression->getFPFeaturesInEffect(context.getLangOpts());
```

The initial supported proof requires:

| Effective Clang state | Required value |
| --- | --- |
| `getAllowFPReassociate()` | `true` |
| `allowFPContractAcrossStatement()` | `true` |
| `getNoHonorNaNs()` | `false` |
| `getNoHonorInfs()` | `false` |
| `getNoSignedZero()` | `true` |
| `getAllowReciprocal()` | `false` |
| `getAllowApproxFunc()` | `false` |
| `getRoundingMode()` | `NearestTiesToEven` |
| `getExceptionMode()` | `FPE_Ignore` |
| `getAllowFEnvAccess()` | `false` |

`CompilerInstance::getCodeGenOpts()` must additionally report IEEE input and
output handling for both `FPDenormalMode` and `FP32DenormalMode`. Implicit
recovery is disabled at optimization level zero.

One reproducible positive Clang 21 command profile is:

```sh
-O2 \
-ffp-contract=fast \
-fassociative-math \
-fno-signed-zeros \
-fno-trapping-math \
-fhonor-nans \
-fhonor-infinities \
-fno-reciprocal-math \
-fno-approx-func \
-fno-rounding-math
```

This spelling is test input, not the proof. Later options, pragmas, target
defaults, and toolchain state may change the effective values. Extraction and
host compilation must receive materially identical flags.

Reject recovery when effective semantics include strict reduction order,
within-statement-only or disabled contraction, no-NaN/no-Inf assumptions,
approximate functions, reciprocal substitution, signed-zero preservation,
dynamic rounding, observable/trapping exceptions, FENV access, or non-IEEE
subnormal handling. `-ffast-math` is not an acceptable shortcut because its
additional permissions exceed the reviewed profile.

Rounding mode, exception-mask, and FTZ/DAZ runtime state remain guard inputs as
required by ADR-0009. A mismatch executes the original loop in the recovered
path.

## Effects and barriers

The recognizer rejects a recognized candidate if the region or its enclosing
context contains any of the following:

- a macro-originating loop, bound, index, arithmetic update, or store;
- source outside the main `.mdsl` file;
- a template body or instantiation, class-template context, or lambda;
- a `constexpr` or immediate-function context;
- `OptimizeNoneAttr`, `#pragma clang optimize off`, or optimization level zero;
- a loop-specific `LoopHintAttr` or unsupported controlling pragma;
- a volatile-qualified access;
- an atomic type, expression, builtin, or library operation;
- inline assembly, OpenMP/OpenACC constructs, or another parallel directive;
- a function call, overloaded operator, constructor, destructor cleanup, or
  hidden conversion with observable behavior;
- `if`, `switch`, conditional expression, `break`, `continue`, `return`,
  `goto`, label, throw, coroutine, or exception region;
- a side-effectful bound, base, or index expression;
- any additional input write, output read/write, accumulator escape, or
  unsupported address space; or
- an unsafe or non-main-file source range.

Unknown constructs reject recovery. They do not trigger an attempt to infer
purity from mathematical appearance.

## Matcore IR v1 and MLIR boundary

Matcore IR v1 remains the authenticated, deterministic capture DTO for the
explicit operation surface. A recovered loop has no canonical
`matcore::mdsl::gemm` declaration and must not forge one merely to pass through
v1.

The recovered candidate therefore enters the internal Matcore MLIR builder
directly after Clang Sema and the separate permission proof. Both the v1 bridge
and the recovered path must use one verified semantic GEMM construction API so
they cannot silently diverge in types, destination behavior, effects, aliases,
or numerical fields.

The versioned source-origin record distinguishes at least:

```text
explicit:
  kind = explicit_call
  canonical_callee = matcore::mdsl::gemm

recovered:
  kind = recovered_cpp_loop
  pattern = canonical-row-major-f32-gemm-v1
  permission = guarded
```

Canonical callee identity is explicit-call provenance, not the mathematical
identity of `mdsl.gemm`. Likewise, ordinary-C++ preservation policy belongs to
the recovery/compilation context rather than being forged as the explicit
source operation's `fallback::error` policy.

No persistent recovered JSON IR is introduced.

## Source provenance

For a recovered site, preserve at least:

- the physical source file, line, column, and byte offset of the outer loop;
- the exact half-open byte range of the whole outer loop;
- ranges for the loop-control clauses, accumulator update, output store,
  operand bases, and dimension expressions;
- the stable source and compilation identities;
- the exact source snapshot parsed by Clang;
- the recovered pattern version;
- the expanded, source-derived FP proof; and
- a stable site ID derived from source identity, compilation identity, source
  bytes, outer-loop offset, and `recovered.cpp.gemm.v1`.

Begin/end locations must be non-macro spelling locations in the main `FileID`.
Use `Lexer` token boundaries; do not estimate a semicolon or closing brace with
source text search. Diagnostics may use presumed locations for display, but
physical offsets and the frozen source snapshot own rewriting identity.

## Diagnostics

Recovery diagnostics are deterministic and source ordered. Normal builds use
the following behavior:

- `not_recognized`: silent;
- `recognized_rejected`: optional note/report and successful ordinary C++
  compilation;
- `recognized_guard_required`: optional note/report, no default replacement;
- `raised`: report the exact proof and guard set when explanation was requested.

A test-only require mode may fail if an expected fixture was not recognized or
raised. That mode must never become an implicit normal-driver policy.

Stable rejection reason keys should include:

```text
optimization_barrier
macro_origin
non_main_file
template_context
lambda_context
unsupported_control_flow
volatile_access
atomic_access
observable_call
dependence_unproven
runtime_guard_unavailable
fp_reassociation_forbidden
fp_contraction_forbidden
fp_nan_contract_mismatch
fp_signed_zero_mismatch
fp_rounding_dynamic
fp_exceptions_observable
fp_denormal_mode_mismatch
unsafe_source_range
```

A human diagnostic should state both the recognition result and preservation
decision, for example:

```text
recognized: canonical-row-major-f32-gemm-v1
rewrite: rejected
reason: optimization barrier
ordinary C++ preserved
```

These records are diagnostics, not a second optimizer IR schema.

## Explicit/recovered equivalence

The equivalence test compares verified `mdsl.gemm` operations, not source text
and not Matcore IR v1 JSON.

Normalization may remove or alpha-rename only:

- site and private function symbols;
- source-expression spellings;
- source file, line, column, offsets, and ranges;
- explicit-call versus recovered-loop origin;
- operation-local dynamic-symbol spelling; and
- external compilation/execution context that is intentionally separate from
  mathematical operation identity.

It must still require equality of:

- operand, destination, and result types;
- M/K/N relationships, layouts, and strides;
- accumulation dtype;
- destination/result identity and overwrite behavior;
- mutability, effects, and synchronization;
- alias preconditions; and
- every expanded numerical field.

A recovered origin remains distinguishable in the unnormalized IR and never
claims trusted-header provenance. Tests must mutate each non-normalized
semantic contract independently and prove that equivalence then fails.

## Fixture matrix

### Positive and equivalence fixtures

1. Exact canonical loop under the required effective FP state.
2. Renamed parameters, dimensions, indices, and accumulator.
3. Harmless parentheses, comments, and unusual whitespace.
4. A/B overlap remains legal because both are read-only.
5. Dynamic M/N/K map to operation-local symbolic relationships.
6. Explicit call and recovered loop have normalized semantic equivalence.
7. CRLF input.
8. UTF-8 text before the loop.
9. Input without a final newline.
10. Two canonical functions receive distinct stable site IDs.
11. Runtime non-overlap/alignment guard success once Milestone E is enabled.
12. Zero-dimension fallback executes the original loop once Milestone E is
    enabled.
13. Output/input-overlap fallback preserves the original sequential behavior
    once Milestone E is enabled.

### Recognized but rejected fixtures

1. Default strict FP semantics.
2. Disabled contraction.
3. `-ffast-math`.
4. No-NaN or no-Inf assumptions.
5. Signed-zero preservation.
6. Dynamic rounding or FENV access.
7. strict or may-trap exception behavior.
8. FTZ/DAZ, preserve-sign, positive-zero, or unknown denormal handling.
9. `[[clang::optnone]]`.
10. `#pragma clang optimize off`.
11. optimization level zero.
12. `#pragma clang loop vectorize(disable)`.
13. macro-generated outer loop.
14. macro-generated subscript or update.
15. header-originating inline loop.
16. function-template body.
17. instantiated function template.
18. class-template method.
19. lambda.
20. volatile input.
21. volatile output.
22. atomic output or update.
23. function call in the right-hand side.
24. side-effectful loop bound.
25. early return.
26. `break`.
27. `continue`.
28. `goto` or label.
29. conditional inside the K loop.
30. throw or try/catch.
31. inline assembly.
32. OpenMP/OpenACC directive.
33. unsafe or non-main-file source range.
34. unavailable overflow, alias, alignment, or numerical-environment guard.

### Not-recognized fixtures

1. `C += A*B` rather than overwrite.
2. accumulator initialized from C.
3. output store inside the K loop.
4. `i,k,j` loop order.
5. transposed-B indexing.
6. padded or arbitrary-strided indexing.
7. alpha scaling or another extra arithmetic term.
8. double accumulator.
9. signed induction variables.
10. `<=` bounds.
11. decrementing loops.
12. `i += 1` increments.
13. pointer-offset bases.
14. multiple output stores.
15. GEMM-like text in a comment or string.

For every recognized rejection, the test must verify successful ordinary C++
compilation, no recovered call symbol, no rewritten semantic artifact, and
correct execution against the original behavior. Generated host bytes must
remain identical to the frozen input snapshot when no rewrite was authorized.

## Acceptance boundary

Milestone D is accepted only when:

- the exact positive loop reaches a verified `mdsl.gemm` candidate;
- explicit and recovered WHAT semantics compare equal under the narrow
  normalization above;
- every numerical permission is proven from effective Clang state;
- every unresolved alias/alignment/shape/environment fact remains a required
  guard, not an optimizer fact;
- all barrier fixtures fail closed without changing host behavior;
- source provenance is exact and deterministic;
- no recovered declaration identity or Matcore IR v1 record is fabricated;
- existing explicit frontend, v0/v1, package, and host-only behavior remains
  green; and
- independent review finds no unresolved high- or medium-severity semantic
  issue.

This milestone does not authorize broad idiom recognition, GPU lowering,
backend scheduling, public API changes, or a claim that implicit recovery is a
production default.
