# Closed-region native admission lane v1

Base: canonical `a94b01067d390f0b3f997cd09692dba38cdba455`.
Scope: new `compiler/lib/frontend/ClosedRegionAdmission.{h,cpp}` and
`compiler/tests/closed_region/fixture_language.h`. Existing native extraction,
public header, CPU runtime/provider route and execution driver are unchanged.

## Implementation and authority

The private inspection API accepts an immutable source string, a display label
and one selected annotated free-function name. The label is not physical-file
authentication. Tool-owned declarations are embedded into the frontend binary,
injected into an empty in-memory filesystem, and parsed with fixed Clang C++20
options. There are no caller compiler arguments, include paths, physical header
loads, VFS overlays, object construction callbacks or eager execution fallback.

Clang's raw lexer rejects source preprocessing directives (including digraphs),
pragma operators and escaped-newline/trigraph token cleaning. The source AST is
then admitted using an explicit positive list, not textual operation matching.
Primitive declarations are identified by canonical declarations in the owned
header, exact annotations and every redeclaration; user definitions or merged
attributes invalidate that binding. Names/annotations alone are not authority.

Successful native admission seals source bytes and a frontend-neutral transient
program. No Clang objects cross that boundary. Paired MLIR verification reparses
the sealed bytes under the fixed contract before comparing the complete
untransformed semantic representation. This is an in-process inspection seal,
not a signature, serialized interchange promise or execution permission.

## Closure and expressiveness demonstrated

- Parameters: by-value canonical external Storage handles and unsigned 64-bit
  Shape scalars. Local immutable bindings may hold canonical values, resources
  or shapes. Distinct resource parameters have no inferred noalias relation;
  copying a handle preserves the same resource identity.
- Source-sequenced read, GEMM, publication and observation declarations become
  distinct semantic operations. Numerical profiles are explicit canonical enum
  references; no implicit low-precision or policy conversion is admitted.
- Both dependent GEMM input roles retain their exact noncommuting operand
  order. Read shapes can be static literals, symbolic parameters or shape
  queries on earlier immutable values.
- Ordinary Clang-instantiated function-template helpers can be reused. Their
  bodies are recursively checked and expanded into fresh value identities;
  helper source sites and each outer call-site chain are retained. Helpers do
  not execute and cannot import/publish/observe external resources.
- Shape selection uses an ordinary builtin unsigned comparison. Both branches
  remain in the semantic graph; admission never evaluates the condition or
  selects a branch. Branch-local bindings cannot escape.
- Trivial copies of the exact tool-owned handle records are allowed. Unknown
  constructors, conversions, cleanup, attributes, host calls, pointer/reference
  escape, volatile/atomic behavior, exceptions and unsupported control fail
  closed. Nested checked arguments are rejected rather than assigned an
  invented C++ argument evaluation order.
- Ordinary type aliases formed outside the closed body are resolved by Clang
  and must result in the exact canonical admitted types. Direct expression-
  bearing or attributed type spellings inside region/helper signatures or
  local declarations are rejected. This is a deliberately bounded staging
  distinction, not a claim to interpret arbitrary C++ metaprogramming.

No shadow evaluator, operator-overloading proxy system, custom C++ parser or
template substitution engine was introduced. These tests support B's bounded
feasibility, not every possible mathematical abstraction or final public syntax.

## Deliberate limits

The private source vocabulary has no ordinary implementation. Its `noexcept`
declarations do not select a future execution/failure ABI. Ordinary inclusion
fails with an explicit inspection-only diagnostic. The semantic guard/failure
contract remains inspection-only and no runtime predicate is discharged.

Unsupported: public syntax/API, lambda capture, arbitrary host headers inside
the private fixture, dynamic scalar arithmetic, loops, helper shape-control,
branch-result joins, external IO callbacks, raw-pointer imports, arbitrary views,
buffer donation/materialization, physical completion and generated execution.
Static literal extents are limited to the signed-index domain; symbolic inputs
retain representability obligations. Helper expansion has explicit depth and
AST-visit bounds. These limits are falsification-instrument scope, not permanent
language commitments.

## Review corrections and validation

Independent adversarial/root review caused the following strengthening before
the final candidate: authenticate all primitive and helper redeclarations;
reject selected-name overload ambiguity including prototypes; validate callee
expression form rather than only `getDirectCallee`; reject implicit cast and
cleanup paths; check terminal-return ownership; close preprocessing digraph/
pragma escapes using Clang's lexer; and distinguish explicit external type
staging from unadmitted expression-bearing local type spellings.

On the exact Clang/LLVM/MLIR 21.1.8 Release build, this lane directly reran
`build-closed-release/bin/matcore_closed_region_admission_tests`:
**271 checks, zero failures**. The integration owner owns the harness and
expanded regression/CI evidence. The independently maintained hostile corpus
at this run had 64 cases, with Clang syntax/preflight classification checked
separately from admission rejection.

Direct ordinary compilation of `fixture_language.h` with Clang 21, C++20,
`-fsyntax-only` returned status 1 with the required inspection-only error.
An initial single-file compile exposed a nonexistent Clang `isCoroutine()`
query; it was removed because the compound-body positive list already rejects
coroutine bodies. An initial direct test invocation used the wrong build-tree
subdirectory (status 127); rerunning the located `bin/` executable produced the
271/271 result above. No benchmark or generated GEMM execution was performed.

### Follow-up: fully instrumented prebuilt-library boundary

The first combined frontend/MLIR ASan run failed before builtin dialect
construction completed, with `AddressSanitizer: use-after-poison` and `f7`
shadow bytes. Object-symbol inspection identified weak LLVM bump-allocator
templates emitted by this instrumented frontend translation unit. They could
interpose on the pinned non-ASan MLIR library's allocation fast paths, whose
manual poisoning protocol differs. This was a real dependency-integration
failure, not a language-semantic counterexample or a passed sanitizer run.

Three Clang header paths instantiated those templates:

1. `SourceManager::getFileOffset` through lazy loaded-source entries;
2. `FunctionDecl::redecls` through lazy redeclaration-chain completion;
3. `ASTContext::getTranslationUnitDecl` through the TU redeclaration chain.

The final correction uses existing upstream interfaces with package-owned
implementations: out-of-line buffer/character queries with same-FileID,
invalid-result and integer-address range checks; the base `Decl::redecls`
iterator with checked function casts; and a parsed top-level declaration's
out-of-line TU query before traversing the same complete translation unit.
No redeclarations, header declarations, source checks or semantic checks are
dropped. No custom redeclaration chain or line/column parser is introduced.

A narrowly unsanitized query adapter was considered, then discarded before
integration because these upstream APIs preserve full frontend instrumentation.
Fixing only the source-offset query was insufficient; the two remaining lazy
redecl paths were identified and corrected as well. The final frontend object
has no `BumpPtrAllocatorImpl` or `LazyGenerationalUpdatePtr` definitions and
retains `__asan_init` and UBSan checks. There is **no new sanitizer exclusion**.

Independent adversarial review also corrected a proposed late-attribute witness:
Clang had ignored an attribute placed after a helper definition, so that
fixture did not exercise retained AST attributes. Moving the attributed
redeclaration after the region but before the helper definition supplied a
genuine negative; the unchanged admission rule rejects it.

Final direct validation used ASan+UBSan, with only symbolization disabled to
avoid network debug-symbol lookup and UBSan set to halt on error:

```sh
ASAN_OPTIONS=symbolize=0 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build-closed-asan --verbose -j1 \
  -R 'closed_region|allocator-protocol'
```

**4/4 tests passed**: 486 admission checks, 71 semantic checks, ordinary-compiler
rejection, and 4/4 allocator controls. Those controls require the deliberately
mixed allocator protocol to fail, the matching protocol to pass, and live
heap-overflow/manual-poison defects still to be caught. They would reject
globally disabled poisoning rather than masking the integration failure.

The retained detailed local log is
`build-closed-asan/closed-region-asan-evidence-final.log`, SHA-256
`62f86c68dc91ed88c7a0a854a8cd44d4d978b6fb85f250d7cb8506e0aa16239e`.
The integration owner records subsequent full regression and hosted evidence.

Verdict: **B remains viable for the demonstrated closed subset**, with genuine
helper reuse and symbolic selection rather than a marker-only syntax proof.
Imported-resource realization and publication/completion contracts remain
separate work; this lane supplies no execution authority for them.
