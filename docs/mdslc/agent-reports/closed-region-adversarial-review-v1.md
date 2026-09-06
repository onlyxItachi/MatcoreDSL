# Independent closed-region admission adversarial review

Date: 2026-09-06. Lane: `closed_region_adversary`.

## Verdict

**ACCEPT the bounded inspection-only feasibility milestone**, subject to the
integration owner's affected regression and hosted-CI gates. The demonstrated
subset satisfies both closure and useful expression. No counterexample found
requires changing the approved B architecture to standalone MDSL.

This verdict does not establish a public frontend, arbitrary C++ admission,
host-header integration, storage realization, floating-environment adaptation,
provider failure adaptation, generated execution, or Windows admission support.
Those claims remain outside this milestone.

The lane read the canonical repository guidance, native declaration/region
authentication and tests, and the approved language decision review before
implementation. Canonical main was clean at
`a94b01067d390f0b3f997cd09692dba38cdba455`. Production implementation was owned by
other agents; this lane supplied hostile sources, independently reviewed their
implementation, and supplied an additional SSA-trace/mutation oracle.

## Acceptance evidence

- Nine positive source specimens exercise lhs/rhs carried GEMMs, rectangular
  noncommuting geometry, dynamic shapes, a reused ordinary C++ template helper,
  symbolic shape branches containing mathematics or effects, immutable old
  values after publication, numerical profiles, value shape queries, dead-result
  checks, and explicitly staged host type aliases.
- Helper reuse is actual Clang-instantiated body admission, with fresh semantic
  values per call and retained body/call-site provenance. No helper is executed
  to construct the graph. Runtime branches retain both arms; they are not chosen
  by sample dimensions or a C++ callback.
- The independently authored source catalogue contains 65 valid C++20 sources:
  58 must pass isolated Clang parsing/Sema and then fail semantic admission;
  seven deliberately fail no-preprocessing preflight. The two classes are
  checked separately, rather than counting parser failures as closure proof.
- All 65 sources additionally passed ordinary `/usr/bin/clang++-21 -std=c++20
  -fsyntax-only` with the exact embedded declarations prepended. The temporary
  checker and per-source files were retained in
  `/tmp/mdslc-adversarial-syntax.Cz034BJd/`. This check ran no source program.
- Rejection probes cover unknown calls, stream/file/network/system operations,
  implicit conversions, construction/destruction, cleanup attributes, default
  arguments, helpers/templates hiding effects, recursion, escaping references
  and pointers, volatile/atomic access, exceptions, virtual/indirect calls,
  captures, persistent locals, hidden condition effects, nested checked argument
  evaluation, numerical casts, unsupported low precision, user literals,
  forged/redefined primitives, foreign attributes, assembly and control flow.
- Direct expression-bearing type spellings inside regions/helper signatures
  reject. An ordinary alias formed outside the region is explicitly Clang
  staging. The rejected `decltype(host(), Shape{})` probe is not falsely
  described as an executed host effect: its significance is the stated grammar.

## Independent ordering oracle

`adversarial_trace_checks.h` follows the emitted order-token SSA ancestry for
the RHS-carried rectangular specimen. At the second checked GEMM frontier it
requires this exact prefix:

```text
read A -> read B -> check GEMM0 -> publish C -> observe C -> late read D
```

Final publication E must not be an ancestor. The same test verifies the first
publication's value dependency, the immutable C carried in the RHS, and D's
post-publication resource epoch even though D and C have different resource IDs.
This is a required semantic prefix conditional on the preceding effects
succeeding. It is not a storage write, numeric execution or failure simulator.

A compound adversarial mutation moves D's read before publication C and adjusts
the affected dimension definitions, order-token uses and epoch metadata. The
mutant remains structurally valid, but native source pairing rejects it. This
demonstrates why intrinsic consistency and source authority are distinct; the
test does not rely merely on a dangling token or malformed attribute.

## Findings corrected during review

1. Helper/region authentication now checks all declarations and attributes,
   calling convention, parameter defaults and competing selected overloads.
   Primitive definitions introduced after the region cannot acquire authority.
2. Clang raw-token preprocessing screening covers digraph directives, pragma
   operators and spliced tokens. Checking for a literal `#` alone was inadequate.
3. Expression-bearing TypeLocs require explicit staging instead of silently
   passing because their final canonical type matches a handle or shape scalar.
4. Source Shape scalars remain unsigned-64 semantics through signless-i64 SSA
   and unsigned comparisons; target-dependent index narrowing is not inferred.
5. Structural read verification checks explicit dimension operands against the
   returned tensor type, and read failure ordering is explicit.
6. Numerical contracts explicitly retain f32 operation-boundary rounding,
   per-reduction permissions and unresolved machine FP/status/trap adaptation.
   The abstract mathematical value is not relabeled provider execution.
7. The connected admission authority mutation now alters the actual prefixed
   authority attribute, rather than failing only MLIR's generic module-attribute
   spelling rule. Sanitizer test-count integration was also reconciled.

## Independently rerun validation

On the reviewed Release binaries in `build-closed-release`:

```text
bin/matcore_closed_region_admission_tests: 486 checks, 0 failures
bin/matcore_closed_region_semantic_tests:   71 checks, 0 failures
ctest --test-dir build-closed-release --output-on-failure -R closed_region -j1:
  3/3 PASS, 0.15 seconds
```

The tests include ordinary-compiler rejection of the private fixture header.
The lane did not run the complete runtime/package/ABI suite or hosted workflows;
those results must be supplied by the integration owner, not inferred here.

Reviewed SHA-256 identities:

| File | SHA-256 |
| --- | --- |
| `compiler/lib/frontend/ClosedRegionAdmission.cpp` | `9c272c6acfba71196918dffdaf08ee52d3b980b5f02d189edcecea3cc632efaa` |
| `compiler/lib/frontend/ClosedRegionAdmission.h` | `fb0d9c34a551fd0b15f804c37c1a7f8a3d4e6131c4db9c7352b455d26b95ad48` |
| `compiler/lib/mlir/MatcoreClosedRegion.cpp` | `afd0c2a23c3d85e94cb3b415a043f1504706e760cca0061e786b1608829b8367` |
| `compiler/lib/mlir/MatcoreClosedRegion.h` | `939240c209f448f72ced696a5c92ecf81a67146d9ec9bcb2aa18b0cbb4156816` |
| `compiler/tests/closed_region/fixture_language.h` | `71e29e275eb10a229e4ecda94c13719d433508cf9dae740bcc7f5a3daef05191` |
| `compiler/tests/closed_region/admission_test.cpp` | `8efde6d5f408cd35a46f7bcdd8c87a41342dcb9588244527a3d48aa42e7b6698` |
| `compiler/tests/closed_region/adversarial_sources.h` | `eca0ce421bc9a7e5448dc80764899f2ed76d9fb0506115f9ef86c14a763f9293` |
| `compiler/tests/closed_region/adversarial_trace_checks.h` | `a83cd3ef2792e2909b6019a4b93a7a00d6dd118e968ef3d9505a062d1bbfae32` |
| `compiler/tests/mlir/matcore_closed_region_test.cpp` | `45e1d69f955b149b2a6b70b5ac43ea9adee5958597d7a937a917509c98181177` |

## Sanitizer and test-witness follow-up

The first combined admission/MLIR ASan run failed with `use-after-poison` during
MLIR builtin type registration. The model-only binary passed. The retained
local pre-fix log is `build-closed-asan/closed-region-pre-fix-asan.log`, SHA-256
`47a44cb6c5f2b456b1a55058880cf526b46e3618d1fd1a9da33b1eab48344143`.
Symbol and disassembly inspection identified new inline Clang clients emitting
the instrumented LLVM bump allocator slow path into a process with prebuilt
non-instrumented MLIR fast paths. This is the repository's documented mixed
allocator-protocol failure, not evidence that an admitted region executed.

The independently reviewed correction uses upstream out-of-line Clang queries
for source-buffer character locations and complete translation-unit discovery,
and the upstream generic `Decl` redeclaration iterator. It does not disable
instrumentation or introduce an allocator adapter. Same-FileID checks, invalid
lookup checks, bounded monotonic byte ranges, all canonical redeclarations,
and discovery of the complete TU including the forced header remain checked.
Source-span tests now independently check exact operation bytes and line/column
positions rather than only comparing two outputs of the same implementation.

For the reviewed frontend object, `nm -C --defined-only` has no
`BumpPtrAllocator`, `PagedVector`, `getDecomposedLoc`, or `LazyGenerational`
definitions. Undefined ASan and UBSan runtime references remain present.
The lane independently reran:

```text
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
ctest --test-dir build-closed-asan --output-on-failure -j1 \
  -R '(closed_region|allocator-protocol)'
4/4 PASS, 0.49 seconds
```

The allocator-protocol test includes four controls: intentional mixed-protocol
failure, compatible normal allocation, intentional heap overflow, and manual
poisoning detection. Passing it does not mean silently accepting the intentional
invalid clients. No new sanitizer opt-out was accepted for this milestone.

A proposed hostile fixture initially added an attribute after a helper's
definition. Clang ignores such attributes with a warning; the initial expected
rejection was therefore an invalid test witness, not a closure bug. The lane
rejected that interpretation and corrected the fixture: prototype before the
region, attributed redeclaration after the region but before its later
definition. Ordinary Clang AST inspection confirms the `AnnotateAttr` is retained
and inherited by the definition; admission rejects it. The intermediate
486-check run with two failed assertions is superseded by the final 486/0 run,
not counted as a passing result or used to justify a production workaround.

## Exactly one next justified boundary

**Authenticated real host-translation-unit context integration for this same
inspection-only subset.** Reuse the existing native frontend's compiler/options,
physical source/dependency and trusted-declaration authentication while retaining
closed-body admission and the exact value/effect contract.

The empty-filesystem experiment establishes AST closure and useful expression,
not coexistence with a real C++ application's headers, macros and dependency
closure. Closing that remaining B-specific gap follows before choosing storage
publication/materialization strategy. It requires no public syntax, new numeric
surface, generated execution or commitment among physical storage strategies.
