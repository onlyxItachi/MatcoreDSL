# Experimental region source authentication and owning Result: independent review

Reviewer: independent CPU-candidate/host-boundary agent; no production frontend
or opaque Result implementation was authored by this reviewer. Reviewed commit:
`6dbd4b4e0853748b44312aa3f332af2b6303ac3d`.

## Verdict and scope

**ACCEPT — PROVEN WITHIN A BOUNDED CONTRACT.** The experimental admission path
preserves the existing closed mathematical grammar and now authenticates the
canonical declaration semantics needed by its owning result interface, not just
the physical header bytes or aggregate layout. The opaque Result transfer also
survives the independently tested retirement, allocation-failure and mixed host
standard-library configurations below.

This is not approval of an installed source-to-executable driver, arbitrary host
compiler flags, arbitrary imported IR, generic C++ sandboxing, or a frozen ABI.
Those consumers must preserve the source seal, compiler-owned helper context,
exact ABI checks and supported configuration boundaries independently. Result is
an output/diagnostic interface, never a source-supplied execution-authority token.

## Counterexamples that changed the implementation

**OBSERVED:** identical physical canonical headers did not initially imply
identical C++ declarations. The following valid Clang 21.1.8 translation units
were accepted before their respective corrections:

| Attack | Why physical identity was insufficient | Final result |
| --- | --- | --- |
| `#pragma clang attribute push([[noreturn]], apply_to=function(is_member))` around the canonical headers | Attributes reached owning methods without attributing the enclosing record | Syntax-valid, admission rejects member attributes |
| Host out-of-line `Observation::~Observation()` with an unknown host effect | The first method declaration was owned; its later body was not | Syntax-valid, admission rejects function redeclaration origin |
| Host out-of-line `Result::Result(Result&&)` with an unknown host effect | Layout and the canonical declaration survived while ownership behavior changed | Syntax-valid, admission rejects function redeclaration origin |
| Host completion of the forward-declared private `Session` | Its friendship permitted a source-defined helper to invoke Result's private constructor and construct a completed result | Syntax-valid, admission rejects record redeclaration origin |
| Host completion of the opaque `ObservationBlock` with unrelated layout/destructor | An owned forward declaration acquired an unowned definition | Syntax-valid, admission rejects record redeclaration origin |

These are deliberately source-authentication adversaries. Defining runtime-owned
types/methods differently would also violate the intended whole-program ODR
contract; the compiler now refuses them explicitly rather than relying on an
unobserved link-time diagnostic. No arbitrary-memory hostile-process sandbox is
claimed.

The first member-noreturn attack was genuinely reproduced. Clang 21 rejected the
GNU `__attribute__((noreturn))` pragma spelling as unsupported, whereas the C++11
`[[noreturn]]` spelling compiled with warnings and no errors. The unsupported
GNU spelling and other syntax-invalid pragma experiments are **not** counted as
semantic rejection evidence. Upstream's [multiple-declaration attribute
documentation](https://clang.llvm.org/docs/LanguageExtensions.html#specifying-an-attribute-for-multiple-declarations-pragma-clang-attribute)
describes the relevant pragma mechanism; the pinned compiler observations above
are the version-specific evidence.

Final ownership checks inspect all canonical function/member redeclarations,
member/parameter attributes and owned function bodies; record/enum redeclarations
remain in the owning header FileID. Owned record, member and enum attributes
fail closed except the precise canonical annotations. Keyword macro expansion
inside the immutable owned declarations remains rejected independently.

The corresponding cases are retained in
[`admission_test.cpp`](../../../compiler/tests/experimental_region/admission_test.cpp).
The final independent scratch harness additionally exercised one positive pure
mathematical helper and ten syntactically valid negative programs: member
noreturn/annotation, Observation/Result destructor and Result move redefinitions,
Session/block definitions, two keyword macros and enum-only attributes. Results:
**one admission, ten refusals, zero unexpected results** under ASan+UBSan.
Enum attributes were rejected as contract hardening; no numerical drift was
inferred merely from `enum_extensibility`.

## Owning Result evidence

The abandoned direct-`std::vector` public representation failed the separate-TU
`_GLIBCXX_DEBUG` challenge despite identical public header bytes. Its earlier
same-configuration tests are historical evidence only, not approval of that
representation. The surviving public Result/Observation store scalars and an
opaque pointer and invoke owning operations out of line.

Independent permanent tests establish:

- **16 retirement checks:** an attempted `takeResult` during a running candidate
  cannot steal prior observations or issue the candidate value; eventual failure
  preserves publication/observation prefixes; transfer occurs once; a retained
  observation survives Session and Result destruction.
- **16,068 mixed-configuration/OOM/concurrency checks:** a normally built
  producer/runtime interoperates with a consumer using `_GLIBCXX_DEBUG=1` and
  `_GLIBCXX_USE_CXX11_ABI=0`; actual allocation failures at first/later observation
  retain the exact prefix; eight threads repeatedly copy/move/read/release
  independent immutable handles after ownership transfer.
- The observation block and vector growth occur before the associated effect
  retires. Moving its pointer into Result does not introduce allocation after
  publication. Failed observation allocation cannot manufacture an observation.

These tests are `independent_retirement_test.cpp`,
`independent_result_producer.cpp` and `independent_result_consumer.cpp` in the
[experimental tests directory](../../../compiler/tests/experimental_region).
Thread-confined Session use is unchanged; concurrent immutable-handle testing
under ASan/UBSan is not a TSan claim. The allocator/deallocator remains trusted
and must satisfy the documented host/FP contract.

## Exact final validation

Linux x86-64; Clang/LLVM/MLIR 21.1.8. Independent final rerun at the reviewed
commit:

| Scope | Result |
| --- | --- |
| Release affected frontend/semantic/runtime suite | **16/16 PASS**, 25.11 seconds |
| Debug ASan+UBSan same affected suite | **16/16 PASS**, 32.24 seconds |
| Additional independently linked source-auth harness | **11/11 expected outcomes**, ASan+UBSan |

The suite includes 97 experimental admission checks, 506 existing admission
checks, 179 real-host-context checks, 71 semantic checks, 137 adapter checks,
217 independent adapter checks, 132 actual-allocation checks, both public/private
header orders, the independent ownership tests above, and rejection of ordinary
linking of source-only mathematical intrinsics. Counts are assertions within
their respective tests, not unrelated test cases summed into a product claim.

```sh
ctest --test-dir build-region --output-on-failure -j1 \
  -R 'experimental_region|runtime.closed_host|frontend.closed_region|mlir.closed_region_semantics'
env DEBUGINFOD_URLS= \
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-region-asan --output-on-failure -j1 \
  -R 'experimental_region|runtime.closed_host|frontend.closed_region|mlir.closed_region_semantics'
```

No full repository, hosted CI, Windows runtime, performance or accelerator result
is claimed by this independent review.

## Exact source and negative-evidence identities

| Reviewed file | SHA-256 |
| --- | --- |
| `ClosedRegionAdmission.cpp` | `6dbae447249c0ce64f9389fb5ccd674df416ab67ada42679bee66b80a31dfc81` |
| `ClosedRegionHostAdmission.cpp` | `dbffc7c135839d23d53c271c2b1d30603f7a9369c19d210e4888bf6197923a97` |
| public `region.h` | `2717d27b8787f976e2b5b4af2645b38d1409ce0c91181f4e7413e2a00265b45e` |
| detail `region_storage.h` | `acfcb2427fe9efc564fdda78723a9454d313a3ad4d2bb854a25de1f27355ab74` |
| runtime `closed_host_v1.cpp` | `b51ac9b89be9255709b868287e5c4d71e97bc561945659760d6bf9791356bed0` |
| runtime `closed_host_v1.h` | `2d888bb3ccc62180da9524c9063a79be5183ddac5c493912a0339725077b0a6d` |

Local retained negative evidence lives outside production at
`/tmp/mdslc-result-owning-review.d9I9Zs`. The before-member-auth ASan archive is
`a86f020316b0a0f9fab211c003a67b9f5abcf4e48f379343eb692ab79a38342b`;
the final independent harness executable is
`ab4d47e571469071afa8a51c32c25651a5c93a9acc5c3c5071ca0f2d2fad5814`.
The scratch directory is local reproduction evidence, not a durable distribution
dependency; permanent source fixtures retain the falsification requirements.

**Next integration requirement:** preserve these declarations/ownership semantics
while connecting the exact authenticated original host ABI to compiler-owned
orchestration. Neither source text rewriting nor successful LLVM verification
alone proves that host C++ observation, ABI, or source identity was preserved.
