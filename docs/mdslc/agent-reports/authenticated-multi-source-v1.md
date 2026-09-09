# Authenticated multi-source program engineering v1

## Bounded result and checkpoint

The source-only `--program` route now compiles 2–8 authenticated C++ TUs in one
invocation, with one selected region per region TU and ordinary host utility TUs.
It retains single-source mode, source-visible mathematical helper admission,
private candidate ownership, and the existing region execution contract.
See the [operator contract](../MULTI_SOURCE_PROGRAMS_V1.md).

Local integration passed at **`ed8f7fed1d4a1969328e4f0c9626f0d98151169e`**:

| Gate | Actual result |
| --- | --- |
| Full Release, OpenBLAS ON, row-contiguous strict schedule | 160/160; 535.32 s |
| Affected Debug/O1 ASan+UBSan, OpenBLAS OFF, same schedule | 106/106; 420.08 s |
| Mechanically enumerated region-driver hosted scope | 35 tests |
| Independent source/alias/publication suite | 39/39 in each profile |
| Independent real-Clang call ABI/effect unit | 36/36 in each profile |
| Installed `--program`, genuine three-TU executable | 37 mathematical/effect checks in each profile |

These are local correctness/integration results, not timings for a performance
comparison. [Draft PR #67](https://github.com/onlyxItachi/MatcoreDSL/pull/67) was
opened at this exact head while the local gates ran. Hosted acceptance and normal
merge are not claimed. The separately pending generated-reassociate candidate
lane must be composed through the shared policy parser and independently tested;
this checkpoint does not claim that future composition passed.

## Implementation and authority

The stack starts from accepted H1 head
`75c78f586801a108ec3cc49b4003b83c34a9cb26`. Behavior-preserving private driver and
frozen-host inspection extraction is separate commit `607dd5e`; multi-source
behavior is `a65a825`. Corrections are `15f1665` and `3ef0a78`, independent fixtures
are `7d980b9` (original reviewer commit `1876063`), and installed/CI registration
is `ddd6a8c`. The final normal merge adds canonical `b7e6cc0` documentation only.

[ProgramHostInterface](../../../compiler/lib/frontend/ProgramHostInterface.cpp)
uses Clang's out-of-line DynamicRecursiveASTVisitor and actual physical header
identity. Header-free ordinary TUs need no forced include. Canonical issued
symbols require genuine source declarations; unrelated ordinary overloads and
truthful host weakrefs remain ordinary C++. Every original TU retains its own
frozen bytes, main FileID, include/query closure and preprocessing transcript.
Removing the public source prelude changes the actual recorded context rather
than concealing that change in an identity normalization.

[ExperimentalProgramCompiler](../../../compiler/lib/codegen/ExperimentalProgramCompiler.cpp)
freezes all TUs, obtains unoptimized Clang LLVM, and applies source, entry,
artifact, retired-Value-helper and foreign ABI/effect ownership checks before
cross-TU linking. Fresh compiler-issued declaration and direct-call witnesses
reuse the original structural ABI/attribute algorithms. They neither import
foreign LLVM nor excuse a dirty declaration or call-site effect promise. The
existing sealed per-region compilation and ABI thunks remain authoritative.

[DriverSupport](../../../compiler/tools/mdslc-region/DriverSupport.cpp) shares
the original exact artifact pins, staging, compiler environment, source rechecks,
native linking and atomic no-clobber publication between both CLI modes. No
production `.cpp` inclusion, serialized module format, object-import authority,
or alternate algorithm copy was introduced.

## Retained counterexamples and corrections

The [independent review](authenticated-multi-source-independent-v1.md) preserves
the exact fixtures, raw records and hashes for three material findings:

- A real `weakref` carrying `pure`/`const` was initially admitted when Clang kept
  the target function clean but attached a false effect promise only to its
  call. The issued executable still passed the mathematics oracle: this was an
  authority defect, not an observed numerical miscompile. Targeted source alias
  ownership plus actual call-witness comparison now rejects it before link.
- Name-only reservation initially over-rejected an unrelated ordinary overload.
  Exact protected linker-symbol ownership removes that restriction without
  allowing alternate source names to borrow the genuine interface proof.
- The inherited inspection prelude rejected valid `int mdsl_probe` host code.
  Public-region and ordinary host capture now explicitly select no prelude;
  inspection defaults remain. A separate valid-LLVM mutation exposed a missing
  called-pointer address-space check, now covered by the original type matcher.

The first no-prelude unit run preserved 128 passing admission checks and one
failure: its private-grammar negative depended on injected declarations while
requiring ordinary C++ syntax validity. The fixture now explicitly declares its
own namespace/Storage and still requires both valid syntax and refusal. All 129
checks then passed. No assertion was dropped. Initial setup errors—wrong root
CMake project, one not-yet-built legacy test executable, and an EXDEV hard-link
fixture—were corrected and are not represented as product regressions. Earlier
six-test and 29-test runs are historical scopes, not the final inventories above.

## Reproduction and evidence identity

Configure the standalone `compiler/` project with the coherent Clang/LLVM/MLIR
21.1.8 tuple. Both final configurations were refreshed from the clean committed
head before complete builds and tests. Release used `ctest --output-on-failure
-j1`; sanitizer selection was extracted from the same two workflow expressions
whose actual inventories are asserted in CI. ASan used leak detection,
halt-on-error, strict string checks and initialization-order checks; UBSan used
halt-on-error and stack traces. No sanitizer suppression was added.

Final driver SHA256 values, unchanged after all gates:

- Release: `61da18ddbbf5ef47563d92a3b0bf92657e136d91712731dfcd1fc9f700c1952e`.
- ASan/UBSan: `060d31e67e4476d74045d77202397682fc32f8c2e1ec20dd26db0da5e2f6fc88`.

Local complete logs:

- `/tmp/mdslc-program-final-full-release-tests.log`, SHA256
  `11f6eb906059dffb523c96a9ea032e51e7ce2e3108f70e0b3416cf7da56d8222`.
- `/tmp/mdslc-program-final-affected-asan-tests.log`, SHA256
  `8ed575f77c222eb1c49dc96c90b256c9388a8c253196b8aab39682f450aa6e12`.

The installed-program test copies the six-source closure into a fresh prefix's
adjacent test directory, executes the installed driver, checks exact phase and
37-check output, and rehashes driver, executable and sources. It does **not**
claim source/build-hidden multi-source execution; the existing single-source
source-inaccessible package gate passed separately in the full Release suite.

## Limits

No mathematical module import, cross-region transformation/fusion, program-wide
rollback, general host-effect verification, accelerator support, generated-region
Windows, zero-copy or performance-parity claim follows. Ordinary C++ lifetime,
storage, allocator and trusted loader/provider preconditions remain. A separate
unavailable Value helper body is not imported merely because its ABI links.
The remaining mathematical frontier is a checked transformed region/storage
derivation, not another interpretation layer or relaxed source-pair equality.
