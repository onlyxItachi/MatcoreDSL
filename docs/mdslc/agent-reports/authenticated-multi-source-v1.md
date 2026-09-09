# Authenticated multi-source program engineering v1

## Bounded result and checkpoint

The source-only `--program` route now compiles 2–8 authenticated C++ TUs in one
invocation, with one selected region per region TU and ordinary host utility TUs.
It retains single-source mode, source-visible mathematical helper admission,
private candidate ownership, and the existing region execution contract.
See the [operator contract](../MULTI_SOURCE_PROGRAMS_V1.md).

## Composed local acceptance

Final local validation passed at clean
**`e56e5a648295f7b5ec60bff31f30bf37c031c854`**:

- Full Release: **184/184**, 639.99 seconds.
- Affected ASan/UBSan: **129/129**, 538.91 seconds.
- Actual CI region inventories: **42 OpenBLAS ON / 41 OFF**.
- Independent genuine three-TU policy composition: **164 strict + 152
  reassociated checks**, host trace 123, in each profile. Both exact wrong-oracle
  controls exit 1 with 143/155 expected assertion failures; no crash or skip.

Normal merge `fa292efec1728996854a1bf212c0265af397a999` combines report head
`17405ff42c81cb8228c758d2efee49dd7939048c` with reviewed PR #66 head
`76bcf62183712544b202ce58290f04fef7502e48`. The only manual production resolution
adds the generated-reassociate arm and usage spelling to the shared DriverSupport
parser; both CLI modes still use that parser. Independent composition fixtures
are `c11a4aa` (original `c3346f8`), followed by CTest/CI registration `e56e5a6`.
The normal canonical PR #66 merge is now
`c3b7b0b9a05e11d1b8d8cd3eff1c7ef3551a8280`. Its prospective merge with `e56e5a6`
has exactly the already tested tree
`521a0282c1bc80a694f7460f7e19a89a68811be3`; acquiring that ancestry needs no
production content change. Hosted PR #67 acceptance and its normal merge remain
separate gates, not results inferred from the local suites.

The [independent composition record](program-reassociate-composition-v1.md)
distinguishes real full 4x8/K2 FMA arithmetic from strict multiply/add, preserves
an owning observation across both Result lifetimes and ordinary host mutation,
and checks the later strict operation's exact physical-source failure frontier.
The new test passed rather than skipped in both registered suites. Installed
strict multi-source execution and the existing installed FMA matrix passed
separately; no installed combined-policy or source-hidden multi-source claim is
made.

All six driver/candidate/runtime artifacts were rehashed unchanged after the
full suites. Final driver SHA256 values:

- Release: `23ca6986a99a1b0acdeb7a73bc77a6903c3a650c1664475f4bd2595dfe90a02c`.
- ASan/UBSan: `07d3ce88800b45a0959787750cce36e160db681fc34ce11bc4890c2765d05661`.

Complete composed logs:

- `/tmp/mdslc-program-composed-full-release-tests.log`, SHA256
  `176a66c29e440c3b1d5eb73a9ee37e536c86d16a9c8f6e8fe6f1ba22e770c6f6`.
- `/tmp/mdslc-program-composed-affected-asan-tests.log`, SHA256
  `d373317e6238a2d5f4ab03c862f303c2ac7a60bb723572e6459cf3366f19400d`.

This report followup is prepared separately from the frozen PR #67 source head;
it does not replace, rewrite or restart the recorded execution evidence.

## Hosted acceptance and mergeability reconciliation

Reviewed head `e56e5a648295f7b5ec60bff31f30bf37c031c854` completed **21/21
hosted checks**. Independent raw completed-job inspection of
[native run 34305784001](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34305784001)
and [Windows run 34305783999](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34305783999)
established the following exact scopes, separately from the local 184/129 runs:

| Hosted profile | Passed | Skipped |
| --- | ---: | ---: |
| Release MLIR ON / OpenBLAS ON | 182 | 0 |
| Release MLIR ON / OpenBLAS OFF | 180 | 0 |
| Release row-contiguous / OpenBLAS OFF | 180 | 1 |
| Debug MLIR ON / OpenBLAS OFF | 180 | 1 |
| ASan+UBSan affected scope | 128 | 1 |
| Release MLIR OFF / OpenBLAS ON | 82 | 0 |
| Release MLIR OFF / OpenBLAS OFF | 81 | 0 |
| TSan existing runtime | 4 | 0 |
| Windows existing Release | 48 | 1 |
| Windows existing focused Debug | 31 | 1 |
| Windows focused clang-cl ASan | 1 | 0 |

Every skip was the existing `runtime.cpu.packed_avx512` capability control.
All ten program gates, including `driver.program_reassociate`, actually passed
in all five compiler-enabled Linux profiles. Default Release/OpenBLAS OFF also
passed the separate 1/1 vector-readiness control. Windows evidence remains the
existing standalone/runtime route, not generated-region or multi-source support.

The [independent composed review](https://github.com/onlyxItachi/MatcoreDSL/pull/67#issuecomment-5595377303)
accepted `e56e5a6`. The first normal merge attempt nevertheless received GitHub
HTTP 405 (reported conflicts), while local Git calculated a clean prospective
merge and the API still reported stale base `b7e6cc0`. No canonical mutation
occurred. Normal merge `c22ba0792e9fbe5ffec526e0e1641092b8d0d575` incorporated
actual main `164eb96739c75c051830673bc474cea72320f105` without conflicts or code
edits. Its parents are `e56e5a6` and `164eb967`; its tree is
`e85e768d8b9a61140ac84fd341ebf9fdd1046bff`. Only the two PR #68 documents differ
from `e56e5a6`; `compiler/`, `AGENTS.md` and `.github` are byte-identical.
GitHub then reported mergeable. This normal-history ancestry correction neither
relabeled the earlier executions nor bypassed fresh exact-head hosted acceptance.

## Initial checkpoint retained

Initial local integration passed at **`ed8f7fed1d4a1969328e4f0c9626f0d98151169e`**:

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
merge are not claimed for this initial checkpoint. The then-pending candidate
composition is validated separately above, not retroactively by these old tests.

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

Initial-checkpoint driver SHA256 values, unchanged after its gates:

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
