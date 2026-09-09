# Source-visible pure mathematical libraries v1

2026-09-09. Bounded implementation of source-visible C++ mathematical helpers
in ordinary headers. No public import syntax, opaque modules, second C++ parser,
interpreter, custom linker or ABI-as-mathematics contract was introduced.
Final local validation: full Release **146/146**, affected ASan/UBSan **92/92**,
and independent source/ownership review accepted the corrected engineering
input. The first failed gate and subsequent specialization counterexample are
preserved below; no hosted CI or integration result is inferred.

## Identity and decision

- Initial source: `0e8c040809ae1fe6b0f1c01e41fc509b07212be7`.
- H1 implementation: `7d44425621aa2f454ff951e0510acdf920feb5dc`.
- Normal canonical merge: `c3e45ad`, bringing in current main
  `0b2c21f126288501fe27afea9320ec9fce392dcc` (row-contiguous opt-in, PR #60).
- Independent test source: `9659fb5187144721a8bb63844824ae73e0ed4c4c`,
  cherry-picked as `ebb0159`; registration `87fd2b7`.
- Initial engineering input validated before the specialization correction:
  `2082852915a688baf6f2c7a8bf7a2bcdcd153332` (adds affected CI selection).
- Final corrected engineering/validation input:
  `486d3e5d74e17fbf305c73fca3e3760ef7da59b6`.

The preceding feasibility experiment, research commit
`3669d4d92178ea330227294e2f489c73f3f80906`
(`docs/mdslc/agent-reports/semantic-modules-feasibility-v1.md`),
distinguished reusable, fully available source from separately distributed
semantic authority. This implementation takes only H1: Clang sees every helper
definition, the existing host capture freezes its dependencies, and admission
expands the same restricted mathematical grammar. A same-ABI opaque declaration
or object still cannot supply a mathematical definition.

## Representation and source authority

`compiler/lib/mlir/MatcoreClosedRegion.h` now has a mandatory frontend-neutral
SourceFile table with dense nonzero ids, paths, SHA256 digests and byte sizes.
Record 1 is explicitly the main source and must agree with Program's existing
main identity/digest. Additional semantic-bearing files are sorted by path and
all site references remapped deterministically. Zero/missing file ids never
mean main. Repeated inclusion can share one file record; different paths are
not collapsed merely because their bytes or offsets match.

This semantic file table is **not** the complete input closure. The existing
private HostInputSnapshot still captures all consumed includes, failed lookups,
directory metadata, compiler options and toolchain context. Admission obtains
file bytes and spelling identity from actual Clang SourceManager/FileEntry
records in that captured filesystem. There is no new ambient filesystem loader
or path-only allowlist. The internal builtin GEMM explicitly supplies its own
contract-string record; synthetic verifier tests explicitly supply synthetic
records. Neither receives source authority from an implicit default.

`ClosedRegionAdmission.cpp` keeps entry discovery, marker, ABI and completion
main-owned. Each helper's definition and redeclarations are separately checked.
Call arguments bind in the caller's current body owner; only then does expansion
enter the actual callee body FileID, restoring the caller on exit. Type spellings,
local declarations, operations and source ranges remain macro-free and confined
to that physical body. A semantic file id is never converted back into one
assumed Clang inclusion FileID. This matters when a header is included twice.

The existing Value/Shape-only helper grammar, bounded Sema-resolved template
instantiations, recursion/depth/node budgets, exact canonical primitives,
numerical policies and effect restrictions are preserved. No template evaluator
or general C++ execution was added. Header observations now fail actual purity
checking, rather than being rejected merely because the definition is in a header.

## Witness, diagnostics and ownership

`MatcoreClosedRegion.cpp` extends the existing source dictionaries and module
table verification. Every operation origin and helper call references a valid
bounded file range; region definitions remain main-owned. The complete table,
file references and existing operation facts participate in exact paired MLIR
comparison. Entry/helper source binding comparison also includes file ids.
This remains an **exact untransformed witness**, not transformation authority
or an import format. The whole-region execution architecture is unchanged.

Admission errors use physical spelling filenames and line/column, not presumed
`#line` coordinates. Full helper origin and outermost-first call chains survive
expansion. `ExperimentalRegionEmitter.cpp` preserves the existing public
failure-location policy: select the outermost helper callsite, now resolving its
explicit file identity. Completion remains attributed to main. No observation,
publication, f32 rounding or required-failure frontier is removed.

Original host LLVM is still produced from frozen inputs before LLVM passes.
Only sealed concrete Value-helper symbols are retired. The existing
AuthenticatedHostThunk requires them to be present/unique and rejects remaining
nonhelper uses. Shape-only helpers retain ordinary host uses. No ownership check,
weak-linking rule, candidate DSO isolation or selected adapter publication
guarantee was weakened.

## Tests and counterexamples

The initial scalar Release prototype passed MLIR semantic checks (80), private
source admission (506), and expanded experimental admission (118): 3/3 CTest.
Additional mandatory-table and same-offset/different-file paired negatives were
then added before final validation. They require missing/zero/unknown file ids,
invalid bounds/table order and mismatched main identity to fail; a structurally
valid different-file origin must still fail exact pairing.

The expanded admission test also checks transitive two-GEMM/Shape libraries,
source-file digests, nested cross-file call chains, helper definition files,
header redeclarations, deterministic parse/freeze header replacement, and
historical replay versus live-header freshness. Hidden observation, inherited
FP permissions, macro bodies, nested intrinsic arguments and unknown external
bodies retain precise refusal reasons. A spoofed `#line` cannot change the
physical diagnostic location.

The independent [library adversaries](../../../compiler/tests/experimental_region/library_adversary/README.md)
first ran against scalar prototype driver SHA256
`9ba071d2f2a9abfec9663686950fcc118d377f0ead968d3682d0034f3f0447b2`.
The repeated-inclusion executable passed 67 assertions: two namespaces from one
header, physical `#line` immunity, Shape-only ordinary host calls, old immutable
values, owning observations and a dynamically failing unused nested GEMM.
That GEMM fails at frontier 8 after effect frontier 6; earlier publication and
observation remain, later publication does not occur, and public failure is the
outer main call at line 29, column 17. This preserves the selected adapter's
stronger normal-return publication contract, not just generic partial writes.
Two escaped Value-helper cases must reach the exact retirement rejection and
publish no executable; generic admission failure does not satisfy them.

The initial three independent cases were registered as `driver.source_library_*`
in the ordinary Release suite and affected sanitizer selection, increasing the
CI region inventory from 17 to 20 and row-schedule ASan inventory from 83 to 86.
The specialization controls below extend these inventories further; none of
these registrations is a hosted CI outcome.

## Pre-correction validation

Engineering input was `2082852915a688baf6f2c7a8bf7a2bcdcd153332`.
Both builds explicitly select `matcore-mlir` and `row-contiguous`, coherent
Clang/LLVM/MLIR 21.1.8. The pre-correction Release driver SHA256 is
`908dc80083d5801016d4717b0ded1d00e82f7a93601c60b7d32e6028c0af6850`.

| Scope | Current outcome |
| --- | --- |
| Fresh Release, OpenBLAS ON, full build | Passed, then row-selected rebuild passed. |
| First full row-selected Release CTest | 138/140 passed; two cleanliness refusals described below. |
| Clean full row-selected Release rerun | 140/140 passed, 380.18 seconds, clean head `8b5b8c31ed8b389293aef4179214a06e192a811e`. |
| Fresh Debug ASan/UBSan, OpenBLAS OFF | Full build passed. |
| Affected sanitizer CTest | 86/86 passed, 199.47 seconds; inventory mechanically checked against the CI regex. |

The first full Release run took 311.30 seconds and is **not a successful gate**.
The owning agent created this untracked report while tests were running.
`package.installed_source_inaccessible` and
`benchmark.cpu.native_blas_parity_runner_contract` correctly refused the dirty
worktree, explicitly identifying this report path. All 138 remaining tests,
including all new library cases, passed. No production change or weakened check
was justified. The report was committed, the build reconfigured/rebuilt from
clean head `8b5b8c3`, and the entire suite rerun without concurrent edits.
The rerun passed. Final semantic/private/experimental/host-context checks were
93/506/118/179 respectively, with all three independent source-library cases.

Raw log SHA256 values:

- First failed Release: `9d252d52f624dc5383ee15dc24396b5ee641806149ff7e20a4392ef46d9c6739`.
- Clean Release rerun: `799e9a708e74c56a3f6d3a54baf9a67b200e02a2240eb2ed98063da7b70329f2`.
- Affected sanitizer: `8fda84d47ffbbd489bc54e88dfbc9454d8a5dfe88aa4de1492113a247cfe9c03`.

The pre-correction ASan driver SHA256 is
`81b670b1892d5dabd33e1c9db8b73ab1cedf4380a7313d63f8811881417b2f6b`.
It used Debug `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`,
`DEBUGINFOD_URLS=''`,
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`
and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.
The independent reviewer also executed all three library cases successfully
with this exact driver and environment.

Build/test logs are under the owning worktree's ignored `build-library-release`
and `build-library-asan` directories. No other worktree/build was modified.
Elapsed times are test-gate bookkeeping, not benchmark data; these were not
coordinated quiet performance windows.
Existing installed-package and compatibility source-inaccessible isolation
tests passed on the clean rerun. Those tests alone were not a new installed
header-library demonstration; the separate installed-SDK supplement below
supplies that narrower evidence without hiding source. Hosted CI, tests-disabled
configuration and normal integration/merge are not claimed.

## Independent specialization counterexample and correction

Final review of `8b5b8c3` found a useful source-form rejection: a primary
template in `primary.h`, an explicit specialization in `specialization.h`, and
another ordinary instantiation failed both pre-correction drivers with
`primary.h:3:21: closed-region admission rejected: closed declaration spelling is not body-source-owned`.
Header-primary-only compilation passed, and the equivalent flattened main-file
program compiled and executed the distinct noncommuting mathematical bodies.
This was an ownership compatibility rejection, not incorrect execution.

An independent Clang 21 LibTooling dump established the cause. Clang's canonical
nondefining explicit-specialization stub has its name location in the selected
specialization header, but its entire lexical range, TypeLoc ranges and parameter
source ranges are copied exactly from the primary template. The actual selected
definition and body remain wholly in the specialization header. The earlier
code incorrectly used the stub's name location as its lexical owner.
[Upstream FunctionDecl source-range construction](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/lib/AST/Decl.cpp#L4129-L4131)
retains separate outer-start and stored-end locations; it does not recover a
physical declaration from a filename.

The bounded correction authenticates only this canonical, nondefining concrete
explicit-specialization stub against its exact primary-template lexical range,
full TypeLoc range sequence and every parameter location/range/TypeLoc sequence.
Its selected definition pointer and name owner must also agree. Existing
captured-file, macro, concrete-type and attribute checks still run; actual
selected-body ownership never changes. This is not a general cross-file range
relaxation. The subsequent focused tests and clean full/sanitizer reruns are
recorded below; the 140/86 results above remain explicitly pre-correction evidence.

The independent reviewer accepted correction commit
`c6ff41b763fc97c7cb016ee20fe764c76f7f968e`. Its focused semantic/admission/library
suite passed 6/6 in Release and ASan/UBSan; experimental admission now checks 125
assertions. Independent fixture commit
`4d4e1bdd7f464da939afc749b8257ba0523617c8`, integrated as `865239b`, adds six
[durable specialization controls](../../../compiler/tests/experimental_region/library_adversary/template_provenance.md).
All six independently passed both corrected drivers: generic-T primary plus
separate explicit specialization executes 59 numerical/owning-observation/failure
checks, while macro primary type, macro selected body, hidden selected effect,
escaped selected helper and an arbitrary split cross-file body each require its
specific rejection and no artifact. The late reversed-product shape mismatch
retains first publication/observation, fails at frontier 6 after effect 5, and
attributes failure to the main callsite. Fake success/failure tools do not
satisfy the runner.

Corrected driver SHA256 values are Release
`6ebd962629218e2dbf3f7291be0948c0245994c66711a99e4f4347031580eda0` and ASan
`61ddbb12e17de334819c9b98643c301fedcb1d042ce6cbb245ead78e2ee29341`.
The full Release and affected sanitizer inventories were mechanically checked
as 146 and 92, and the CI region-driver subset as 26, using the committed
selection expressions. Independent integration review accepted exact
`486d3e5d74e17fbf305c73fca3e3760ef7da59b6`: all imported adversary files were
byte-identical to the independent commit, the six new cases were registered,
and prior CI selections were retained.

## Final corrected validation

Both builds were reconfigured from clean engineering head
`486d3e5d74e17fbf305c73fca3e3760ef7da59b6`, then fully rebuilt at `-j2`.
No source or report edits occurred during either final `ctest -j1` run.
The tested drivers retained the corrected SHA256 identities above throughout.
The eventual report-only follow-up is not a new tested engineering input.

| Local scope | Outcome |
| --- | --- |
| Linux x86_64 Release, OpenBLAS ON, matcore-mlir, row-contiguous | Full build passed; **146/146 tests passed**, 339.04 seconds. |
| Debug ASan/UBSan, OpenBLAS OFF, same pipeline/schedule | Full build passed; **92/92 affected tests passed**, 207.67 seconds, with the sanitizer environment above. |

The final logs record 93 semantic, 506 private-admission, 125 experimental-
admission and 179 host-context checks, plus all nine source-library cases.
Existing package isolation, authenticated host/helper ownership, storage/failure
conformance and strict generated primitive controls remain in the passing scopes.
These are local execution gates, not timing comparisons or candidate-identity
telemetry from the public Result API.

Final raw logs in the owning worktree:

- `build-library-release/corrected-full-ctest.log`, SHA256
  `c22b55c98907b0227b7fdc576be72cb7db34fd64d51419206594753089d8fe1c`.
- `build-library-asan/corrected-focused-ctest.log`, SHA256
  `65a334faf57239a4482ac05922dda4c1d0e65672f9b8d1db680e4da77e5bb8a8`.

The implementation and evidence are ready for integration-owner review. No
push, PR creation or normal engineering merge was performed by this lane.

## Fresh installed-SDK source-library supplement

After final validation, `cmake --install build-library-release --prefix
/tmp/mdslc-installed-source-libraries.v8oBiQ` installed the validated Release
artifacts into a newly created empty prefix. No rebuild or production/test edit
occurred. Source was clean report-only head
`485b4966143eb04e79b0c3df2622b756bf9c53b5`; engineering remained `486d3e5`.
The installed executable was invoked by its absolute prefix path, with no
source/build include override and `LD_LIBRARY_PATH=''`, through the existing
`run.cmake` and `template_run.cmake` fixtures. The driver's installed-layout
branch resolves public headers and the private SDK relative to that executable.

All **nine installed-driver cases passed**: repeated inclusion executed its 67
checks, cross-header template specialization executed its 59 checks, and the
seven escaped-use/macro/hidden-effect/split-body controls required their exact
diagnostic and no output artifact. Source/build directories stayed accessible;
there was no hiding, renaming, deletion, or claim of source-inaccessible H1
coverage. All cases forced `generated-strict`; the OpenBLAS-enabled package does
not make these provider comparisons or sanitizer evidence.

Installed SHA256 identities (paths relative to the prefix):

| Artifact | SHA256 |
| --- | --- |
| `bin/mdslc-region` | `31453127bbc7dd4e006ad76936f724dd5388cc32919c8a9a5695ad18b9f11849` |
| `include/matcore/region.h` | `2cbff60e4c5685d1388604df9642126e31296e067bd471e6eae9a70a26c2a39b` |
| `include/matcore/detail/region_storage.h` | `8803a9b31de0e72ffb1045acfede828473b08f9f7292e4fefd27e1dc9b1c0927` |
| `lib/mdslc/experimental-regions/include/closed_host_v1.h` | `d8f3d0312c4cd71e869f32e350d0aa3baa5bbc139b7bdef948099ad6b7a3db7f` |
| `lib/mdslc/experimental-regions/libmatcore_closed_candidates_isolated_v1.so` | `5c58ef46468577475998a3fa42ec842a9787446a92ec0ca13e034bc004977ad3` |
| `lib/libmatcore_runtime.so.0.0.0` | `bbfbbd75b6c73aa7f1cd22227cdc2ab2fa6bdc21b3dee045f9b4346e6dc1851e` |

These are installed-byte identities, separately recorded from build-tree
artifacts. Prefix-local logs remain `install.log`, SHA256
`84b2b71e6802ebcfcda5398d776c70c7b58e9e388f2f781e04869ee75f2cb6e3`, and
`library-cases.log`, SHA256
`5aa0dbd45379c382e4e7345f04a369b5fb0344b4e06ca375ba7efa12dfbdb4a7`.

## Limits

All bodies remain source-visible and admitted through Clang/Sema. Opaque
semantic libraries, independent module loading, source/ABI stability, arbitrary
header C++ effects and general cross-module optimization remain unsupported.
The compatibility per-call main-file rewrite boundary is unchanged. No new
target, storage/view operation, memory reuse, fusion, performance advantage or
BLAS parity follows from this source-organization capability.
