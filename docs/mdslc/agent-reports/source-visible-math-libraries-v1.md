# Source-visible pure mathematical libraries v1

2026-09-09. Bounded implementation of source-visible C++ mathematical helpers
in ordinary headers. No public import syntax, opaque modules, second C++ parser,
interpreter, custom linker or ABI-as-mathematics contract was introduced.

## Identity and decision

- Initial source: `0e8c040809ae1fe6b0f1c01e41fc509b07212be7`.
- H1 implementation: `7d44425621aa2f454ff951e0510acdf920feb5dc`.
- Normal canonical merge: `c3e45ad`, bringing in current main
  `0b2c21f126288501fe27afea9320ec9fce392dcc` (row-contiguous opt-in, PR #60).
- Independent test source: `9659fb5187144721a8bb63844824ae73e0ed4c4c`,
  cherry-picked as `ebb0159`; registration `87fd2b7`.
- Initial engineering input validated before the specialization correction:
  `2082852915a688baf6f2c7a8bf7a2bcdcd153332` (adds affected CI selection).

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

All three independent cases are registered as `driver.source_library_*` in the
ordinary Release suite and affected sanitizer selection. The CI region inventory
increases from 17 to 20 and the selected row-schedule ASan inventory from 83 to
86; these are expected registrations, not hosted CI outcomes.

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
Root's concurrent compilation means elapsed times are not benchmark data.
Existing installed-package and compatibility source-inaccessible isolation
tests passed on the clean rerun. This is not a new separately isolated installed
header-library demonstration. Hosted CI, tests-disabled configuration and
normal integration/merge are not claimed.

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
relaxation. New focused tests and clean full/sanitizer reruns are required before
claiming the corrected input validated; the 140/86 results above remain
explicitly pre-correction evidence.

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
The full Release and affected sanitizer inventories are expected to become
146 and 92; the CI region-driver subset becomes 26. Clean final reruns remain
pending and will be recorded separately from the pre-correction gates.

## Limits

All bodies remain source-visible and admitted through Clang/Sema. Opaque
semantic libraries, independent module loading, source/ABI stability, arbitrary
header C++ effects and general cross-module optimization remain unsupported.
The compatibility per-call main-file rewrite boundary is unchanged. No new
target, storage/view operation, memory reuse, fusion, performance advantage or
BLAS parity follows from this source-organization capability.
