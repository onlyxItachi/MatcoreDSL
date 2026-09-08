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
- Final engineering input under validation:
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

## Final validation (in progress)

Engineering input is `2082852915a688baf6f2c7a8bf7a2bcdcd153332`.
Both builds explicitly select `matcore-mlir` and `row-contiguous`, coherent
Clang/LLVM/MLIR 21.1.8. The current Release driver SHA256 is
`908dc80083d5801016d4717b0ded1d00e82f7a93601c60b7d32e6028c0af6850`.

| Scope | Current outcome |
| --- | --- |
| Fresh Release, OpenBLAS ON, full build | Passed, then row-selected rebuild passed. |
| First full row-selected Release CTest | 138/140 passed; two cleanliness refusals described below. |
| Fresh Debug ASan/UBSan, OpenBLAS OFF | Full build passed. |
| Affected sanitizer CTest | Running; inventory 86 mechanically checked against the CI regex. |

The first full Release run took 311.30 seconds and is **not a successful gate**.
The owning agent created this untracked report while tests were running.
`package.installed_source_inaccessible` and
`benchmark.cpu.native_blas_parity_runner_contract` correctly refused the dirty
worktree, explicitly identifying this report path. All 138 remaining tests,
including all new library cases, passed. No production change or weakened check
is justified. Commit the report, reconfigure from a clean head and rerun the
entire suite with no concurrent edits before claiming final acceptance.

Build/test logs are under the owning worktree's ignored `build-library-release`
and `build-library-asan` directories. No other worktree/build was modified.
Root's concurrent compilation means elapsed times are not benchmark data.
Hosted CI, final installed-package isolation, tests-disabled configuration and
normal integration/merge are not claimed by this report at this stage.

## Limits

All bodies remain source-visible and admitted through Clang/Sema. Opaque
semantic libraries, independent module loading, source/ABI stability, arbitrary
header C++ effects and general cross-module optimization remain unsupported.
The compatibility per-call main-file rewrite boundary is unchanged. No new
target, storage/view operation, memory reuse, fusion, performance advantage or
BLAS parity follows from this source-organization capability.
