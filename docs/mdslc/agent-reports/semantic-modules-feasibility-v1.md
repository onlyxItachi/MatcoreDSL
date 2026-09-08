# Semantic library/module feasibility v1

2026-09-09. Independent bounded experiment on canonical source
`0e8c040809ae1fe6b0f1c01e41fc509b07212be7`; no production implementation,
public syntax, import format, build, benchmark, push or merge added.

## Decision: separate source reuse from separately compiled semantic authority

**H1 remains the minimum plausible next source-library step:** ordinary C++
headers containing fully visible, frozen, Clang-admitted mathematical helpers
can support reuse without another parser or a new public language. Current
MDSLC does **not** support this yet: only main-file helpers are admitted.
This experiment does not implement or prove the safety of lifting that limit.

**H2 is needed at the unavailable-body boundary, not simply at a file boundary.**
If reusable code is distributed without its admitted source body, and callers
must optimize across it, the consumer needs an authenticated semantic definition
or sufficient certified summary with a binding to its actual implementation.
That is a semantic module/function contract, whether its surface remains C++ or
not. An ELF symbol, C++ signature, MLIR function type, or valid bytecode is not
that contract. A fully source-visible library need not retain runtime calls;
an opaque callable library need not support cross-boundary optimization.

Do not conflate four choices: source organization, semantic identity/interface,
body availability to optimization, and executable ABI/linkage. LLVM LTO alone
does not recover the source-only Value, numerical permissions and ordered
failure obligations missing from an arbitrary external helper declaration.

## Actual evidence and identity

The experiment links only its own small consumer against an existing Release
admission archive, then invokes the real source driver. `ninja -t commands`
reads the existing test's coherent static-library link recipe; it does not build.
The old link output/object/dependency paths are replaced or removed before
executing the compiler argument list. All new outputs reside in a fresh `/tmp`
directory. The driver, Matcore archives, `mlir-opt` and authored experiment files
are hashed before and after; issued executables are also checked unchanged
after execution. This inventory is not a separate toolchain authentication proof.

| Input | Exact identity |
| --- | --- |
| Existing Release build | `/home/hamza-usta/MatcoreDSL-wt-private-candidate-dso-v1/build-isolated-release` |
| Its implementation | `b0e7ac9ac67c3c314a44e5db108be390485404eb`; `git diff b0e7ac9 0e8c040 -- compiler` is empty |
| Real driver SHA256 | `eee73557f8769569f45b51e95a269237d2863fff5fb3b5146a69daa57fdc69e6` |
| Admission archive SHA256 | `5a10d9f19972f4c40d9850d105b833a2f6ae9bc4b707aebc9858dc11be196ebf` |
| `mlir-opt` SHA256 | `18c322575dc399010d8186ff72cbf10047ff7ce99d8e11aec162f33820c89ed3` |
| Toolchain | `/usr/bin/clang++-21`, coherent LLVM/Clang/MLIR 21.1.8; extracted MLIR prefix below |

The final provider-enabled run passed **32/32 admission/closure checks**, six
driver compilations and **14 executable invocations**: eight positive math
checks, four shape-failure prefixes, one strict-provider refusal and one expected
wrong-math failure. The driver independently refused a header helper without
publishing an executable. Eight upstream MLIR invocations passed their expected
outcomes, including a required wrong-type refusal. Total: **37 commands**.

Raw run: `/tmp/mdslc-semantic-modules-v1-d/evidence.json`, SHA256
`0154716e9f5551ef6eff33ac4a3dc44ff24cd6eec7f3cfaedfaa3ce01ee94114`.
The initial scratch run correctly exposed unsupported nested shape calls; a
second run stopped on the experiment runner's empty-argument logging bug after
a successful generated execution. Neither partial run is counted as final.
The corrected complete run retained the nested-call form as a negative control.

Reproduction and supported invocation details are in
[the experiment README](../../../compiler/experiments/semantic_modules_v1/README.md).
The optional-provider-lanes-omitted path also passed 32/32 admission checks and
all 31 expected commands, using the **same provider-ON build**. Its raw record is
`/tmp/mdslc-semantic-modules-v1-e-no-provider-lanes/evidence.json`, SHA256
`ddc9aa91f53308ee04d7b5db404e2a9097b68c50873f4e4c5204d7bfdb63a030`.
No fresh production build, sanitizer execution, provider-OFF build, hosted CI,
installed-package test or performance comparison is claimed. Root's concurrent
compilation/tests make command elapsed times unsuitable as measurements.

## C++ helper and closed-host controls

The positive helper computes `(A*B)*D`, with per-GEMM `strict_f32`, and a
Shape-to-Shape helper preserves the `cols(B)` dimension used for reading D.
Admission exposes seven operations (three reads, two GEMMs, publication,
observation), with helper call sites attached to both GEMMs; it does not retain
a semantic function/call. Completion retires frontier 8, last effect 7.

| Actual fixture | Outcome and precise meaning |
| --- | --- |
| Main-owned pure helper + bound shape helper | Admitted and exactly paired with its MLIR witness. |
| Same body in `math_library.h` | Well-formed C++; refused: `only source-owned free function definitions are admitted`. |
| Helper prototype in header, definition in main | Refused: `helper or region redeclaration is not source-owned`. |
| Entry-only prototype in header | Admitted by the existing explicit entry exception; not helper support. |
| Unknown external helper body | Well-formed C++; refused: `unknown host call or unadmitted helper`. No object acquires authority. |
| Header helper hiding `observe(hidden_storage)` | Refused at origin, **not** evidence of cross-file purity analysis. |
| Same hidden effect moved to main | Refused: `pure helper cannot access external storage or effects`. |
| Numerics helper parameter | Well-formed C++; rejected parameter category. Numerical profiles currently occur directly at admitted GEMMs. |
| Nested `identity_shape(cols(b))` | Well-formed C++; refused nested calls. Separate immutable bindings pass. |
| Main-source or included dependency replacement during parse/freeze | Existing deterministic test callback changes an owned file; neither case can issue evidence. |
| Included dependency replaced after sealing | Live `unchanged()` refuses; historical frozen replay still pairs correctly. Historical replay is not current-file authorization. |

The generated-strict and native-strict drivers execute the strict helper;
existing-native and optional OpenBLAS execute the freshly admitted reassociated
helper. For each of those four lanes, the independent double oracle gives exact
row-major results `[36,36,-18,-18]` for dimensions `(m,k,n,p)=(2,4,3,2)` and
`[-33,12,-9,6]` for `(2,2,2,2)`. All products/sums are small exact integers;
the intermediate product is explicitly rounded to f32. Publication and owning
observation both match. These are small semantic discriminators, not numerical
coverage of arbitrary inputs or provider parity.

A mismatched D descriptor fails read frontier 3, completed frontier 2, zero
effects/publications/observations, and unchanged `-777` output. Strict source
forced to OpenBLAS fails GEMM frontier 4 after the three reads, again with zero
effects and unchanged output. Broader permissions on a fresh source make that
provider legal; they are not inferred from the identical C++ helper ABI.

The operand-swapped helper `(D*(A*B))` has the same mangled C++ symbol, is admitted
as a **different** mathematical program, and returns `[-3,-18,-9,-24]` on the
square fixture. The original oracle correctly fails. This is a negative control
against ABI-as-equivalence reasoning, **not a compiler correctness bug**.

## Concrete production seams that an H1 change would have to cross

At this checkpoint, `ClosedRegionAdmission.cpp:159` makes physical main FileID
ownership explicit; `:222` checks same-file source ranges; `:451` validates
definitions and `:468` limits header redeclarations to entry prototypes.
`helperCall` at `:720` requires a definition, binds only Value/Shape arguments,
rejects recursive/excessive expansion, and expands the body into the current
operation list. `:668` prohibits helper storage/effects. These are Clang/Sema
operations, not textual recognition.

`compiler/lib/mlir/MatcoreClosedRegion.h:17` carries only file-relative
offset/length/line/column in `SourceSite`; Program has one source identity.
`ClosedRegionAdmission.cpp:211` prefixes rejection diagnostics with that main
identity even when the spelling line came from a header. The observed header
refusal names `source.mdsl:2:7`, although the definition is in `math_library.h`.
That diagnostic limitation is real; removing `inMain()` alone is insufficient.

`ClosedRegionHostAdmission.cpp:195` binds parsed main bytes/FileID and trusted
canonical declarations; `:321` freezes, replays and checks the host closure.
`ClosedRegionHostInputs` already records dependency contents/metadata and offers
historical replay plus live freshness, but that does not automatically create
multi-file semantic sites or authorize helper definitions there.
`ExperimentalRegionEmitter.cpp:80` retires Value helper LLVM symbols from sealed
frontend bindings; helper bodies/sites participate in replay comparison at
`ClosedRegionAdmission.cpp:1020`. Cross-file identity and ownership must remain
coherent all the way through original-host compilation and helper retirement.

## What upstream supplies, and what it does not

The pinned upstream executable verifies and inlines `visible.mlir`; an external
declaration in `opaque.mlir` remains a call after `--inline`. Both add-body and
same-type subtract-body versions serialize to actual MLIR bytecode, round-trip,
and inline successfully; a wrong result type is rejected. Thus type checking
is active, but it intentionally does not establish mathematical equivalence.
These are upstream scalar function mechanics, **not a Matcore importer or a
separately compiled two-GEMM implementation**.

[Symbols and Symbol Tables](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)
provides named definitions/declarations, scoped lookup and visibility.
[Func](https://mlir.llvm.org/docs/Dialects/Func/) provides function/call interfaces,
typed direct calls in the same symbol scope and body-less external declarations.
[Bytecode](https://mlir.llvm.org/docs/BytecodeFormat/) provides versioned transport
and dialect compatibility hooks. Our inference: these are appropriate reusable
mechanisms, not a filesystem import resolver, source-authentication mechanism,
Matcore effect summary or transformation-legality theorem. No second function
IR, interpreter or custom bytecode container was introduced here.

## Minimum next boundary and falsifiers

First implement one source-visible pure header helper under a multi-file
authenticated source-site/closure model. Preserve the existing pure-helper
grammar and host isolation, rather than widening effects or public syntax at
the same time. For future separately distributed semantics, minimally bind
module/body identity and dialect version to value types, dimension relations,
per-operation numerical permissions/rounding, ordered required checks/effects,
and the actual implementation/import dependency closure. A pure math helper
can still fail; purity is not totality or permission to speculate away checks.

Three decisive falsifiers for either realization:

1. Replace a transitive helper/header with the same signature during capture or
   later artifact loading: stale evidence must not authorize the replacement.
2. Hide a storage observation or a failing nested GEMM: expansion/inlining must
   not lose the effect, required check, original file/call provenance or failure
   prefix. Preserve the selected adapter's stronger normal-return publication
   guarantee, not merely the generic dialect's partial-write allowance.
3. Swap a same-ABI body, operand order or numerical profile: fresh different
   source may describe a different valid program; stale imports/equivalence
   claims must not pass merely because symbols/types/bytecode still verify.

This is a feasibility recommendation, not completed module support or a reason
to pause the independent generated-execution/optimization campaign.
