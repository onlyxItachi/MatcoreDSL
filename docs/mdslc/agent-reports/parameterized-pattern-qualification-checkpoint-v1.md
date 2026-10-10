# Parameterized GEMM pattern qualification checkpoint v1

Date: 2026-10-11. Scope: compiler-private CPU strict output-domain pattern,
not performance, general scheduling, a public format or new GPU authority.

## Provenance

- Starting canonical main: `4c222dd68176de19fcf82696939b74be73f80968`.
- Engineering PR: [#91](https://github.com/onlyxItachi/MatcoreDSL/pull/91).
- Exact premerge head: `4fde88b8c3e6b09bc88ce466a0bd28d9dafe6600`.
- Engineering merge: `c09a9d0e4384441238bf169e7e1122bf13ea5e23`.
- Qualified compiler tree: `eea7d0c25ac3dcc9197da5b9d701e13da6be6544`;
  AGENTS blob `504f01d536b38bcf96c936375c829c68d8b18c1b`;
  workflows tree `d303ddec9e1f7283fae732cfe9f9f5032064a172`.
- Focused commits: `a5d94e9` implementation, `e866903` independent fixtures,
  `cdb8660` interoperability/core audit, `4fde88b` source/install/CI integration.
- Canonical checkout was clean and left untouched during implementation;
  task work used a separate branch/worktree. SPIR-V #89/draft #84 was not changed.

## What is proved within the bounded contract

The same built-in strict GEMM semantic/structured/bufferized witnesses feed a
parameterized upstream Transform recipe. Each M/N tile is an integer in 1..64;
K stays full, increasing and scalar with separate f32 multiply/add. A structural
scalar envelope and exact parameter-bound replay reject altered coverage,
tails, indexing, arithmetic, destination, effects and authority attributes.
Target/ISA refinement remains separate. The source adapter's existing extent
bounds and empty/K0 bypass prove raw tile-step safety without a new failure
frontier. Private output, ordered effects and source/artifact ownership remain.
See [the contract](../PARAMETERIZED_GEMM_PATTERN_V1.md).

This is one reusable generator, not a general pattern optimizer. Production
selection is explicitly build-time; it is not source-time specialization,
automatic tile selection, GPU mapping or a measured cost model.

## Local execution on exact clean premerge head

Coherent LLVM/Clang/MLIR 21.1.8; Release; production `output-tiled` M=3/N=5;
authenticated OpenBLAS 0.3.32 enabled; GPU issuers compiled but no new real-GPU
execution enabled. Compiler jobs capped at two, heavy links serialized. No
system toolchain installation or LLVM build was performed.

| Scope | Result |
| --- | --- |
| New output-pattern tests | 32/32 passed; zero skips |
| Issuer/CLI | 242 checks; 22 well-formed IR mutations rejected; 17 CLI refusals/four accepts |
| Generated CPU/candidate/source selection | 120/120 passed; zero skips; 99.57 seconds |
| Complete clean standalone suite | 299/299 passed; zero failures/skips; 1053.18 seconds |
| Configure parameter controls | 14 exact-diagnostic refusals; four boundary accepts with BUILD_TESTING=OFF |
| Repository hygiene / whitespace | Passed |

Selections overlap and must not be summed. Each of six normal and ASan leaf
profiles ran 269 cases: baseline 3x5, 4x16, 1x1, 64x64; AVX2 4x8; AVX512F 4x16.
These are qualification points, not exhaustive runtime coverage of all 4096
parameter pairs or every shape; structural checks apply to each issued instance.
Actual local ISA guards admitted both ISA-specific profiles. Independent volatile
increasing-K arithmetic checks finite/zero results bitwise and NaNs by class,
not payload. Cases include noncommuting rectangular/tail shapes, zero extents,
input overlap, subnormals, non-finites and discriminating FMA/reduction-order
counterexamples. Deliberately corrupted output and actual generated-code ASan
out-of-bounds controls passed; expected diagnosis is not an unqualified crash.

The complete suite includes source authentication, ordered failure/publication,
provider partial failure, huge-empty/extent overflow, private artifact ownership,
production BUILD_TESTING=OFF installation and relocated source/build-inaccessible
execution at exact head `4fde88b`. Local JUnit is retained outside the repository
at `mdslc-work/pattern-generation-v1/build-release/full-release.xml` under the
workspace root; SHA-256
`506ee68a5a86da60768ea673a864b1a7fbcfa534bd15c7628e56ae204a6d620c`.

Prior clean issuer `9f7f9af` has relevant source identical to starting main.
Seven retained CPU configurations produced 49 byte-identical LLVM/MLIR/manifest
artifacts; isolated and fused-pair NVVM sm_89/ROCDL gfx1150 produced another 28
byte-identical artifacts. This is regression evidence, not renewed physical
GPU qualification. Object inspection found separate YMM multiply/add for AVX2
4x8 and ZMM multiply/add for AVX512F 4x16, not a speed result.

## Hosted qualification

All 26 engineering checks succeeded before the normal merge; no pending or
failed engineering check was waived.

All runs are associated with premerge head `4fde88b`. Push checks checked out
that commit. PR checks used synthetic merge
`0c3a4d61b2c4454b33169b855511860b97c8b052`, whose parents are starting main
`4c222dd` and head `4fde88b`; its complete Git tree
`c6f3234d08645943fd8687c8cd5a33e984549be9` is identical to the head's tree.
The final engineering merge is checked against those same qualified trees.

Linux [PR run 38091519342](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38091519342)
and [push run 38091516320](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38091516320):
counts below are **selected / executed-and-passed / skipped**, not a claim that
skips executed. Every completed row has zero test failures.

| Linux scope | PR | Push |
| --- | --- | --- |
| Release, MLIR off, OpenBLAS off | 84 / 83 / 1 | 84 / 83 / 1 |
| Release, MLIR off, OpenBLAS required | 85 / 84 / 1 | 85 / 85 / 0 |
| Release, MLIR on, OpenBLAS required | 299 / 299 / 0 | 299 / 280 / 19 |
| Release, scalar, OpenBLAS off | 296 / 277 / 19 | 296 / 296 / 0 |
| Release, row-contiguous, OpenBLAS off | 297 / 297 / 0 | 297 / 278 / 19 |
| Release, output-tiled 3x5, OpenBLAS off | 295 / 276 / 19 | 295 / 276 / 19 |
| Debug, MLIR on | 294 / 275 / 19 | 294 / 294 / 0 |
| Focused global ASan+UBSan | 232 / 213 / 19 | 232 / 232 / 0 |
| Focused TSan runtime | 4 / 4 / 0 | 4 / 4 / 0 |

Both scalar Release jobs also passed the separate one-test vector-readiness
selection. Every MLIR Release job passed its separate BUILD_TESTING=OFF
production-build/manifest proof. All applicable native gates registered exactly
32 new pattern tests; the sanitizer selection registered exactly 232 tests.

The one-test skip is existing `runtime.cpu.packed_avx512`. The 19-test skip set
is that test plus 13 existing strict-ISA AVX512F controls and five new
`output_pattern.avx512f_4_16` controls, all unavailable hardware qualifications.
Such runners executed the other 27 pattern tests. All 32 pattern tests physically
passed in push ASan, push scalar Release, push Debug, PR row Release and PR
OpenBLAS-required Release, as well as locally. No skip was converted into support.

Native [ARM64 run 38091519285](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38091519285)
passed all three fail-closed source/runtime qualifiers: scalar 126/126,
row-contiguous 128/128, output-tiled 3x5 126/126, zero skips. Each executed all
22 ARM-applicable new pattern controls. The extra two row tests are its existing
SIMD inspection/sanitizer controls. Artifact authentication checked real native
AArch64 objects and, for tiled, exact M3/N5/untiled-K manifest. Artifact IDs are
scalar `11684876021`, row `11684462140`, tiled `11685020617`.

Windows [PR run 38091519303](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38091519303)
and [push run 38091516324](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38091516324)
passed the unchanged compatibility workflow, relocated consumers, distribution
checks, native pipeline and ASan scope. PR Release selected 51, passed 50 and
skipped packed AVX512; Debug selected 33, passed 32 and skipped the same test;
ASan passed 1/1. Push executed all 51/33/1 respectively with zero skips.
This does not qualify generated regions on Windows.

Legacy [PR CI 38091519298](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38091519298)
and both repository-hygiene runs
([PR](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38091519308),
[push](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38091516246)) passed.
No engineering run was cancelled, retried or given a reduced test scope.

## Independent review and negative findings

The independent core reviewer statically accepted `a5d94e9`, then rechecked
the unchanged four core files and final CLI/CMake/install/CI wiring at exact
`4fde88b`. No unresolved blocker was identified in that bounded scope. The
reviewer ran no executable tests; static guard/FP/storage reasoning does not
replace local or hosted execution. See [the immutable core audit](parameterized-pattern-independent-core-audit-v1.md).
An independent fixture lane authored the mathematical/corruption controls and
later independently authenticated the complete local JUnit, all 32 controls and
the installed 3x5 artifact/producer-removal evidence. Final static and local
review is also preserved in [PR91's review record](https://github.com/onlyxItachi/MatcoreDSL/pull/91#issuecomment-6103077733).

Retained construction/test failures: the first verifier rejected valid affine
normalization/unit loops, corrected by scalar inspection without canonicalizing
away unit loops; an initial test wrongly demanded different final LLVM for a
1x1 tile, corrected because canonicalization may legitimately erase that
decomposition. Derived scheduled IR still must genuinely differ. Wiring review
caught and resolved a tile macro mismatch and stale workflow expectations/count.
None was fixed by broadening arithmetic, source authority or admitted effects.

## Boundaries after integration

No performance/BLAS-parity result, optimal schedule, matrix instructions, K
tiling, padding, new fusion, residency, GPU tile mapping, runtime policy or source
syntax follows. Existing default and native/provider/GPU routes remain intact.
Windows regression is not generated-region Windows support. Interchange remains
research-only: [official-interface audit](../COMPILER_INTEROP_BOUNDARIES_V1.md).

Pattern-specific follow-up: source-time selection and private ownership of a
bounded requested instance through the same issuer, without an implicit cost
model or expanded numerical permission. It was not started. The independent
SPIR-V #89 engineering frontier and draft #84 are unchanged; no qualification
transfers between these workstreams. Issues #15/#20 remain open.
