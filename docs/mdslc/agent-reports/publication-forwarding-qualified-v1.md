# Publication-to-read forwarding: qualified canonical checkpoint

Date: 2026-10-04. Status: normally merged bounded engineering boundary,
not broad fusion, residency, new operation support or a performance result.

## Exact identities

- Canonical engineering merge:
  `355e281fe0de4e54c03243d332eafbbe2095591b`,
  [PR #78](https://github.com/onlyxItachi/MatcoreDSL/pull/78), merged
  `2026-10-04T15:57:20Z`.
- First parent / prior canonical main:
  `43ce3cb2557aad4dc3b810ea715c00531c76628f`.
- Qualified premerge head / second parent:
  `20c8473fae24d7396d384612064e9b69be0a0c29`.
- Compiler subtree, identical at premerge head and canonical merge:
  `048bcfe3eeef63364070dc7f3ac618dd01f7ce5b`.

This documentation checkpoint identifies that engineering merge, not its own
later documentation commit. The [contract](../PUBLICATION_READ_FORWARDING_V1.md)
and [independent report](publication-forwarding-independent-v1.md#exact-commit-forwarding-review)
retain implementation/proof ownership and reviewed commit identities.

## Qualified semantics and authority

Explicit `--optimization publication-read-forwarding` is separate from candidate
selection; `none` remains the default. Only a compiler-issued immutable in-process
plan can authorize reuse, after replaying the unchanged sealed Program and exact
untransformed paired witness. Consumption recomputes and checks the complete
bounded records. A diagnostic proposal, serialized attribute or imported artifact
does not issue execution authority.

The retained value follows a dominating successful publication to the same sealed
resource/version. Every publication invalidates all older MAY-alias bindings;
branch entry/join are conservative barriers. Original requested extent/shape and
descriptor guard order, source sites/frontiers, immutable ownership, owning
observations, caller FP state and earlier successful effects remain required.
Only the later snapshot allocation/copy opportunity is removed. Candidate policy,
per-GEMM arithmetic/rounding, public APIs and Session/Value layout are unchanged.

## Executed local qualification

Coherent Clang/LLVM/MLIR 21.1.8 Linux x64 Release execution at the qualified head:

| Distinct scope | Outcome | Saved owner evidence |
| --- | --- | --- |
| Full regular standalone scope, excluding only the two separately run disposable-package builds | 391/391 passed, zero failures/disabled/skipped; 1008.20 s | `forwarding-full-main.log` and `.xml` |
| Both disposable source/build-inaccessible package tests | 2/2 passed, zero failures/disabled/skipped; 172.31 s | `forwarding-full-packages.log` and `.xml` |

Combined coverage is **393/393 distinct registered tests with no skips**,
not one 393-test CTest process or a sum with repeated focused runs. The package
tests include relocated installed-driver compilation/execution without their
disposable producer source/build trees; the installed log pins the exact head.
The documentation agent inspected logs/XML and identities; it did not rerun
builds/tests. Execution belongs to the integration owner.

Final regular-run XML records 67 independent runtime checks and 130 derived-plan
checks, both zero failures. Every local forwarding source policy reports six
executed cases, zero refusals and zero oracle failures in BOTH optimization modes:

- Native-strict, generated-strict, automatic and the two-TU program: 119 checks
  per mode.
- Generated AVX/AVX2/AVX512F, reassociate, existing-native, OpenBLAS, NVVM and
  ROCDL: 118 checks per mode. Reassociate/provider fixtures explicitly carry the
  required existing per-operation permission; candidate choice does not invent it.

Actual optimized-only `readForwarded` undefined-reference gates distinguish the
two executables. CPU/provider execution and physical RTX 4060 Laptop `sm_89` /
Radeon 890M `gfx1150` forwarding execution remain bounded by the existing target
qualification; hosted optional-candidate refusals are not device execution.

Raw local evidence is under
`/home/hamza-usta/mdslc-work/region-optimization-v1/evidence/`.
Frozen log SHA-256:

```text
forwarding-full-main.log     0c16393661feac013fcfa0904a47746017b6fe253b2c6773854d7993b6e04a5f
forwarding-full-packages.log da66104b98a5e5eb23bc6cde204352b976484ed858e9b0529c93ae03a800253b
```

## Exact-head hosted qualification

The following completed pull-request runs all report head
`20c8473fae24d7396d384612064e9b69be0a0c29` and successful conclusions. The eight
native matrix jobs plus Windows, both ARM64 jobs, hygiene and legacy total
13 successful jobs in these five runs; earlier duplicate runs are not counted.

| Hosted lane | Exact run / outcome |
| --- | --- |
| Native Clang 21.1.8 | [37212090477](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37212090477): all eight jobs successful, covering Release OpenBLAS ON/OFF and MLIR ON/OFF, row-contiguous, Debug, ASan+UBSan and TSan's separate runtime scope. |
| Windows x64 clang-cl 21.1.8 | [37212090511](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37212090511): successful existing standalone Windows lane, not generated-region Windows support. |
| Native ARM64 strict | [37212090518](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37212090518): scalar 103/103 and row-contiguous 105/105 passed; each includes connected native/generated/automatic forwarding and program tests. |
| Repository hygiene | [37212090481](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37212090481): successful tracked-generated-output rejection check. |
| Legacy Python/JIT | [37212090452](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37212090452): successful `build-and-test` job. |

Hosted conclusions/heads and the ARM64 test totals were checked through GitHub's
API/logs. The [independent ACCEPT](publication-forwarding-independent-v1.md#exact-commit-forwarding-review)
is bounded review, not an independently rerun execution result or a formal
all-program theorem. Review, local execution and hosted evidence remain distinct.

## Preserved negatives and next boundary

The initial focused run passed only 2/14: runtime/plan checks passed, while driver
harnesses failed. Reserved CMake `POLICY` parsing and semicolon-list splitting
were test-only defects, not passing source execution. The original strict-FP
multi-source direct call was refused because its `strictfp` effect attribute
differs from the exact witness. Both-mode refusal is retained; the positive
fixture uses an admissible ordinary wrapper without relaxing the production
comparator. See [initial failures](publication-forwarding-v1.md#first-coherent-integration-run-retained-failures-and-fixture-correction)
and [independent correction/strict-FP analysis](publication-forwarding-independent-v1.md#integration-harness-failures-and-bounded-corrections).
Historical pending snapshots in those detailed reports are not retrospectively
rewritten; this record supplies their final integrated qualification.

[#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15) remains partial/open;
[#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.
No benchmark, broad fusion, device residency, new numerical permission,
general rank-N or source authority from transformed IR follows.

Exactly one next frontier: **compiler-issued strict two-GEMM CPU row-panel
derivation**, with an isolated upstream scheduling/bufferization/LLVM and actual
object/ASan proof before any source/runtime execution authority. Preserve each
increasing reduction, separate f32 operations and intermediate rounding; prove
bounded caller-owned panel storage and retain full original logical checks.
