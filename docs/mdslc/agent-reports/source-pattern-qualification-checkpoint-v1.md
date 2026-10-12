# Source-time pattern qualification checkpoint v1

**PROVEN WITHIN A BOUNDED CONTRACT:** authenticated single-TU source compilation
can issue, bind, link and execute a requested strict GEMM output-tile instance.
This extends the HOW artifact lifecycle, not the optimizer or execution policy.

## Exact integration identity

- Canonical start: `3077d9c3f5df1479f2fcadfcceeb9444018d6ba0`.
- Prior pattern engineering merge: [PR91](https://github.com/onlyxItachi/MatcoreDSL/pull/91),
  `c09a9d0e4384441238bf169e7e1122bf13ea5e23`.
- Qualified [PR93](https://github.com/onlyxItachi/MatcoreDSL/pull/93) premerge head:
  `598ee6ae3611de7521526592f3b3a597c569b2b2`.
- Normal engineering merge: `e8e8008cb61b61a585d8bc1584ca320c6326472d`,
  2026-10-12 09:31:27 UTC. Its parents are the starting main and premerge head
  above. No squash, rewritten history or branch deletion.
- Compiler tree: `86369ef73e4089f4117bbf8b52e4a0d0401e869f`, identical at corrected
  `877a5feb4f2e978315dd1ca6a6bbe586c9bc4613`, final head and engineering merge.

The [architecture challenge](source-pattern-adversarial-review-v1.md) found the
missing lifecycle: a flag or replacement object cannot override the private DSO's
intrinsically owned leaf. The winning design uses the existing in-process issuer
and a pinned PIC relink kit omitting only that leaf. The new isolated DSO retains
the original runtime/provider owner. Executable overrides, callbacks, exhaustive
tile catalogues, installed-runtime source compilation and submitted executable IR
were rejected. See the [contract](../SOURCE_TIME_GEMM_PATTERN_V1.md).

Bindings establish source/instance/LLVM pairing before publication; they do not
discharge runtime descriptor/capability/shape/numerical guards or authenticate
later dynamic loading. Inspection manifests confer no import authority.

## Local execution evidence

Coherent Clang/LLVM/MLIR 21.1.8 throughout, without toolchain installation. The
main Release test build requires authenticated OpenBLAS 0.3.32; the main Debug
ASan+UBSan and separate production-package builds disable OpenBLAS. Compiler
parallelism is at most two, with heavy links serialized.

| Scope | Source head | Outcome | JUnit SHA256 |
| --- | --- | --- | --- |
| Full Release | `877a5fe` | 303/303, no skips, 1249.20s | `f156d85dff193d81190e409a442221bdac0804fc3d90150933bb09d971fe0aef` |
| Source-pattern four + ordinary source-inaccessible package | `598ee6a` | 5/5, no skips, 150.34s | `03fc0f1457bd1f3b041041cbeafb412f204b520c8e469ce004e78cc0196b63e6` |
| Closed-region source/build-inaccessible package | `598ee6a` | 1/1, no skips, 158.28s | `c05ec1f70014345b00d8be834ae226c3fc282b45743493779f66ae0b84288a2e` |
| Selected Debug ASan+UBSan | `598ee6a` | 236/236, no skips, 1171.28s | `3e743242ffd142f7ac64f8297a5af43298f5db8d603fe645d9fbd8ca2e31977a` |

These JUnit files remain locally under `mdslc-work/source-pattern-v1/build-{release,asan}`:
`qualified-link-state-release.xml`, `final-head-source-package.xml`,
`final-head-closed-source-package.xml`, `final-head-asan-scope.xml`.
The full local Release run is not relabeled as final-head: only workflow budget
and documentation changed after its head. Final-head hosted Release independently
passed the complete 303-test suite below.

The final production package clones clean `598ee6a`, builds with BUILD_TESTING=OFF,
relocates its SDK, deletes only its disposable producer source/build trees, then
issues and executes 3x5 and 5x3 against the independent 68-case oracle. Actual
Transform operands, SCF tile steps and bound DSO/leaf linkage are checked;
SDK/bundle hashes remain unchanged during testing. Its installed driver SHA256:
`1bd969de06c284640dc87dd5ea899fcf6d806e80025e6dd34eafde0864404a2d`.
Package-test binaries are ephemeral, not retained release artifacts. This is not
installed-ASan or installed-provider qualification. Without the private kit,
ordinary no-request compilation still works while an explicit request refuses.

## Falsification and independent review

- Each of 3x5, 5x3, 1x1 and 64x64 executes 68 cases / 111292 numerical/effect
  checks. Tile operands and loop steps discriminate actual requested code;
  correct output alone is insufficient.
- Oracles preserve lhs/rhs order, rectangular noncommuting math, immutable
  snapshots, MAY-alias late reads, observations/publication and late failure
  prefixes, empty/overflow bounds, ordered K, separate non-FMA arithmetic,
  nonfinite/subnormal behavior and caller FP/errno state.
- The phase-mutation/binding runner reports 72 checks, including positive
  pairings and diagnostic/classifier controls. Refusals cover wrong source/entry,
  changed headers, same-source/different-instance publication, stale/substituted
  LLVM/object/DSO, worker/sanitizer mismatch, installation/output mismatch and
  path substitution. Failed cleanup preserves substituted foreign state.
- ASan controls require a report inside the actual source-issued private leaf
  and original-host instrumentation, not a separately compiled probe leaf.
- Review exposed input source/instance cross-wiring, then compiled-output/instance
  cross-wiring. Immutable input and typed compiled-result bindings close both;
  runtime guards and trusted-loader preconditions remain separate.
- Independent agent review accepted the original core/tests, corrected linker
  state at `877`, and immutable final `598`. Its final check confirmed identical
  compiler trees, the 236-test sanitizer selector, two-job build limit, per-test
  deadlines and unchanged ARM workflow. Separate evidence review checked the
  retained JUnit identities, package lifetime claims and authority boundaries.

The [independent tests](source-time-output-pattern-independent-tests-v1.md),
[static review](source-pattern-adversarial-review-v1.md),
[core decisions](source-time-output-pattern-core-v1.md) and
[historical integration log](source-pattern-integration-evidence-v1.md)
retain failed hypotheses and intermediate results.

## Final-head hosted qualification

All 26 PR93 check contexts were successful before merge. Counts below are
**actual passes + explicit skips / selected tests**, not execution claims for skips.
Native [PR run 38224233246](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38224233246)
and [push run 38224227372](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38224227372):

| Native profile | PR passes + skips / selected | Push passes + skips / selected |
| --- | --- | --- |
| Release, OpenBLAS required, MLIR | 303 + 0 / 303 | 284 + 19 / 303 |
| Release, OpenBLAS off, MLIR/vector readiness | 281 + 19 / 300 | 300 + 0 / 300 |
| Release, OpenBLAS off, row-contiguous | 282 + 19 / 301 | 282 + 19 / 301 |
| Release, OpenBLAS off, output-pattern | 299 + 0 / 299 | 299 + 0 / 299 |
| Release, OpenBLAS required, MLIR off | 84 + 1 / 85 | 84 + 1 / 85 |
| Release, OpenBLAS off, MLIR off | 83 + 1 / 84 | 83 + 1 / 84 |
| Debug, MLIR, OpenBLAS off | 279 + 19 / 298 | 279 + 19 / 298 |
| ASan+UBSan, MLIR, OpenBLAS off | 217 + 19 / 236 | 217 + 19 / 236 |
| TSan runtime | 4 + 0 / 4 | 4 + 0 / 4 |

The vector-readiness profiles additionally pass a separate 1/1
`mlir.structured.gemm-vector-readiness-v1` prerequisite. All four new source-pattern
tests actually pass in all 12 MLIR jobs. The closed-source-inaccessible package
actually passes in all 10 nonsanitizer MLIR jobs; ASan instead includes the
ordinary region install contract. The 19-skip group is packed AVX512 (1), strict
ISA AVX512F (13), output-pattern AVX512F 4x16 (5). Single-skip profiles skip packed
AVX512. These optional-ISA outcomes are not new target qualifications.

[Native ARM run 38224233389](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38224233389):
scalar 130/130, row-contiguous 132/132, output-tiled 130/130, all without skips.
Every profile executes the new source-pattern tests and relocated closed-source
package. Jobs: `114727687066`, `114727686720`, `114727686441`, respectively.

Existing Windows [PR run 38224233661](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38224233661):
Release 50+1/51, Debug 32+1/33, ASan 1+0/1. The two skips are packed AVX512.
[Push run 38224229167](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38224229167):
Release 51/51, Debug 33/33, ASan 1/1, no skips. This preserves existing Windows
support; it does not qualify the new source-time route on Windows.
[Legacy 38224233405](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38224233405)
and hygiene [PR 38224233387](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38224233387)/
[push 38224227505](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/38224227505)
also succeeded. Native logs/counts were independently inspected after completion.

Useful negatives were not waived: earlier ARM run `38218341835` refused a DSO
dependency mismatch before publication. Scoped linker push/pop corrects an
accidental global as-needed reset without weakening exact dependency equality.
Earlier hosted ASan jobs cancelled after 231/233 completed passes at 45 minutes;
the 60-minute budget retained all 236 tests/assertions and final runs completed.
Those earlier failures/cancellations remain negative or incomplete evidence.

## Preserved limits and next boundary

The route requires exactly one explicit `generated-strict` selection, native
baseline Linux x64/AArch64 single-TU executable generation, with `none` or
separately checked forwarding. `-c`, program mode, other candidates and
strict-fused-pair requests refuse. Stock runtime/DSO/default-leaf hashes remain
identical to baseline. All 28 retained NVVM/ROCDL single/fused issuer artifacts
match baseline bytes: compile-artifact regression, not new GPU execution.

No autotuning, performance claim, numerical relaxation, GPU schedule, submitted
executable IR, cache authority, per-operation/multi-source pattern selection or
API/ABI freeze. Installation relocation before compilation is tested; moving an
emitted executable/private bundle afterwards is unsupported. Execution still
requires the private bundle and pinned runtime installation at their emitted
paths. Trusted post-build loading is a precondition, not sandbox/crash atomicity.

The independent global SPIR-V Issue89/draft PR84 frontier is unchanged. A future
pattern-specific program-mode extension needs its own cross-TU source/instance
and link-ownership composition proof; this checkpoint does not implement it.
