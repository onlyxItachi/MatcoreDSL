# Private strict output-peel: integration qualification v1

Status: **MERGED; bounded private qualification complete** for
[PR #103](https://github.com/onlyxItachi/MatcoreDSL/pull/103), implementing
[Issue #102](https://github.com/onlyxItachi/MatcoreDSL/issues/102).
Normal canonical merge: `1f7c491f230251d04d42b5d4a0af5eaf29bcb685`.
All 26 hosted checks succeeded for the frozen private head. The earlier
18-success/eight-pending snapshot is retained as historical evidence, not the
final disposition. No source-time or performance qualification follows from
this private boundary. See the [final integration receipt][final-comment].

## Frozen engineering checkpoint

- Canonical merge `1f7c491f230251d04d42b5d4a0af5eaf29bcb685` has parents
  `52fb20a30cae0d001daebf5f71035ba071541181` and
  `dfc35f9e13653db16c7f9a825db2bb1b2a82ea22`; its tree is exactly the frozen
  private tree below. This is a normal merge, not a reconstructed/squashed patch.
- Private branch head: `dfc35f9e13653db16c7f9a825db2bb1b2a82ea22`, inspected
  with a clean tracked/untracked worktree. Its parents are
  `b68b0217dee77ae472e00199f5d0d7e8a20e7884` and then-current main
  `52fb20a30cae0d001daebf5f71035ba071541181`.
- Frozen source tree: `c07a780e28ed14b1669e4a50c898203cce22cac3`.
  Push logs check out the exact private head. ARM PR logs check out synthetic
  merge `cc15f169bb73fb29c6b29c4411aa0fa66c661d20`; the integration owner
  independently verified parents `[52fb20a..., dfc35f9...]` and the identical
  source tree. That synthetic merge is not the canonical engineering merge.
- Local evidence root:
  `/home/hamza-usta/mdslc-work/correctness-campaign-v1/build-peel-release`.
  The writer inspected existing logs/XML and receipts; no build, generated
  execution or timing campaign was started to prepare this document.

The [frozen private contract][contract], [implementation record][implementation]
and [independent fixture record][fixtures] describe the exact construction and
earlier checkpoints. Their initial pending/5x3-only statements remain historical
evidence; the additional completed scopes below do not rewrite those receipts.

## Material change and proof boundary

The additive private flag `--output-tiles-peel-n=M,N` selects checked tile values
in `[1,64]`, not source dimensions or numerical permissions. Pinned upstream
MLIR 21.1.8 tiles/generalizes/interchanges the canonical built-in strict GEMM,
then peels the outer N loop. For `T=tile_n` and `F=T*floor(N/T)`, the full domain
covers `[0,F)` with width T and the separate tail covers `[F,N)` without padding.
Both retain bounded M, complete increasing K, a single preceding +0 fill,
separate f32 multiply/add, original operand order and original C destination.
Unit N statically selects the exact old no-tail graph.

The independent envelope proves structured view lengths before alias folding,
then scalar ancestry, split/bounds, accesses, index dependencies, strict
recurrences, destination and effects after native lowering/folding. Exact
parameter-bound upstream replay is an additional defense, not a substitute.
Semantic mutants must fail with `peeled scalar envelope:`; a separate harmless
constant reaches only the replay diagnostic. Post-lowering footprint checks
remain closed and recipe-specific: one arithmetic copy for unit N/old recipes,
two only for the independently verified nonunit peeled recipe.

The existing justified LLVM noalias fact is C's aligned pointer only; A/B may
alias. It was already present in stock and old tiled artifacts. This work does
not add an alias assertion, private optimizer, importer or accumulator pass.
The P3 mechanism and its code-size/inlining confounds are not speed evidence.

## Completed local evidence

The clean frozen Release configure/build completed all **509 build steps**.
`frozen-full-ctest.xml` contains exactly **352 testcases**, all `status=run`,
with zero failures, disabled tests or skipped tests. The matching log reports
**352/352 passed in 1258.29 seconds**. These are test-suite elapsed times,
not benchmark results.

The separate `^generated_cpu_output_peel\.` gate passed **44/44**, zero CTest
skips: inspection/CLI plus six controls for each of seven profiles. Inspection
reports **627 checks, 56 valid-MLIR semantic mutants, zero failures and zero
unsupported upstream mutants**. It includes all 36 tile pairs drawn from
`{1,2,3,5,16,64}`. CLI reports **22 refused invocations and eight issued
parameter/target combinations**; target emission is distinct from execution.
Those 44 tests are also included in the 352-test full suite, not added to it.

Each profile ran normal and actual generated-ASan positives, both intentional
output-corruption detections, and separate tail/full-domain ASan capacity
violations. The negative probes deliberately violate the raw capacity
precondition and require generated-leaf heap-buffer-overflow detection; their
passing CTests do not mean those invalid calls execute successfully.

Per positive process, normal and generated-ASan each report:

| Tile MxN | Cumulative cases | Cumulative checks | Inapplicable short-K witnesses |
| --- | ---: | ---: | ---: |
| 5x3 | 1,618 | 558,043 | 80 |
| 3x5 | 2,122 | 600,197 | 110 |
| 2x2 | 1,618 | 455,753 | 80 |
| 1x1 | 1,450 | 413,671 | 70 |
| 1x64 | 2,626 | 1,612,944 | 140 |
| 64x1 | 1,450 | 2,239,248 | 70 |
| 64x64 | 2,626 | 19,599,248 | 140 |

Across the focused run's fourteen positive processes this is **27,020 cumulative cases,
50,958,208 checks and zero failures**. The 1,380 inapplicable short-K witnesses
are explicit exclusions, not executed passes or skipped CTests. Reused fixture
cases are already included: for example 5x3's 1,618/558,043 includes the first
printed 269/372,457 subset; do not add it again. Its strict participation counters
full=852, tail-width-one=426 and tail-width-two=426 overlap for mixed shapes.
Unit N has no N tail; tile N=2 has no width-two tail; tile M=1 has no M remainder.

The additional arithmetic corpus places order, non-FMA, rounding, positive and
negative subnormal/underflow, overflow, Inf/NaN-class and signed-zero witnesses
in full and applicable tail lanes, including 5x3 at N=4,5,7,8. It covers short,
odd/even and long K with sensitive terms at the end, plus lane-distinct
last-K-only and ordinary nonuniform rectangles. Valid raw empty/null-data and
K=0 complete-positive-zero cases, overlapping A/B, input immutability and output
canaries are separate checks. This is not execution of all 4,096 tile pairs.

## Hosted evidence and final disposition

Every row below was independently recounted from the named downloaded log's
actual `Passed` records and final CTest summaries, with no skipped/not-run
markers. Counts describe each selected configuration, not one universal suite.

| Completed hosted job | Retained log | Passed scopes | CTest skips |
| --- | --- | ---: | ---: |
| [Native Release, OpenBLAS required][release] | `hosted103-release-push.log` | 348 | 0 |
| [Native Debug, OpenBLAS disabled][debug] | `hosted103-debug-push.log` | 346 | 0 |
| [Native ASan+UBSan][asan] | `hosted103-asan-push.log` | 236 + 2 + 44 | 0 |
| [Native ARM scalar][arm-scalar] | `hosted103-arm-scalar.log` | 130 + 44 | 0 |
| [Native ARM row-contiguous][arm-row] | `hosted103-arm-row.log` | 132 + 44 | 0 |
| [Native ARM output-tiled][arm-output] | `hosted103-arm-output.log` | 130 + 44 | 0 |

The underscore-prefixed private gate is additive: the retained dot-prefixed
236-test sanitizer scope and separate two-test SPIR-V gate were not narrowed or
renumbered. ARM's separate 44-test runs are native execution evidence, unlike
the private CLI's AArch64 artifact emission alone. Other CI jobs, including
Windows regression, do not expand this private recipe's Linux target contract.
The six logs alone are not an all-checks summary. Separately,
`hosted103-final-status.json` records the exact frozen head and 26 completed
`SUCCESS` checks, including the [PR-native run][pr-native]. No pending, failed,
cancelled or skipped check is counted as successful. The earlier
`hosted103-status.json` remains the historical 18-success/eight-pending snapshot.

## Preserved artifacts and diagnostic history

The independent read-only receipt at
`/var/tmp/mdslc-peel-preservation-RtsNtC/audit.json` compares the retained P1
`build-release` and new `build-peel-release`: **84 canonical CPU files and
84 old unpeeled-pattern files are byte-identical**, with no changed or unmatched
files in those explicitly enumerated groups. These are LLVM, semantic,
structured, bufferized, Transform, scheduled and manifest files. **Objects were
not compared**, and recursive scratch-directory identity was not asserted.

The actual new `tests/generated_cpu/peel-5_3-normal.ll` is byte-identical to
`/var/tmp/mdslc-p3-peel-kiv52W/peeled-bound.ll`, SHA256
`7864ea0110a3567ad29c0f27d8a64369f05fb4f440625d457a4712b8021d3496`.
The actual execution tests link the closed issuer's new object, not that
research file or its inspection-only binder. Equality joins the mechanism to
production issuance; it is neither a source connection nor a timing result.

Independent review found and closed these pre-freeze issues:

- Pinned upstream peeling can fail at unit N; the final static identity branch
  avoids the operation rather than swallowing its failure.
- Alias folding can erase malformed unused subview lengths. The final proof
  checks slices before folding and actual scalar accesses afterward.
- Unused dynamic out-of-rank dimension queries and unproved semi-affine
  zero-divisor/index glue can be MLIR-valid but unsafe before DCE. The final
  complete dimension/index-dependency proof rejects them in the envelope;
  replay-only rejection does not count as their proof.
- Inherited 5x3 numerical N=129 is divisible by three and misses N tails. The
  added mixed widths, end-K witnesses, null-C empties and separate full-domain
  sanitizer probe make those blind spots explicit.
- The first focused build failed at two test-only MLIR constant-creation
  overloads. Typed attributes corrected them without changing test intent;
  `focused-build-first.log` is retained as failure evidence, not a passing gate.

## Evidence identities

Paths below are relative to the local evidence root unless absolute.

| Receipt | SHA256 |
| --- | --- |
| `frozen-configure.log` | `dd019bba6d32e7eeb4978f38eadcf53e27b37c11a77f3007b836d41a8957ab71` |
| `frozen-full-build.log` | `1e807072d6c7f952bd32efd4edf8ebdfb0806cb5b060d0642387cb5a93f029c0` |
| `frozen-full-ctest.log` | `9ea6e7dc5273277fce2fcc0732c74c00aab7abfeee25722f54267ba8c84c0dc8` |
| `frozen-full-ctest.xml` | `32bde12012130421cc921eab09c561b912395efb503e4ba0ed784c94ba9d27af` |
| `peel-family-controls.log` | `6dc7f199e1ed6a2afec964fbc7ef222a09e6d4303311c4856a67ccbfa65f69f7` |
| `peel-family-controls.xml` | `12b9bf3d7e2d13b58ff80b42cd272e27d797442647b3f48a60eb5afcae97eee2` |
| `hosted103-release-push.log` | `7fe9eb2db96ad86bb964dbc5d1c6cab626fbd4dcdf1830880cfa9250e14b30f0` |
| `hosted103-debug-push.log` | `80299e99895d12a836ef995df0095b4170e7b1990211234e5a16ac2280031173` |
| `hosted103-asan-push.log` | `1b30dacedfa2645d5d65fe8130b7b8f45d0fdb91f4197efcc4896cf9bfa21d6c` |
| `hosted103-arm-scalar.log` | `39a9fba8ddc4efecfe0f8308cb4290356712bb748be44aecdb20a63ee407b3de` |
| `hosted103-arm-row.log` | `b85ba284ba8037a94490cb5c9f35b908a2f61828eccf845dedfa07dfc59404af` |
| `hosted103-arm-output.log` | `e306b910523d7f48c270350578201419b8e785cc814f7255190e530ed909bdf5` |
| `hosted103-status.json` (historical 18 success / eight pending) | `545c7f806b82c26a6e3486f72e9ff1715318f61e9a1f5a1a8c70e2eeaa48dbac` |
| `hosted103-final-status.json` (26 completed `SUCCESS`) | `3c6d941ffe63dc26e90093630476e69af7353c951eb556bffdbb24d77459ea26` |
| `/var/tmp/mdslc-peel-preservation-RtsNtC/audit.json` | `5cb705291851f3a78cf451eb0a87ebf0335ed90659f54291d748222cf4acb370` |
| `/var/tmp/mdslc-peel-preservation-RtsNtC/audit.md` | `de0353c58c6e8ef9b4746efda6ebc8b681595e059dfa713497f5a8912b06e40e` |
| `focused-build-first.log` (failed historical build) | `40a8a012d132ebcbee1d6066fcf23d40d0a88b4ba01b9db712ceeb0e489ad86b` |

## Preconditions, exclusions and exactly one next boundary

Raw callers retain valid dense rank-two f32 descriptors, checked nonnegative
signed extents and addressable byte counts, live/race-free input storage,
compatible FP controls and private C disjoint from both inputs. A/B may alias.
Descriptor objects must exist even for empty data. Retain next-IV bounds
`M <= INT64_MAX-(tile_m-1)` and `N <= INT64_MAX-(tile_n-1)`; K is untiled.
These preconditions are not discharged by inspection metadata or byte equality.

This qualification is private native-baseline Linux x64/AArch64 on the coherent
Clang/LLVM/MLIR 21.1.8 tuple. Nonbaseline ISA, source-time selection/installed
execution, automatic/default dispatch, GPU mapping, arbitrary IR/object imports
and new source/runtime failure semantics are excluded. P1 stays
`INCONCLUSIVE_TIMING`; its retained bounded loss is neither retried nor
overturned by a register-accumulator code shape.

Exactly one next boundary is
[Issue #104](https://github.com/onlyxItachi/MatcoreDSL/issues/104): separately
explicit authenticated source-time selection with distinct request identity,
exact full/tail consumption, unchanged v1/default behavior, typed artifact and
publication binding, runtime-guard implication, source numerical/effect oracles,
generated sanitizer detection and relocated producer-inaccessible installation.
That later boundary is under qualification in
[PR #105](https://github.com/onlyxItachi/MatcoreDSL/pull/105), not merged source
support. It requires its own evidence; this record does not grant it source
authority or authorize a performance policy.

[contract]: https://github.com/onlyxItachi/MatcoreDSL/blob/dfc35f9e13653db16c7f9a825db2bb1b2a82ea22/docs/mdslc/PRIVATE_STRICT_OUTPUT_PEEL_V1.md
[implementation]: https://github.com/onlyxItachi/MatcoreDSL/blob/dfc35f9e13653db16c7f9a825db2bb1b2a82ea22/docs/mdslc/agent-reports/strict-output-peel-n-implementation-v1.md
[fixtures]: https://github.com/onlyxItachi/MatcoreDSL/blob/dfc35f9e13653db16c7f9a825db2bb1b2a82ea22/docs/mdslc/agent-reports/output-peel-adversarial-tests-v1.md
[release]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42737159785/job/129569078541
[debug]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42737159785/job/129569078519
[asan]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42737159785/job/129569078517
[arm-scalar]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42737168376/job/129569106182
[arm-row]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42737168376/job/129569106678
[arm-output]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42737168376/job/129569106534
[pr-native]: https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42737168351
[final-comment]: https://github.com/onlyxItachi/MatcoreDSL/pull/103#issuecomment-6757584731
