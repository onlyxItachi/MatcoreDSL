# Parameterized output-pattern adversarial test scope

Initial scope: two new independent test fixtures, starting from clean main
`4c222dd68176de19fcf82696939b74be73f80968` in the isolated
`mdslc-work/pattern-generation-v1/integration` worktree. Existing files, build
integration, compiler implementation and target policy belong to other agents.
The integration owner subsequently authorized narrowly extending the existing
huge-empty block in `compiler/tests/closed_candidates/candidate_test.cpp`.

## Issuer and inspection falsifiers

`compiler/tests/generated_cpu/output_pattern_test.cpp` exercises the single
closed `OutputTilePatternV1` family at `1x1`, `2x3`, `3x5`, `4x16`, `7x2`,
`64x64`, `1x64` and `64x1`. It requires identical strict semantic/structured/
bufferized witnesses and different actual scheduled computation for distinct
parameter pairs, not
merely different manifests. Nonunit tiles must also differ from scalar LLVM;
upstream may legitimately fold the unit-tile schedule back to scalar LLVM.
Pairwise different tile parameters must produce different scheduled/LLVM
artifacts. Repeated issuance and reconstruction under the
wrong parameter pair are checked separately.

Invalid tuples include absent/partial-zero, negative, 65 and signed-64 extrema;
non-tiled/unknown schedule, unknown target/ISA and ARM/x86-ISA mixing cannot
borrow the pattern parameters. Successful and failed derivations must leave
their input untouched. Module/nested authority, noalias, library call, tile
bounds/step/start, tail clipping, subview stride/K offset/extent, reduction
iterator/maps, operands/destination, zero seed, arithmetic permissions,
accumulator/yield, output identity and symbol mutations are inspected. The
corpus separately counts well-formed corrupted IR.

Target-only specialization is normalized by removing only the known machine
triple, canonical leaf/wrapper renaming, target CPU/features and vector-width
attributes. LLVM computation must otherwise remain the same. This is an issuer
inspection test, not an ISA hardware/OS or source-authentication certificate.

## Independent generated-leaf execution oracle

`compiler/tests/generated_cpu/output_pattern_execution_test.cpp` directly
calls the issued three-descriptor leaf. Its scalar oracle uses separately
rounded volatile f32 products and a full increasing-K accumulator. Results are
bitwise equal except that NaN payload identity is not promised. Inputs,
descriptor objects and output canaries must remain unchanged as appropriate.

Coverage includes parameter-relative M/N tile boundaries, SIMD-width tails,
K31/32/33/63/64/65, all zero dimensions, positive-zero overwrite at K=0,
rectangular lane-distinct inputs, exact/partial A/B overlap in both directions,
FMA and split/re-zeroed-K counteroracles, signed zero, infinities, NaNs,
subnormal input/results and reproducible independent f32 bit patterns. Existing
`strict_isa_test_support.h` discovers hardware/OS capability before optional ISA
leaf entry. A skip is not physical target qualification.

`--corrupt-output` is a numerical negative control. `--oob` deliberately violates
only raw B capacity so generated-code ASan must diagnose the issued read; this
is not a legal runtime invocation or an adapter guard test.

## Source-adapter tile-step boundary

Read-only inspection confirms `closed_host_v1.cpp`'s existing `extent()` bounds
nonnegative signed dimensions and f32 count by `PTRDIFF_MAX/4`; `allocate()`
checks this before creating output storage. A nonempty output has M,N >= 1,
so M,N <= M*N <= INT64_MAX/4 < INT64_MAX-63 on the admitted Linux LP64 targets.
This is stronger than the raw leaf's M/N next-IV bound for every 1..64 tile.
`gemm()` tests empty output and then K0 before its production candidate switch.
No new source dimension restriction or tile-step failure guard is necessary.

The narrowly extended existing adapter fixture exercises automatic and forced
generated-strict choices when linked/available, with `(INT64_MAX,0)`,
`(0,INT64_MAX)` and a small nonempty K0 result. It asserts no leaf/provider
invocation, original full logical shape, positive-zero K0, earlier publication,
and exact GEMM/publication/observation/completion frontiers. Symmetric huge
nonempty K0 outputs retain the existing `extent_overflow`, previous immutable
result/effect prefix and sticky failure at the original GEMM frontier.
There is no callback, alternate runtime target or huge data allocation.

## Integration-owner validation results

This agent ran no builds or executable tests. The following are the integration
owner's reported local results after core commit `a5d94e9`, not independently
rerun results by this agent or a hosted/full-suite claim:

- The focused `generated_cpu.output_pattern` CTest scope passed **32/32**,
  **zero skips**.
- The issuer unit passed **242 checks**, **22 well-formed IR corruptions**,
  **zero failures**. The closed CLI test passed separately within that scope.
- All six normal generated-leaf executables passed **269 cases each**, with
  the following exact comparison/check counts and zero failures:

| Realization | Output tile | Normal check count |
| --- | --- | ---: |
| Baseline | 3x5 | 344782 |
| Baseline | 4x16 | 361755 |
| Baseline | 1x1 | 321149 |
| Baseline | 64x64 | 1279267 |
| AVX2 | 4x8 | 360083 |
| AVX512F | 4x16 | 361755 |

The six ASan counterparts, all deliberately corrupted-output expected-failure
controls and generated-read ASan OOB controls also passed in the 32-test scope.
No separate ASan comparison counts are supplied here. Optional ISA hardware/OS
guards returned available in this run; no optional ISA test was counted as a
passing skip.

At this handoff the integration owner's existing 120-test generated-CPU and
closed-candidate regression scope was still running, with no failure reported
yet. That scope, the full standalone suite, clean-build/source/ownership/replay
checks and any hosted acceptance remain the owner's separate obligations.
Physical CPU, GPU issuer/image inspection, GPU execution, sanitizer lanes,
unavailable refusal, hosted CI and performance are separate evidence categories.
No GPU scheduling, runtime/default dispatch, numerical permission, source/effect
or execution-authority expansion is intended here.

The integration owner reported the first issuer-unit run as 207 checks, 22
well-formed IR corruptions and one failed assertion: the test incorrectly
required unit `1x1` tiling to differ from scalar LLVM after canonical lowering.
The assertion was narrowed as above; the other checks were retained. Mixed-unit
`1x64` and `64x1` issuer tuples were then added. The 242-check passing rerun above
supersedes the initial failing run without erasing it. Final broader validation
belongs to the integration owner's recorded commands/results.
