# Source-time peeled output pattern: independent test lane

Starting checkpoint: `dfc35f9e13653db16c7f9a825db2bb1b2a82ea22`'s private realization.
Status: fixtures authored; Python synthetic controls passed. **No compiler
build, new source compilation, native execution, sanitizer run or timing was
performed by this lane.** Root owns those gates and their retained receipts.

## Scope and reuse

Only `compiler/tests/source_pattern_peel/` and this report belong to this lane.
The original `compiler/tests/source_pattern/run.py`, fixtures and frozen P1
performance plan/runner remain unchanged. The new runner imports the old helper
under a unique module name from the sibling `source_pattern/run.py`, without
mutating its globals. Both directories may survive producer deletion together;
no absolute producer path is required.

The old connected source fixture supplies its unchanged 68-case independent
scalar/effect oracle, FP restoration, aliases, empties, ordered late failures,
canaries and owning observations. The old forwarding source fixture supplies
the actual checked `readForwarded` discriminator. The unchanged same-Value and
RHS-carry fixtures cover both noncommuting operand roles for each nonunit request.
The new narrow main-file
`end_k.mdsl` region reuses the existing mathematical helper and verifies source
publication plus owning observation against frozen analytic f32 bits and
last-K-only lane-distinct exact integer products. It is not a second region
compiler, broad oracle architecture or source grammar.

## Receipt and consumption discriminator

`run.py` exposes
`validate_bundle(binary, tiles, region, optimization, sanitized, invoke)` returning
`(manifest_fields, file_hashes)` for the installation tests. Its helper dependency
is only the surviving sibling old runner.

The new source schema is exactly `matcore-source-output-peel-n-v1`, with the old
field set plus `realization=output-tiled-peel-n-mkn`. The independently recomputed
length-delimited binding uses `matcore-source-output-peel-n-binding-v1` and the
exact manifest prefix before the binding field. Duplicate/missing/unknown fields,
wrong domain, changed request, wrong schema/realization and false authority claims
refuse. The old manifest parser is separately required to reject the new schema.

The bounded canonical-emission classifier connects the actual root matmul
match to output tiling, generalization and M/K/N interchange. Nonunit N must peel
the actual N result handle after interchange; unit N must not peel. Scoped,
definition-before-use SSA resolution identifies the shared M loop, requested
increments, `F=tile_n*floor(N/tile_n)`, full `[0,F)` and tail `[F,N)` loops,
original A/B/C slices, bounded M extent, fixed full width and `N-origin` remainder
width consumed by each actual generic. Sibling SSA reuse is handled separately;
neither comments nor unrelated counts can supply a missing relation.

This is a consumption discriminator for pinned canonical emission, **not an
MLIR parser, complete arithmetic verifier, importer or execution authority**.
Production independent envelope/replay checks retain that responsibility.
Synthetic generic bodies deliberately need not be valid MLIR and are never
advertised as valid-IR semantic mutants.

The bundle inspector also joins all retained stage/object hashes, the private
issuer's peeled schedule/pattern, exact source request/profile, actual executable
dependency, DSO hash, unique local leaf/wrapper, intrinsic symbolic binding and
canonical Runtime ownership. It rejects stock/foreign candidate dependencies,
exported private leaves and compiler/provider ownership leaks. These observations
do not self-authenticate post-publication loading or create artifact authority.

## Planned native controls, not yet results

- New explicit 3x5, 5x3 and 5x1 source instances execute the unchanged connected
  fixture and its corruption control. New 3x5/5x3 retain identical semantic,
  structured and bufferized WHAT witnesses, but distinct consumed HOW artifacts.
- Old/new 5x1 instances intentionally retain identical Transform, scheduled,
  LLVM and object computation; their source binding, SONAME and DSO bytes must
  differ. The unchanged old graph-only helper may accept the identical unit-N
  graph. Its complete manifest gate rejects the new schema. Frozen P1 has no
  unit-N role: this does not claim a P1 unit-N bypass or reinterpret its evidence.
- New 3x5/5x3 `source_peel_end_k` programs take the requested tile integers as
  host test arguments. Sensitive terms occur at the end of K for K=1..6,33,65,
  M=tile_m/tile_m+1, and N=tile_n+1/tile_n+2/2tile_n+1/2tile_n+2. Both full and
  width-one/width-two tails are source-observable. Frozen constants distinguish
  FMA, increasing K, RNE, subnormal input/product, underflow, overflow, infinities,
  NaNs and positive-zero seeding; every padding product in the zero witness is
  negative zero. NaNs are class-only; all other expected bits are exact.
- The exact expected end-K receipt is 792 cases, full=728, tail1=364, tail2=364,
  inapplicable-K=40, zero failures. Branch participation overlaps and must not be
  added as disjoint cases. Short-K inapplicable witnesses are not passing tests.
  Output checks also cover input immutability and canaries; publication storage
  is overwritten before its owning observation is compared.
- Source `publication-read-forwarding` is tested with the unchanged eligible
  source oracle and actual undefined `readForwarded` symbol discriminator.
- Malformed/unsupported/mixed/duplicate requests, both old/new orderings,
  nonbaseline/other candidates, imports, compile-only, program and fused-pair
  requests must refuse without executable/private-bundle publication.
- Sanitized runs compile the noninstalled new raw probe, which reuses the old
  source probe's exact ABI/loader vocabulary. It deliberately omits the last B
  element with N=1 (tail) and N=tile_n (full), then requires the ASan heap-buffer
  diagnostic in the actual source-owned generated DSO leaf. This violates the
  private raw capacity contract; it is not a supported source call or callback.

The new runner accepts the old-style `--driver`, `--output`, `--sanitized`,
`--cxx`, `--runtime` and `--case connected|refusals` arguments. Root owns CMake,
workflows, existing typed mutation tests and relocated/producer-inaccessible SDK
execution. Sanitized relocation also needs the sibling old `raw_asan_probe.cpp`
because the new probe includes it; ordinary validation needs only the two runners.

## Local synthetic observations

`PYTHONDONTWRITEBYTECODE=1 python3 compiler/tests/source_pattern_peel/runner_test.py`
passed **15/15 unittest methods**, 0.004 seconds, without compiler or native
program invocation. Controls include wrong-root/N handle, wrong split/step/axis,
gap/overlap, swapped or unconsumed slices, wrong width, missing/extra peel,
comments, missing sibling SSA, forward definitions, receipt/domain attacks and
missing end-K participation. Independent source review found that collecting
slice names from a whole domain could accept text with views moved after their
generic use. The final classifier requires each definition's identity in the
generic entry snapshot, and a synthetic forward-use attack must refuse. Exact
Transform enclosure/yield-order controls were added too. Initial 13-method
controls and an intermediate corrected 14-method run also passed; an
intermediate 14-method run failed four positive controls when an overbroad filter
counted the named-sequence container as an instruction. Narrowing that filter
fixed the test-harness bug; it was not a generated compiler failure.

During drafting, offline classification of retained PR103 emitted Transform/scheduled texts for
3x5, 5x3, 64x1, 1x64, 1x1 and 64x64 succeeded. This was inspection of previously
produced files, before final scope/order hardening, not a new compilation or
source-connection pass. Native/ASan,
installed/ARM, full regression and hosted results remain pending. No timing,
optimization benefit, default change or permission to alter retained P1 follows.
