# Independent audit: alias and register/FMA measurements

Overall assessment: **SHARE WITH CAVEATS**. The raw evidence supports engineering
a separately gated reassociate-profile generated candidate. It does not support
replacing strict execution, selecting a default, provider crossover thresholds,
or Native BLAS Parity. No measurement discrepancy or frozen-artifact mismatch
was found. The auditor authored the earlier register prototype, not this timing
harness or run; this is independent recomputation, not an independent review of
that prototype's implementation.

## Identity, grain and method

- Raw artifact, unchanged: `/home/hamza-usta/MatcoreDSL-wt-mlir-reassociate-execution-evidence-v1/benchmark_reports/hpc-alias-register-comparison-v1/evidence.json`.
- Raw SHA-256: `1abb973b37dd75c1c4fe10bafd8e03e0a0d8a77f6f51b5a0a5c08668639fd833`.
- Frozen harness: `62c6237e0396084cc3b96f16548f24577794edfa`, clean both in recorded state and at audit.
- Started `2026-09-09T00:27:31.887415+00:00` (UTC).
- Eleven lanes, six shapes, three rounds, three samples per round: 198 timing
  batches, 594 batch-mean samples, 66 lane/shape medians. Each sample averages a
  power-of-two repetition count; the median is over nine batch means, **not**
  nine individual calls or independent machines.
- CPU 2 pinned on the local AMD Ryzen AI 9 HX 370; performance governor retained.
  All agents/builds stopped for the run; compiler preflight was empty. SMT sibling
  14, desktop/system tasks, thermals and clocks were not isolated. Lanes are
  deterministically shuffled per round; shape order remains fixed. Buffers are
  warm and reused. No cold-cache, multithread, scaling or full-envelope evidence.

The validation/data-quality skills prompted explicit separation of layer,
numerical profile, ISA and exact artifacts before considering ratios. No charts
are needed for six exact mappings; table entries below are independently
recomputed, not copied from a narrative summary.

## Checks and outcomes

All **eight source hashes and 30 artifact hashes** match current bytes, including
all compared output executables. Each lane's executable hash agrees with the
top-level artifact record. The driver and provider identities are frozen inputs;
this does not extend the product's trusted-loader contract into universal
runtime loading authentication.

All 321 command records reconcile:

| Recorded activity | Verified count/outcome |
| --- | --- |
| Non-execution setup, compilation and inspection | 57, all exit 0 |
| Positive execution lanes | 11; 240 cases = eight strict lanes × 24 + three reassociate lanes × 16 |
| Output corruption controls | 11, each exit 4 and the expected oracle mismatch diagnostic |
| Shape/access controls | 21 + 21, exact failure/prefix checks in harness, exit 0 |
| Forced strict incompatibility | Two executables × 24 cases, exit 0 |
| Timing commands | 198, every command pinned to CPU 2, all exit 0 |

Every timing command's three sample records equal the corresponding raw timing
entry. All `(lane,shape,round)` keys are unique, all groups contain exactly nine
samples, and every median/minimum/maximum matches recomputation. Positive
repetitions/times and sample IDs were checked; `total_ns/repetitions` agrees with
printed ns/call within its 0.001 ns rounding. The largest nine-sample max/min
spread is 12.15% in the source forced-reference 512-square lane. No values were
removed as outliers or pooled into an overall cross-shape speedup.

## What the numbers actually say

Median microseconds per call. The first four columns are **standalone
primitives**, excluding source-region snapshots, validation, FP adaptation and
publication. The last column intentionally shows the different **source
end-to-end** provider layer as context, not an equal-overhead kernel comparison.

| M×N×K | Old strict row | Output-noalias strict row | Strict row, O3/v3 | Register/FMA, O3/v3 | OpenBLAS source route |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16×16×16 | 0.460 | 0.448 | 0.499 | 0.108 | 0.724 |
| 128×128×128 | 125.876 | 125.534 | 99.237 | 48.836 | 31.232 |
| 512×512×512 | 7,965.197 | 7,960.114 | 5,296.839 | 4,519.061 | 1,958.656 |
| 128×256×64 | 127.214 | 125.175 | 80.062 | 46.489 | 40.084 |
| 16×512×256 | 123.413 | 123.524 | 79.166 | 70.243 | 48.473 |
| 65×67×63 | 24.912 | 24.010 | 18.509 | 12.291 | 4.878 |

**OBSERVED:** the alias-fact correction changes these primitive medians by
−3.62% to +0.09%; most shapes are effectively neutral in this narrow experiment.
It remains justified as correct fact preservation, not a large performance win.

**OBSERVED:** the register/FMA object takes respectively 21.70%, 49.21%, 85.32%,
58.07%, 88.73% and 66.41% of the matched O3/v3 strict-row time. It wins all six
measured primitive comparisons. Matching ISA removes the obvious baseline-v3
confound, but register blocking, loop order, memory traffic and numerical
permissions still change together. This is not causal attribution to FMA alone.

**OBSERVED negative evidence:** the register primitive loses to the **source**
OpenBLAS route on five of six shapes, by factors 1.56, 2.31, 1.16, 1.45 and 2.52
for the final five rows. That is despite excluding region overhead on the
register side. Providers remain important; the 16-square win is not a measured
source-level crossover or sufficient input for a planner threshold.

## Numerical, layer and ISA boundaries

The strict lanes retain all 24 controls. The explicit reassociate lane uses the
16 ordinary, exactly representable dyadic cases; it does **not** pretend to pass
the eight strict rounding/underflow controls. The separate
[prototype study](mlir-reassociate-register-v1.md) exercises the chosen
full-tile fused/tail-unfused realization, IEEE-special values, aliasing and
generated ASan. Its deliberate strict-result refusal is preserved. The two
evidence sets complement one another; this benchmark alone does not establish
the full reassociate contract or source execution authority.

The strict-v3 object is `ba3af011aaa43f48dcc68513f80b03bb9181990e22cf836cba4a86bc96b81466`,
compiled by the parent from pinned output-noalias `strict-normal.ll` with Clang
21 `-O3 -march=x86-64-v3 -ffp-contract=off`. That LLVM source hash also matches the
raw record. Independent disassembly shows separate vector mul/add; the register
object `453ed371fb64dc7df300b341eb8a08083ed683b3c976f9f7c9008aa496867eb9` contains
actual vector FMA. Neither implies that a globally v3 product is supported.

Raw `primitive-baseline` means the **new output-noalias row** object in this run;
`primitive-row` means the older row object. Names must not be reused as a claim
that the former is the old scalar algorithm. Source and compiler build commits
differ intentionally and the raw report lists those differences; artifact
identity is not collapsed into the harness HEAD.

`existing-native` is explicitly `FORCE_REFERENCE_V2`, with the adapter requiring
stable ID `cpu.reference.f32.v1` and one actual thread
(`compiler/lib/runtime/closed_host_v1.cpp`, `legacyGemm`). Its slow timings say
nothing about the legacy packed/AVX2 native alternatives. Public Result does not
expose candidate identity telemetry; the adapter's implementation establishes
this forced policy, not the benchmark's public result object.

## Decision and falsification boundary

**INFERRED:** this is sufficient evidence to prioritize one bounded generated
reassociate candidate over more speculative strict cache-tile tweaks. Preserve
the current strict candidate and external provider paths. Candidate engineering
must independently gate source profile and ISA, preserve private-output and FP
scope/retirement guarantees, reject strict use before effects, and execute real
source programs before claiming a new product execution path. No thresholds,
autotuning, defaults or Issue #15 closure follow from this run.

Required next falsifiers include strict-profile refusal at the correct frontier;
full/tail and K=0 source execution; overlapping A/B with private C; failure after
earlier publication; and actual generated instrumentation. Re-run same-layer
source comparisons after integration, retaining the provider losses. A future
sample showing no benefit may reject a policy/default decision without denying
that the upstream register-retained lowering is mechanically viable.

Audit recipe: parse `evidence.json`; hash every path in `source_sha256` and
`artifact_sha256`; group `timing[*].samples[*].ns_per_call` by `(lane,case)`;
assert three unique rounds, three sample IDs per round and nine samples per
group; recompute `statistics.median`, `min`, `max` against all 66
`timing_summary` entries; compare each timing-command JSON stdout sample list
against the matching ordered raw `timing` record. Inspect source
`oracle_profile`, `Data`, `valid_outcome`, and `legacyGemm` for the contract
checks summarized above. This audit ran no builds, tests or new timings and
modified neither the harness nor the raw evidence.
