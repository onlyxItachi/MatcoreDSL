# P5: authenticated source-requested full-N/tail comparison

Disposition: **INCONCLUSIVE_TIMING. The primary and every secondary comparison
are unqualified. No speedup or slowdown ratio is reported.**

One preregistered collection ran on 2026-11-10 and exited 78 after 27.280753 s.
Its complete correctness, identity and chronology audits passed; its timing
acceptance did not. There was no retry, sample rescue, estimator substitution,
new tile search or automatic policy change. This completes the bounded evidence
collection, not a performance qualification or [Issue #15](https://github.com/onlyxItachi/MatcoreDSL/issues/15).

## Exact question and qualification gate

The primary was explicitly requested peeled 5x3 versus the old 5x3 request at
M/K/N = **33/35/37**: multiplying 33x35 by 35x37. Five roles used one physical
source and one relocated, uninstrumented, baseline Linux x64 installation:
stock `scalar-mnk`, old/peeled 3x5, and old/peeled 5x3. N=36 and N=38 supplied
adjacent full/tail context, not a representative workload population.

The [frozen protocol](../../../compiler/benchmarks/source_pattern_peel/README.md)
fixed 2048 calls per aggregate, two warmups, nine measured samples, a 5 ms floor,
all 15 role/cell combinations then their exact reverse, and CPU2. It retained
P1's zero-involuntary-switch predicate, CPU/wall >=90%, CPU<=wall+100 us,
within-direction max/median <=1.25 and direction-median ratio <=1.10.
Every role needed all nine samples in both directions before pairing. The
primary was selected from P1's known loss and the subsequent code-generation
hypothesis; it is not an unbiased survey or a population performance estimate.

The measured interval remained **source-region-call-return + Result move,
retained ownership, release excluded**. Source guards, allocation/read/copy,
publication, owning observation and completion were included. Verification and
retained-result retirement were outside the timer. This is neither compute-only
nor a complete allocation-to-release lifecycle. The 2048-result retention differs
from P1's workload, so no cross-campaign timing ratio is valid.

| Identity at collection | Exact checkpoint |
| --- | --- |
| Canonical main then | `1f7c491f230251d04d42b5d4a0af5eaf29bcb685`, PR103 |
| Executed producer / installed SDK | `1ad7fe56b5fcb10efe37e523206228e9ef303f6a` |
| Qualified source PR105 head, then unmerged | `cf154deacea2bddb3240660c1af27ff4ffbc4042` |
| Qualified harness PR107 head, then unmerged | `ea7accfe94fbb05a79c5135f074e25390b0e147d` |
| Frozen harness/source bytes | `b63ed32028955594708985519c3d59cedbad7652` |

Both PR heads had all 26 fresh hosted checks successful before collection.
Source acceptance included local Release 356/356, committed x64 source ASan 4/4,
native ARM source execution, installation and adversarial reviews. The final
PR-side Release/Debug/ASan logs retained 19 explicit AVX512 skips in their baseline
scopes; none was relabeled as an executed pass. Final harness Debug actually
executed `benchmark.source_pattern_peel.runner` within 332 passed +19 skipped
tests out of 351. Local synthetic tests are separately recorded as 28 methods.
See the [source qualification](../../mdslc/agent-reports/source-output-peel-qualification-v1.md).

The original protocol required completed qualification and separate operator
coordination, not a canonical merge. The [operator gate](https://github.com/onlyxItachi/MatcoreDSL/issues/106#issuecomment-6760504328)
explicitly retained the qualified-but-unmerged labels while documentation PR95
was pending. A later merge does not relabel these executed artifact identities.

## Actual outcome

Of 270 measured aggregates, **192 passed the sample predicate and 78 failed**.
Every failure was solely a nonzero involuntary-switch delta: 75 samples had one,
three had two. There were no sub-floor, CPU-ratio or positive clock-margin
failures. Four of 30 direction/cell processes satisfied all their sample rules;
none of the 15 role/cell pairs satisfied both directions. Thus all 18 planned
comparison ratios, including the primary, remain **null**.

Each entry below is accepted forward / reverse samples, each out of nine.

| Role | N=36 | N=37 (primary shape) | N=38 |
| --- | ---: | ---: | ---: |
| Stock | 6 / 6 | 9 / 6 | 8 / 7 |
| Old 3x5 | 5 / 7 | 6 / 9 | 5 / 9 |
| Peeled 3x5 | 7 / 6 | 8 / 7 | 8 / 7 |
| Old 5x3 | 3 / 3 | 4 / 5 | 4 / 3 |
| Peeled 5x3 | 8 / 7 | 9 / 6 | 7 / 7 |

The primary's old 5x3 had only 4/9 and 5/9 accepted samples; peeled 5x3 had
9/9 and 6/9. Its clean forward direction cannot replace its rejected reverse
direction, nor can another role replace the primary. Raw medians are retained
as evidence, not promoted into qualified call times, ratios or a winner.

Timed wall intervals were 12.958396..53.571731 ms, above the fixed floor.
The minimum observed thread-CPU/wall ratio was about 0.998625. These are noise-
predicate diagnostics, not evidence that interruptions were harmless. The frozen
RUSAGE/thread-clock reads bracket a wider interval than the wall timer: the
record does not prove that each switch occurred inside the loop, quantify its
interference or authorize a one-switch exception. Warmups separately had
43 accepted and 17 switch-rejected aggregates; they are not timing samples.

## What the run does establish

- Exactly 40 processes: five initial controls, 30 measurements, five final
  controls; 120 raw files, complete order/argv/exit/stderr and identity joins.
- Ten controls each reported 17 categories, 469 checks and zero failures:
  4,690 checks total. Timed execution reported 552,960 checked calls and
  675,164,160 checked output elements; warmups reported 122,880 calls and
  150,036,480 elements. An independent integer/f32/FNV oracle reconstructed
  the expected ordered output hashes and counts from raw receipts. This is not
  a retained dump of every output element or hostile-origin attestation.
- All 225 pinned inputs and actual process module inventories matched. Ten
  source bindings and 80 actual primitive-field checks include 48 timed/control/
  end-K cross-instance joins. Friendly timed integer inputs alone would not
  distinguish every strict arithmetic violation; artifact-bound numerical
  qualification and end-K controls remain essential.
- Actual source-generated full/tail code was inspected before timing. For 5x3,
  the peeled full domain keeps independent output accumulators across K with
  separate multiply/add. Leaf/wrapper sizes and inlining also change, so this
  is not an isolated causal experiment on peeling alone. See the
  [untimed preparation record](../../mdslc/agent-reports/source-peel-p5-preparation-v1.md).
- All 62 host snapshots passed the declared predicates on an AMD Ryzen AI9
  HX370, Linux7.0.0-43, logical CPU2 / SMT sibling14. Observed load1 was
  0.3501..0.7319 over 24 logical CPUs; temperature50.125..74.875 C; existing
  performance governor and boost=1 were unchanged. Before/after observations
  recorded one task and CPU2 affinity, not continuous exclusivity, fixed clocks
  or absence of sibling/OS interference.
- Peak benchmark-owned payload bound10,776,716 bytes; observed peak RSS maximum
  24,711,168 bytes (different quantities); raw collection219,895 bytes; no new
  generated artifacts during collection. All process and campaign budgets held.

No CPU/provider/default/GPU/SPIR-V selection policy or numerical permission changed.
P1's older globally inconclusive campaign and its one qualified 2.45145x old5x3
loss remain untouched. Neither P1 nor P5 is rerun, repaired or combined into a
performance estimate. No BLAS parity, autotuning or crossover policy follows.

## Independent review and durable identities

The [independent raw review](../../mdslc/agent-reports/source-peel-p5-independent-raw-review-v1.md)
used standard-library-only arithmetic and hashing, not the benchmark's acceptance
or summarization code. It agreed with every raw count, binding, rejection and
null comparison. It disclosed test-fixture authorship, not controller or timed-
source authorship. The separate controller-author audit replayed the entire
collection with subprocess/live-host sampling blocked and rejected all 37
in-memory evidence falsifiers. It is author corroboration, not an independent
oracle. Every original preparation and collection file remained unchanged.

| Evidence | SHA256 |
| --- | --- |
| Frozen plan | `2933f108ba13bb7e7bd417e8f53efe416650d22506b0a7e2e514278e94b42d85` |
| Prepared receipt | `add6dfba5ea56d46a743479194bb289f631d780bc85dfa59660a6862e5f8afa6` |
| Actual collection | `7c76073ec7136586bf92a6576cbd7dd995b0159e5ae0a7506a43765bb42965e7` |
| Operator qualification record | `995878b9917ebba8b8ed6f2d9c13c8698d2e7f3377b1aba842b988e0e5a46364` |
| Source final CI rollup | `7f3b25e0a956c4ff906235844a24055d34c19e55caa0e4395f5c9e0c8b5f1b4f` |
| Harness final CI rollup | `1d3424af8a34378169185abb382e5a69082a8158eb2dfc8ad9db6918c11908cb` |
| Independent audit JSON | `79a54b4e91e540142167fd11fe3ff7cb93572fa308e9ade606166d16c398366b` |
| Controller-author audit JSON | `5095945a3fbcddbe81d0bb2e97a29cf3d965d636dbd4f3f170b7dfda4a3d5c24` |

Original capsule: `/var/tmp/mdslc-p5-campaign-FhMFbP/{prepared-v1,collection-v1}`.
Independent audit: `/var/tmp/mdslc-p5-independent-raw-audit-Q2F71g`.
Author audit: `/var/tmp/mdslc-p5-collection-author-audit-BzUnyS`.
The ignored local archive
`benchmark_reports/source-pattern-p5-2026-11-10-v1/receipts.tar.gz` has SHA256
`fea9174fc1b5f7cfe227fa48812960b000ce793675e366c6afb9e86345ef51d3`
and occupies 1,174,843 bytes. It contains 644 original files (5,167,935 bytes)
plus a manifest with SHA256
`4d42b2bfa1783daa5f21b7d7c0fafc8231bfcd506339c91eb47f7a26fa7a6352`:
preparation/collection capsules including generated executables/private bundles,
receipts and raw output, the operator gate, the independent and author audit
scripts/results, preparation falsifiers, and final harness CI
rollup/Debug log/scope. Two agents verified every archived file and its unchanged
original without extraction or rerunning any experiment. This is local evidence,
not a remotely published archive or a backup of the complete toolchain.
The full pinned SDK/toolchain remain separate prerequisites; receipts are not
an executable importer, self-contained SDK or artifact-origin attestation.

## Exactly one next justified boundary

**Separately preregister and qualify a source-call measurement method**, using
identical-binary null controls and controlled comparisons with ex-ante tests for
repeatability, order sensitivity and bias. This host/workload/strict-predicate
combination produced no eligible pairs; it does not prove a broken clock or a
bound on switch interference. Any new rule needs new, independent validation and
may itself fail or remain inconclusive. Null agreement alone cannot validate
comparisons between different allocator/cache/generated-code paths.

Do not loosen P5's predicate from these 78 rejections, reclassify P1/P5, publish
rejected median ratios, select an ISA/provider or change default policy. A
separate method-validation boundary must precede further P5-successor comparative
source-call optimization claims; it is not a new automatic tuner or permission
to repeat this campaign.
