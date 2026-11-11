# P5: matched source-requested N peeling

This is a separate preregistered experiment, not a result, compiler optimization,
cost model or dispatch policy. Preparation and collection are separately
authorized steps. **No timing may start before PR105's source-connection native,
installed and relevant hosted qualification finishes at an immutable checkpoint.**
The five roles must then come from one clean, relocated, uninstrumented baseline
Linux x64 Clang/LLVM/MLIR 21.1.8 installation. No old P1 executable is a comparator.
The preregistered producer checkpoint is
`1ad7fe56b5fcb10efe37e523206228e9ef303f6a`; naming it does not assert that its
pending qualification or merge has happened.

P1's original source, runner, contract and raw receipts are immutable. Its overall
`INCONCLUSIVE_TIMING` and accepted 5x3/stock loss of 2.45145 at 33/35/37 remain
historical evidence at that older checkpoint; this experiment neither repairs nor
reinterprets them. See the [retained P1 report](../../../docs/performance/cpu/source-pattern-p1-2026-11-10.md).

## Fixed question and roles

The primary question is the whole source-call interval of explicitly requested
peeled 5x3 versus explicitly requested old 5x3 at M/K/N = **33/35/37**. This pivot
was selected from an already public loss and a subsequent code-generation
hypothesis, not sampled from a representative workload population. Results cannot
establish general GEMM performance, an optimal tile or the cause of an observed
timing difference.

All roles use `--candidate generated-strict --optimization none` and the same
ordinary C++ mathematical source:

| Role | Source request |
| --- | --- |
| `stock` | No tile flag; verify the actual producer's `scalar-mnk` profile |
| `tiles-3x5` | `--gemm-output-tiles 3,5` |
| `peel-3x5` | `--gemm-output-tiles-peel-n 3,5` |
| `tiles-5x3` | `--gemm-output-tiles 5,3` |
| `peel-5x3` | `--gemm-output-tiles-peel-n 5,3` |

The 3x5 contrasts are secondary same-tile comparisons, not a null-effect control.
Stock is context at the new checkpoint. Its scalar profile must be verified from
the actual producer and installed bytes, not inferred from its role label or
assumed identical to P1.
Unit N is an authority/identity qualification control, not a measurement role.

## Fixed workload and chronology

| Cell | M | K | N | Calls per aggregate |
| --- | --- | --- | --- | --- |
| 0 | 33 | 35 | 36 | 2048 |
| 1, primary | 33 | 35 | 37 | 2048 |
| 2 | 33 | 35 | 38 | 2048 |

For N tile 3 these cover full-only, full plus width-one tail and full plus
width-two tail. They are not a broad shape sweep. Every role/cell has exactly
two warmup aggregates and nine measured aggregates, the original fixed four-A /
fixed-B sequence and seed, and a **5 ms floor**. No role-specific repetitions,
calibration, warmup extension, deletion, resumption or retry-until-quiet exists.

The forward sequence is cell 0,1,2, each in the five-role table order. The reverse
is the exact reverse of all 15 entries. Five initial controls, 30 measurement
processes and five final controls give exactly 40 collection processes. Raw
command indices are initial 0..4, measurements 5..34 and final 35..39. Every
warmup and measured call's output is checked; warmups are not timing samples.

2048 is fixed before observing new code. It is 16 times P1's 128-call pivot
aggregate. P1's lowest stock pivot aggregate was 1.289977 ms, whose arithmetic
16-fold scaling is 20.639632 ms. This motivates headroom, **not a prediction**:
the compiler checkpoint, implementation, retention and cache behavior differ.
If new code falls below 5 ms, it stays inconclusive without increasing the count.

## Exact interval and retention

The label remains **source-region-call-return + Result move, retained ownership,
release excluded**. It includes fixed input-choice/descriptor-wrapper work,
ordinary source guards, allocation/read/copy, publication, owning observation,
completion and moves into a pre-reserved owner vector. Moved-from empty temporary
destruction may occur inside; retained output/observation retirement may not.
Verification and release occur after the stop clock.

2048 retained Results are a new workload compared with P1, not a timer-only
change. At N=38 the planning bound is 10,776,716 bytes for Result size 80, or
12,710,028 bytes at the accepted conservative size bound 1024. Check the actual
size and bound before allocation. This is benchmark-owned payload accounting,
not total RSS, allocator accounting, cache residency or a physical cache-state
guarantee; peak RSS is separately observed. Do not move release into the interval
to fit the cap or call this compute-only, allocation-free or call-plus-release.

## Primary acceptance and conditional claims

The **original P1 primary noise predicate is retained byte-for-byte**: aggregate
wall time must clear the new floor, thread CPU must be at least 90% of wall,
thread CPU must not exceed wall by more than 100 us, and involuntary switches
must be zero. Record voluntary switches as before. Affinity, observed CPU and
one-process-task boundary checks remain mandatory. These are bounded observations,
not proof of absent SMT/cache/OS competition or a noise-free interval.

There is no one-switch or 50-us rescue rule. P1's 19 switched samples each having
one switch and at most 15,462 ns wall-minus-thread was not an interference bound.
Resource/CPU clock observations bracket a wider region than the wall timer;
wall-minus-thread is not exact descheduled time. Longer aggregates can increase
strict-gate failures, and a globally inconclusive experiment is an acceptable
outcome, not permission to revise the plan.

Every role pair requires all nine measured samples in both directions to pass.
The within-direction nearest-rank p95 / median limit is 1.25; with nine samples
p95 is the maximum. Forward/reverse median max/min must not exceed 1.10. No
direction, sample subset or alternative estimator is selected after collection.

A comparison has values only when both participating roles are paired. Report
both directional ratios and the ratio of midpoint-of-direction-medians, including
losses. The predeclared primary pair may be described conditionally even if
another planned cell makes the whole campaign inconclusive; never promote a
secondary survivor to replace a failed primary. No pooled survivor geomean,
confidence/significance, speedup policy, winner or tuning threshold is emitted.

## Identity and implementation reuse

A new whole main-file translation unit is rendered from the exact frozen P1
source with only three guarded substitutions: matrix, floor and receipt schema.
The region, call wrapper, reserve, aggregate/predicate, clocks, retention,
verification and retirement remain unchanged. The complete deterministic diff
and both source hashes belong to preparation evidence. A wrong base hash or
missing/repeated anchor refuses. Do not include/rename P1's main as a workaround:
the selected region definition must remain in the physical main source file.

Only configuration-free P1 helpers are reused read-only. P1's plan, schema,
prepare/collect/summarize logic and module globals are not repurposed or patched.
P5 has its own strict plan, receipt, chronology and offline-audit contract.

For every requested role, join all eight issuer/stage/LLVM/object hashes between
the actual timed leaf and its independently executed source controls. Reuse the
qualified peeled end-K full/tail controls as well. Original WHAT witnesses must
agree; each old/new HOW must be actually consumed under the correct classifier
and source schema/binding domain. Correct output or different flags alone are
not evidence of the requested implementation.

Pin the actual driver/toolchain/linker, stock producer/profile, missing-leaf kit,
headers, Runtime/provider owner, final local-symbol candidate DSO and executable
dependencies. Inspect the actual final DSO leaf, not a research specimen. Each
measured process's executable mappings must join the pinned loader inventory;
recheck artifacts before/after processes and the installation at both endpoints.
No artifact importer, executable callback or post-publication self-authentication
is created. Code shape alone is not a timing or isolated-mechanism causal proof.

## Limits, failures and synthetic controls

Caps remain 64 MiB benchmark-owned live payload, 64 MiB new raw logs, 256 MiB new
generated artifacts, 20 seconds per process and 180 seconds total collection,
including controls and endpoint audits. Supplied SDK/toolchain/producer bytes are
separately inventoried prerequisites. All paths are new external directories
outside every Git worktree; exceedance retains partial evidence without cleanup,
resume or retry.

Use one explicitly preregistered CPU, the existing environment stripping/thread
limits and operator quiet-host acknowledgement. Known build/test process counts
must be zero, load1 at most one quarter of logical CPUs, observed temperature
below 85 C. Record topology, SMT sibling, governor/boost and unavailable sensors.
Do not change system settings or infer continuous exclusivity.

Source/output/call-count/module/identity failure is FAILED or INCOMPLETE, never
excused as timing noise. Raw stdout/stderr/commands, complete chronology, hashes,
exit codes and every derived acceptance/normalization/ratio must be rechecked
offline. Missing/reordered/duplicated evidence or edited derived values refuses.

`python3 -B compiler/benchmarks/source_pattern_peel/tests.py` is synthetic only:
it must not build a compiler, prepare a capsule or launch a benchmark. Native
preparation and a single collection require separate explicit coordination after
qualification and review; this README reports no such run.

The synthetic collection fixture writes 40 command records and 120 real raw
files. It mocks **only** the preparation prerequisite; module files are inert,
not executable or loadable. Full collection chronology, raw-file/catalog joins,
clock/status/normalization/ratio derivation, retained host predicates and caps
are checked with subprocess launch forbidden. This is not a successful complete
preparation, native SDK, physical execution or collector-origin attestation.

After separate authorization, the runner's modes are `prepare`, `collect` and
`summarize`. Preparation requires `--driver`, `--prefix`, `--output`,
`--stock-profile` and the actual `--linker`; optional tool paths are `--cxx`,
`--objdump` and `--size`. Use the existing stock catalog format with **new**
checkpoint/producer paths, never the archived P1 producer. Collection requires
`--prepared`, a new `--output`, explicit `--cpu` and `--exclusive-host`;
summarization requires `--collection`. An offline audit revalidates retained
evidence without spawning a compiler, inspection program or timed process.
