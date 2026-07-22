# Milestone 5 performance-evidence synthesis lane

Date: 2026-07-22

## Ownership

This lane owned only:

- `docs/performance/cpu/milestone-5-advanced-cpu-2026-07-22.md`; and
- this agent report.

No compiler, runtime, planner, benchmark source, test, raw JSON, build product,
or legacy file was changed.

## Evidence reviewed

The report was recomputed from the four schema-v4 evidence sets under the
external archive `/home/hamza-usta/archives/MatcoreDSL-M5-perf-20260722`:

- 20 compact/native planner-regret files;
- 6 provider-permitted planner-regret files;
- 2 tiny call-overhead files; and
- 18 forced 4096-cube scaling files.

All 46 files embed clean Git-worktree provenance for
`5f634aef2a0b47cd033df77c40d709456603b405`. A `jq` integrity pass verified
schema/version, provenance, top-level correctness, timing validity, timed-final
output authentication, regret validity, and forced-plan authentication for
every legal timed regret candidate.

Tables were derived directly from `results[0]`, `planner_regret.candidates`,
configuration, environment, affinity, timing, workspace, and correctness
fields. Family speedup and efficiency were recomputed from each retained
4096-cube one-thread median:

```text
speedup = T1 / Tthreads
efficiency = T1 / (actual_threads * Tthreads)
```

Compact regret was sorted directly from all 20 files. With the documented
benchmark-style upper-middle median and nearest-rank p95, it is 1.005207 /
1.047343 / 1.363081 (median / p95 / maximum). Provider-permitted regret across
six files is 1.000000 / 1.343740 / 1.343740. No prior narrative values were
used as table inputs.

## Validation

```text
JSON files reviewed:                 46
Schema-v4 files:                     46/46
Clean exact source provenance:       46/46
Correct and timing-valid results:    46/46
Authenticated final timed outputs:  46/46
Compact regret above 1.50 / 2.00:   0 / 0
Provider regret above 1.50 / 2.00:  0 / 0
```

The combined SHA-256 over the sorted list of the 46 content digests is:

```text
2b654699d05dbd2ba0516760d2be977b4f56f356e4a6ae94737182dfca37ac2b
```

Focused validation used `jq -s` over each evidence set, regenerated the
compact, provider, and three scaling tables into `/tmp`, and compared every
rendered row with `diff -u`. Aggregate bounds were asserted directly against
the sorted JSON regret arrays. Final documentation checks were:

```text
mechanical table and aggregate checks: PASS
provider and scaling table checks: PASS
repository hygiene check passed
git diff --cached --check: PASS
```

The report explicitly separates compact/native and provider-permitted planning,
native affinity from OpenBLAS scheduling, one-thread packed from multithread
parallel variants, the degraded AVX-512 baseline, caller-isolation limits, and
the measured source snapshot from later documentation integration. It makes no
universal or unavailable-hardware claim.
