# Native BLAS parity methodology v1

Status: corrected before final Milestone 7 acceptance. The original blind
holdout claim is withdrawn; the complete declared matrix remains unchanged.

This contract defines the host-bounded evidence required to evaluate native
MDSLC F32 GEMM against the configured OpenBLAS provider. It does not claim
universal BLAS parity, and an automatic plan that selects OpenBLAS does not
count as native parity.

## Scope

- Platform: the physically measured Linux x86-64 validation host.
- Operation: row-major contiguous `f32` GEMM with `f32` accumulation/output.
- Native candidates: packed AVX2/FMA and packed AVX-512/FMA, including their
  persistent parallel variants when legal.
- External comparator: authenticated CBLAS SGEMM from the configured OpenBLAS
  provider.
- Requested thread ceilings: 1, 4, and the discovered physical-core count.
  Native/OpenBLAS parity cells use the exact native task capacity under each
  multi-thread ceiling. If the output task graph cannot use the full ceiling,
  both native and OpenBLAS are measured at the same lower exact count and the
  ceiling/capacity mapping is retained in the authenticated manifest. A
  capacity-limited cell is not reported as physical-core-count parity.
- Primary timing: hot-cache, caller-owned reused workspace, transient packing
  included, allocation excluded.
- Repeated-weight timing: caller-owned authenticated prepacked B, with its
  one-time preparation reported separately and amortized over 1, 4, 16, and 64
  executions.

Reference, tiled, and compiler-vectorized variants remain correctness and
regression controls. They are not eligible to establish native parity.
Compute-only timing remains a diagnostic and is never compared with a complete
CBLAS call.

## Declared shape partitions and chronology correction

The identifiers `calibration` and `holdout` remain in case keys so evidence
paths and the complete declared matrix stay stable. They must not be
misinterpreted as a blind experimental split.

Independent completion review reconstructed the commit chronology and found
that candidate AVX-512 and AVX2 measurements had already used several shapes
later labeled `holdout` before this methodology document was committed. A
cooperative-packing prototype also used `32x8192x1024` to narrow a threshold.
The latter threshold branch was removed before final evidence collection.
Manifest v3 records:

```text
calibration -> candidate-development-and-validation
holdout     -> declared-validation-not-blind
```

Both partitions are still reported in full. No shape was removed or moved to
improve the result, but Milestone 7 makes no unbiased-holdout claim. This
methodology limitation is part of the final verdict.

### Calibration

| Family | Shapes `(M,N,K)` |
| --- | --- |
| medium square | `96^3`, `192^3`, `384^3` |
| tall-skinny | `4096x64x4096`, `4096x128x1024` |
| short-wide | `64x4096x4096`, `128x4096x1024` |
| tail-heavy | `63x65x67`, `255x257x259` |

### Declared validation (`holdout` case-key identifier)

| Family | Shapes `(M,N,K)` |
| --- | --- |
| medium square | `128^3`, `256^3`, `512^3` |
| large square | `768^3`, `1024^3`, `1536^3`, `2048^3`, `4096^3` |
| tall-skinny | `8192x32x1024`, `2048x256x4096` |
| short-wide | `32x8192x1024`, `256x2048x4096` |
| tail-heavy | `31x33x35`, `127x129x131`, `511x513x515` |

The vector-like family remains a diagnostic because Milestone 6 already found
that it is governed by a different low-arithmetic-intensity regime. It is
reported but excluded from the medium/large parity aggregate.

## Fairness and provenance

Every retained result must:

1. use manifest v3, record the corrected partition interpretation, come from
   a clean exact source commit, and record the benchmark-binary and
   runner digests;
2. use the same seeded inputs and double-precision correctness oracle;
3. authenticate the final timed output;
4. report requested and actual threads, placement, cache mode, packing mode,
   workspace, provider configuration, compiler, and CPU capabilities;
5. run both stable-forward and stable-reverse candidate order;
6. use at least five warmups and eleven retained aggregate samples;
7. aggregate fast operations until the timer floor is met;
8. reject NaN, infinity, incorrect output, illegal fallback, and unauthenticated
   timing;
9. keep raw JSON outside Git and commit only a deterministic sanitized summary.

Single-thread comparisons use the same compact placement when it is
authenticated for both implementations. Multi-thread parity uses an explicitly
unbound stratum (`affinity=none`) for both native and OpenBLAS when provider
worker affinity cannot be authenticated. Bound-native scaling is reported
separately and must not be relabeled as equal-placement BLAS parity.
The deterministic task-capacity projection is contract-tested against the
planner's 128-row, 256-column, cacheline-safe, work-floor model. Every forced
comparison still requires exact requested/actual equality; silent thread
clamping remains invalid evidence.

When task capacity is below a requested 4-thread or physical-core ceiling, the
runner compares both implementations at the smaller exact capacity or omits a
multi-thread cell when capacity is one. This preserves equal-thread fairness,
but it does not establish performance at the original requested ceiling. The
summarizer reports exact-capacity coverage separately and leaves the complete
declared-ceiling coverage gate unmet.

## Metrics

For each shape/thread/mode cell:

```text
native ratio = fastest legal native throughput / OpenBLAS throughput
planner regret = selected complete-call time / fastest comparable legal time
speedup(T) = native one-thread time / native T-thread time
efficiency(T) = speedup(T) / T
```

Forward and reverse passes are paired per cell before family aggregation.
Median, nearest-rank p95, minimum, and maximum are reported. Timer-noise cells
are listed and excluded from aggregate claims rather than silently removed.

The built-in complete-registry planner-regret timing is deliberately bounded
to eleven medium/tail shapes because forcing scalar candidates on the largest
declared problems is not a practical acceptance run. Automatic and parity
cases cover the complete shape matrix but do not time every legal registry
candidate, so they cannot reconstruct exact full-registry regret. The bounded
regret values are diagnostic; the full-envelope regret gate remains not met in
Milestone 7 rather than being inferred.

The deterministic summarizer therefore distinguishes its bounded
paired-measurement assessment from the manual milestone disposition. Bounded
regret, full-envelope regret coverage, catastrophic-regret coverage, and
original 4-/physical-core ceiling coverage are non-acceptance diagnostics in
the summary. A summary `passed` result means only that every
acceptance-enabled measured row passed; it cannot by itself complete Milestone
7. Complete milestone acceptance still requires the full-envelope and original
thread-ceiling evidence below.

## Acceptance thresholds

The declared Milestone 7 targets are:

- single-thread medium/large native/OpenBLAS median ratio at least `0.90`;
- no core family median below `0.75` without an approved explicit limitation;
- at least one meaningful supported family at or above `1.00`;
- multi-thread large-shape median ratio at least `0.85` at equal requested and
  actual thread count;
- four-thread native speedup at least `3.0x` for sufficiently large GEMM;
- planner regret median at most `1.05`, p95 at most `1.15`, and maximum at most
  `1.35` inside the complete declared parity envelope;
- no automatic-selection regret above `2.0` across that complete envelope.

If these thresholds are not met after correct evidence-backed implementation,
or cannot be evaluated from a complete authenticated evidence pair, the
milestone is reported manually as partial. A complete authenticated pair that
misses any acceptance-enabled measured threshold receives a summarizer
`failed` verdict, not `partially-passed`. Malformed or incomplete bundles are
rejected without a performance verdict. Shapes, timing modes, or comparators
must not be removed merely to manufacture a pass.

## Change-control rule

Candidate changes must preserve the stable C ABI, explicit workspace,
fail-closed legality, source provenance, and Windows compatibility. A candidate
is promoted only after forced correctness, sanitizer, exact-instruction, and
paired complete-call measurements pass. Public GEMV/GEVM APIs, runtime
autotuning, and API/ABI freeze remain out of scope.
