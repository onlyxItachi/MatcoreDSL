# Native provider evidence campaign v1

Date: 2026-09-05

## Scope and checkpoints

This lane started from canonical `main` and `origin/main` at
`327530d287e41c4115365598e76b17e149a1c45a` in the isolated worktree
`/home/hamza-usta/MatcoreDSL-wt-native-provider-evidence-campaign-v1` on branch
`mdslc/native-provider-evidence-campaign-v1`. The Milestone 7 merge
`e5069758ad04bdb459de2026cad8498b47fda707` is an ancestor of that checkpoint.

The primary evidence-audit implementation checkpoint is
`9f5b5dba53b80579be7228010bf832b5ce29b8b4`; claim-boundary wording first landed
at `9c6d4647d8bca3590429a00d72d3498833118855`; atomic output safety was hardened
at `58d947fc0954aa6604d792ac2f934065f42ffc12`; and independent-review closure for
ambiguous provider identity and bounded output failures is captured at
`7ff052ce8ddf458c87e555266a4798e0f6b3be0e`. No production planner, runtime,
kernel, public API, public ABI, threshold, or provider-selection policy changed.
The candidate is published for review as draft PR #23; it was not merged by
this lane.

Live GitHub Issue #15, **MDSLC Milestone 7 — Native BLAS Parity**, remained open
at inspection time. Its 2026-07-26 owner disposition still says that the
complete authenticated same-checkpoint forward/reverse envelope, final-code
scaling aggregates, and full-envelope planner regret are unestablished, and
that cooperative packed-B preparation remains production-dormant. This lane
does not redefine those conditions.

## Observed evidence

### Local artifact inventory

A targeted inventory of every `manifest.json` below
`/home/hamza-usta/.tmp` and `/home/hamza-usta/archives` found sixteen manifests.
Every one declares `matcore.cpu-performance-deep-audit.manifest`; none declares
`matcore.native-blas-parity.manifest`. The two paths whose directory names begin
with `m7-deep-reject` are one-case deep-audit fixtures, not Native BLAS parity
bundles.

The repository records an older partial forward manifest with SHA-256
`26e75ecbcfbb19d024fa8a5fa9790b65a2deb5743b39f16a4f22dd39381cfe69`
and 258 of 368 cases. A targeted search of the two artifact roots found neither
that digest nor a parity manifest. The historical receipt therefore cannot be
re-authenticated locally and cannot be combined with the current checkpoint.

For the current 12-physical-core host, the frozen manifest-v3 authority builds
368 cases per stable order:

| Mode | Cases per order |
| --- | ---: |
| complete hot | 171 |
| automatic complete hot | 72 |
| bounded planner regret | 33 |
| repeated hot | 48 |
| prepacked-B hot | 32 |
| diagnostic complete hot | 12 |
| **Total** | **368** |

The diagnostic audit at `9f5b5db` found zero current forward records, zero
current reverse records, and zero paired cells. It exited `1` with
`overall_status=incomplete`. The audit JSON is intentionally outside Git at
`/home/hamza-usta/.tmp/mdslc-provider-evidence-9f5b5db.XzsV6R/`
and has SHA-256
`1a0ed7c73b8848826e50054af6cc39a673f89e0ca6898ec6459187f698c69271`.

### Host and provider inspection

The inspected host was Ubuntu 26.04.1 x86-64 on an AMD Ryzen AI 9 HX 370,
with 12 physical cores, 24 logical processors, one NUMA node, Clang 21.1.8,
CMake 4.3.2, Ninja 1.13.2, and pkg-config OpenBLAS 0.3.32 pthread. A clean
Release build with `MDSLC_ENABLE_OPENBLAS=ON` and
`MDSLC_REQUIRE_OPENBLAS=ON` succeeded. The exact `matcore-bench` binary at the
implementation checkpoint has SHA-256
`d06480a0c28f74665670be7dfdded90895a116dad7abfda4f7273e0e04ffa165`.

A one-thread, one-element provider probe was used only for provenance and
legality inspection, not as performance evidence. The parity runner accepted
its raw report. It recorded:

```text
source_commit = 9f5b5dba53b80579be7228010bf832b5ce29b8b4
provider_name = OpenBLAS
provider_version = 0.3.32
provider_config = OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=128
requested_threads = actual_threads = 1
```

The raw probe SHA-256 is
`e2914c7d3ee02f519e3618e641ef4f7e1e5de39716428dc5773d4022b62500e7`.
The provider's reported `Cooperlake` core on this AMD host is an observed
runtime string, not evidence here that the provider chose either a correct or
an optimal kernel.

Manual linkage inspection resolved `libopenblas.so.0` to
`/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.32.so`, whose
bytes had SHA-256
`be2e7d119279836105e0361be92c78dbcd8a6e7357e74339bdbb32a5195ee35e`.
That inspection is not bound into manifest v3 and must not be presented as
authenticated parity-bundle provider-binary identity.

### Exact multi-thread falsification

The current runtime deliberately publishes an authenticated OpenBLAS thread
ceiling of one because its conformance probe observes only the calling
thread's floating-point environment. This is a correctness boundary, not a
performance policy. The 12-core parity matrix nevertheless contains 33 forced
OpenBLAS `complete-hot` cells above one thread in each order: threads 2, 3, 4,
8, or 12 depending on exact native task capacity.

The first such forward cell was executed as a focused falsification:

```text
case = calibration / medium-square / 192x192x192 / OpenBLAS / requested 2
benchmark process = success
reported actual_threads = 1
parity authentication = rejected
reason = forced parity candidate did not use the exact requested thread count
```

The raw negative SHA-256 is
`33373052859199165b7224588955cbfb28f7bd2e4acd6b347d90361598dcef62`.
This establishes that a current full forward run reaching this cell stops at
that mismatch; it does not establish multi-thread provider performance.
Weakening exact requested/actual equality would manufacture comparability and
was rejected.

The desktop was also non-exclusive during inspection (browser, desktop, and
multiple compiler agents were active). No full timing campaign was attempted.
The exact-thread incompatibility above is independently sufficient to make a
current full paired campaign premature.

## Accepted changes

### Diagnostic completeness auditor

`audit_native_blas_parity_evidence.py` now accepts an explicit expected source
commit and physical-core count plus zero, one, or both order manifests. It
reports, without timing aggregation:

- exact frozen case count, missing/extra/duplicate keys, order, record fields,
  and state counts;
- manifest contract, plan, runner, source checkpoint, and benchmark digest
  differences;
- missing, unexpected, or digest-mismatched raw files;
- grouped per-record authentication failures and observed provider metadata;
- strict bundle-authenticator results for each order;
- strict pair-authenticator readiness.

The auditor returns `0` only after the existing summarizer's strict
`load_bundle` and `pair_bundles` authorities accept both orders at the
explicitly requested campaign checkpoint. That status is named
`ready-for-bounded-summary`; it is not a performance or parity verdict. Missing
or incomplete evidence returns `1`. Tool or output-safety errors return `2`.
Output collision checks prevent overwriting manifests, referenced raw files,
the benchmark binary, runner, summarizer, or auditor. Atomic JSON output uses
an exclusively created randomized sibling, so a crafted output name cannot
alias and overwrite one of those protected inputs through its staging file.

### Provider metadata rejection

Both the live runner and the independent summarizer now reject a raw result
that selected OpenBLAS unless it carries `provider_name=OpenBLAS` and non-empty,
trimmed, case-normalized non-placeholder provider version and configuration
strings. This closes a narrow gap between benchmark schema intent and Python
evidence authentication; it does not claim cryptographic identity for the
loaded provider library.

Adversarial tests prove rejection of selective case omission, unfinished state
with stale raw bytes, raw digest tampering, missing provider identity, a valid
pair from the wrong expected source checkpoint, one-sided evidence, direct
output collision, staging-file collision, uppercase/whitespace placeholder
identity, and a directory selected as JSON output. Output-write failures return
the documented tool/safety status without a traceback. The positive fixture is
a synthetic contract fixture only and is not performance evidence.

## Current gap matrix

| Requirement | Current authenticated state | Disposition |
| --- | --- | --- |
| exact current source/binary | clean `9f5b5db` diagnostic build | available for probes only |
| forward manifest v3 | 0 / 368 cases | missing |
| reverse manifest v3 | 0 / 368 cases | missing |
| authenticated paired cells | 0 / 368 | missing |
| exact multi-thread provider comparator | 33 planned cells/order clamp to one under current proof boundary | mechanically blocked |
| same-checkpoint scaling aggregates | none | missing |
| full-envelope planner regret | contract times 11 / 24 shapes; no current raw | missing by original criterion |
| provider package/config metadata | authenticated for the one-thread probe | available, non-performance |
| loaded provider shared-object hash in manifest | not represented by manifest v3 / benchmark v6 | unresolved; no schema change here |
| cooperative packed-B production selection | dormant | unchanged |

Issue #15 therefore remains partial and open. Neither package availability,
the one-thread smoke, a synthetic passing fixture, nor the new completeness
auditor supplies any missing performance evidence.

## Validation

The following checks passed at the implementation checkpoint; the runner,
summarizer, and auditor contracts, Python byte compilation, Ruff, and
`git diff --check` were repeated after the `7ff052c` independent-review
hardening:

- Python byte compilation for runner, summarizer, auditor, and their three
  contract tests;
- Ruff for those six Python files;
- direct auditor contract and summarizer contract tests;
- clean Clang 21.1.8 Release configure and `matcore-bench` build with OpenBLAS
  required;
- CTest parity runner, summarizer, and evidence-auditor contracts: 3/3 passed
  in 43.28 seconds;
- CTest benchmark core, CLI JSON, provenance-incremental, deep-audit runner,
  and deep-audit summarizer contracts: 5/5 passed in 6.13 seconds;
- the focused one-thread provider authentication and exact-thread negative;
- `git diff --check`.

No full native/OpenBLAS acceptance sweep, sanitizer matrix, Windows job,
package consumer test, or performance threshold evaluation was run or claimed
by this focused evidence lane.

## Adversarial conclusion and next boundary

The implementation is low-risk compiler-test infrastructure and does not
enter runtime execution. Its useful result is diagnostic and negative: MDSLC
can now state every missing evidence cell mechanically, and the current
multi-thread provider comparison is known to be semantically unauthenticated
before spending time on a sweep.

Before another full Milestone 7 collection, a separate, explicitly reviewed
task must establish whether exact multi-thread OpenBLAS can satisfy Matcore's
floating-point-environment and thread-observability contract. That task must
prove provider-worker numerical state and exact thread use, or retain the
current fail-closed ceiling. It must not solve the mismatch by accepting a
clamped call, inventing thresholds, or relabeling one-thread provider results.

A versioned provider-binary identity extension may also be investigated, but
it needs a portable definition and adversarial proof before changing benchmark
schema v6 or manifest v3. It was deliberately not improvised in this lane.
