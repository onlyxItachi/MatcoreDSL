# P1 source-time strict CPU tile evidence

Issue #98 harness, not a timing result. This path never changes the production
compiler/runtime, source API, numerical permission, default candidate or policy.
It requires an already qualified, relocated Linux x64 21.1.8 installation.
Compiling the `.mdsl` host and collecting timings are separately coordinated
steps; the initial implementation has only synthetic Python validation.

## Frozen experiment

One byte-identical `call_return.mdsl` source is processed by native
`mdslc-region --region p1_pipeline --candidate generated-strict --optimization none`:
stock (no tile flag), explicit `--gemm-output-tiles 3,5`, and explicit `5,3`.
The region reads A/B snapshots, performs strict GEMM, publishes C, takes an owning
observation and completes. No imported object, callback, direct leaf invocation,
provider comparator or runtime tuning is used.

| Cell | M | K | N | Calls per aggregate |
| --- | --- | --- | --- | --- |
| 0 | 32 | 32 | 32 | 128 |
| 1 | 33 | 35 | 37 | 128 |
| 2 | 64 | 7 | 19 | 256 |
| 3 | 1 | 257 | 65 | 256 |
| 4 | 65 | 257 | 1 | 256 |
| 5 | 128 | 128 | 128 | 8 |

Each cell has 2 warmup aggregates and 9 measured aggregates, a 1 ms floor,
one explicitly selected CPU and the fixed four-A/fixed-B sequence/seed in
`contract.py`. Every aggregate restarts the same sequence phase. Input values
are small exactly represented integers; products have magnitude <=99. Both
Python and C++ independent integer oracles bound every increasing-K prefix
inside `(-2^24,2^24)` and check every output bit. Untimed non-FMA/order,
subnormal/underflow, signed-zero, overflow/Inf/NaN and nondefault caller-FP
controls cover distinctions that integer timed data alone cannot falsify.
NaNs compare by class only in those untimed controls.

Process order is all six cells in order, each stock/3x5/5x3, then that entire
18-process sequence reversed. No role, repetition count, sample, floor, shape
or order is adapted after seeing timings. Losses are valid results. Sub-floor,
noisy or direction-sensitive observations remain explicit inconclusive evidence,
not retries or automatic schedule selection.

## Primary interval and ownership

The exact label is **source-region-call-return + Result move, retained ownership,
release excluded**. Each timer encloses all fixed input-choice/descriptor-wrapper
work, full synchronous source calls, read/value/observation allocation/copy,
checks/completion and noexcept Result moves into a pre-reserved owner vector.
Destruction of empty moved-from temporaries may occur inside; no retained
observation/output ownership is released inside the interval.

Every Result remains owned until after the stop clock. All observations and
final C are fully verified afterward, with input immutability and output guards;
the output hash is an additional check, not a substitute for per-element checks.
Result/Observation retirement and owner-vector allocation/release are outside.
Warmup and measured aggregates use the same lifetime/rotation/verification policy.
This is neither compute-only nor whole-call-plus-release or allocation-free timing.

The 64 MiB benchmark-owned live-payload planning bound includes four A inputs,
fixed B, external C/expected data, retained observation payload, transient input/
result copies and conservative per-Result metadata slack. It is checked before
allocating test data. Per-process peak RSS is separately observed and includes
libraries/allocator overhead; the payload bound is not a total-RSS/address-space
or exact allocator-accounting claim.

## Preparation and identity gates

`prepare` compiles six actual source executables: three timed hosts and the same
three stock/tile requests for the existing independent 68-case `connected.mdsl`
corpus. Each timed executable also runs its own 17 named untimed controls.
The existing `compiler/tests/source_pattern/run.py` inspector checks requested
manifest/files, actual Transform operand/SCF steps, local leaf ownership and
NEEDED bindings. All eight stage/LLVM/object/issuer hashes are joined between
the control and timed leaf for each requested tile. Distinct requests must have
distinct consumed derived artifacts while original primitive witnesses agree.

Stock is not assumed scalar. Supply a root-frozen producer path catalog:

```json
{
  "schema": "matcore.source-pattern.p1.stock-profile",
  "version": 1,
  "source_commit": "40-lowercase-hex-characters",
  "producer_candidate": "/frozen-build/lib/libmatcore_closed_candidates_isolated_v1.so",
  "cache": "/frozen-build/CMakeCache.txt",
  "leaf": "/frozen-build/lib/regions/strict-normal.ll",
  "object": "/frozen-build/lib/regions/strict-normal.o",
  "build_ninja": "/frozen-build/build.ninja"
}
```

The schedule/flags come from the actual issuer manifest/cache/link graph, with
all retained stage hashes checked. Standard CMake install can remove producer
RUNPATH. The runner copies the producer DSO to a new **nonexecuted** inspection
file, runs pinned CMake `file(RPATH_REMOVE)`, then requires byte-for-byte identity
with the installed DSO (which v1 requires to have no RPATH/RUNPATH). It does not
ignore `.dynstr`, invent an ELF normalizer, mutate the producer/SDK, or execute/
import the derived copy. This is producer inspection, not a source/runtime
authority token or independent build-origin attestation.

Driver/compiler/linker/CMake and compilation dependency closures are frozen.
Each timed process reports actual executable file mappings; these must match its
loader inventory, whose executable/runtime/candidate/provider/transitive bytes
are hashed before/after that process. The whole installation is checked at
collection endpoints. An emitted bundle and installation must remain in place;
post-emission relocation or edited/resumed receipts are not supported.

## Commands, limits and refusals

Use explicit NEWDIR paths outside **every** Git worktree, e.g. under `/var/tmp`.
This avoids the user's enclosing home/Observatory worktree; no `GIT_CEILING`,
unrelated repository or global configuration is changed.

```sh
python3 -B compiler/benchmarks/source_pattern/tests.py
python3 -B compiler/benchmarks/source_pattern/run.py prepare \
  --driver /relocated-sdk/bin/mdslc-region --prefix /relocated-sdk \
  --stock-profile /frozen-producer-profile.json \
  --cxx /usr/bin/clang++-21 --linker /actual/configured/linker \
  --output /var/tmp/mdslc-p1-NEW-prepared
python3 -B compiler/benchmarks/source_pattern/run.py collect \
  --prepared /var/tmp/mdslc-p1-NEW-prepared --cpu 2 --exclusive-host \
  --output /var/tmp/mdslc-p1-NEW-collected
python3 -B compiler/benchmarks/source_pattern/run.py summarize \
  --collection /var/tmp/mdslc-p1-NEW-collected/collection.json
```

CPU 2 above is an example of the owner's proposed host choice, never an automatic
selection. No persistent affinity, governor or boost setting is changed.
Collection allows <=20 seconds per process and one <=180-second total deadline,
including initial/final controls and endpoint identities. Raw command logs have
a 64 MiB cap; six generated binaries/four tile bundles/source/inspection artifacts
have a separate 256 MiB cap. Supplied SDK/toolchain/build bytes are inventoried
external prerequisites, not hidden inside those caps. Excesses refuse and
preserve files; no automatic deletion, resume or retry is implemented.

All `LD_*`, allocator/GLIBC/sanitizer tunables and provider/thread tuning
environment families are removed before children, then recorded OpenBLAS/OMP/MKL
thread limits are set to 1 even though no provider is selected. Exact stripped
names/applied values are recorded without dumping unrelated environment secrets.
The uninstrumented single-thread leaf contract plus process/aggregate task-count,
singleton-affinity and thread-CPU observations are **bounded evidence**, not proof
that no transient worker, SMT sibling or shared-cache competitor ever existed.

Known build/test process **comm counts only**, load, CPU flags/model/kernel,
SMT/cache topology, governor/boost and available CPU temperatures are observed.
Known build/test activity, load1 > one quarter of logical CPU count or observed
CPU temperature >=85C refuses collection. Each aggregate additionally requires
zero involuntary switches, thread CPU >=90% wall time, clock-boundary slack
<=100 us and the floor. Missing temperature data is explicit, not a thermal claim.

Summary uses exact aggregate/repetition normalization, nine-sample medians,
nearest-rank p95/range, and ratio of paired forward/reverse midpoint medians.
p95/median >1.25 or directional median max/min >1.10 marks that role inconclusive.
Both directional ratios and losses remain visible. No confidence/significance,
automatic winner, threshold, default/provider change, broad speedup, BLAS parity,
Issue #15 closure, or GPU/SPIR-V performance claim follows. Offline summaries
rejoin each receipt/exit/argv to the complete raw digest catalog and recompute
normalization, acceptance and pairing; editable derived values are not trusted.
