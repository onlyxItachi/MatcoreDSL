# P1: authenticated source-time strict CPU tiles, 2026-11-10

## Verdict and scope

**Observed: the fixed campaign completed with `INCONCLUSIVE_TIMING`, not a
qualified whole-matrix performance result.** All requested source executions,
arithmetic/effect controls and artifact-identity gates passed. One independently
acceptable stock-versus-tile comparison is a loss: at M/K/N = **33/35/37**, the
5x3 instance took **2.45145 times** the stock call interval. No speedup, optimal
tile, automatic policy, provider parity or GPU performance claim follows.

This is Issue #98's bounded evidence experiment, not a compiler optimization.
Its [harness and frozen contract](../../../compiler/benchmarks/source_pattern/README.md)
use the existing authenticated source-time pattern route. Neither compiler/runtime
behavior nor source numerical permissions, defaults or provider selection changed.
Issue #15 remains partial/open; this experiment contains no OpenBLAS comparator.

## Frozen inputs and interval

- Harness implementation: `e34d01287ffe238736d2f8440058e9d0e4fa13cf` (initial
  implementation `5591e8b562a53e64103b197a76abb799888d530b`). Source mathematical
  host and timing plan are byte-identical across those two commits.
- SDK producer: clean isolated `06affd2ffea4a5459158a7ddcd0c4e75009ea90e`, from
  canonical starting `b2597bed52de46604574f76eb33663788a5ab994`. This is an explicit
  premerge integration checkpoint, not a claim that it was already canonical main.
- One relocated uninstrumented Clang/LLVM/MLIR 21.1.8 SDK; actual source linker
  GNU `ld.bfd`, independently joined to the issued linker fingerprint. CMake's
  `CMAKE_LINKER` entry alone did not identify the source driver's actual linker.
- Stock `scalar-mnk` versus explicit source requests `3,5` and `5,3`, all
  `generated-strict`, optimization `none`, full increasing scalar K and separate
  f32 multiply/add. No direct leaf invocation or executable-IR import.
- Same source: read A/B snapshots, GEMM, publication, owning observation, complete.
  Primary interval is **source-region call-return + Result move, retained
  ownership, release excluded**. Real checks/allocation/copies/completion and
  fixed input-choice/descriptor work are included. Every Result remains owned
  through the stop clock. Full verification and ownership retirement follow it.
- Six predeclared shapes, fixed repetitions, four-A/fixed-B sequence, two warmup
  aggregates and nine measured aggregates, 1 ms floor. Eighteen processes in
  frozen forward order followed by their exact reverse. No adaptive repetitions,
  deleted samples, resumption or retry-until-win.

Host: AMD Ryzen AI 9 HX 370, Linux `7.0.0-43-generic`, CPU 2 pinned, SMT sibling
14, existing performance governor and boost enabled. No system settings changed.
Before/after load1 was 0.509/0.549 on 24 logical CPUs; observed CPU temperature
43.375/61.000 C (observed campaign maximum 68.625 C). Known build/test process
counts were zero. Affinity/task-count,
thread-CPU and environment checks are bounded observations, not continuous proof
of absent SMT/shared-cache/OS activity or a physical-cache-state guarantee.

## Correctness, consumption and refusals

All six actual source executables were compiled from the relocated SDK. Each
timed executable passed 17 named controls / 469 checks; each corresponding
canonical source-pattern executable passed 68 cases / 111292 checks. Controls
include strict non-FMA/order/underflow/Inf/NaN cases, nondefault caller FP state,
empty geometry, aliasing, owning observations and exact shape/capacity failure
frontiers with no publication. No mismatch was accepted as a noisy timing.

Preparation joined all eight semantic/structured/buffer/Transform/scheduled/
LLVM/object/issuer hashes between each tiled timed leaf and its canonical control.
The three original witnesses agree across requests; six derived identities differ.
Actual executable mappings, runtime/DSO ownership, all 126 pins and 23 SDK entries
were checked. Standard CMake RPATH removal on a **nonexecuted copy** joins the stock
producer to installed bytes; this is inspection, not build-origin attestation.

Two preparation refusals remain intact:

1. `prepared-v1`: root supplied CMake's `lld` path; the emitted source-instance
   linker identity correctly refused it. Actual Clang selection and hashes proved
   `ld.bfd`. No compiler or numerical fix was needed.
2. `prepared-v2`: all six correctness paths passed, but the harness mistook Ninja's
   real DSO target and its phony alias for duplicate targets. The exact-target fix
   and alias-positive/wrong-target-negative tests are in `e34d012`; `--linker` is
   now mandatory. Source mathematics, timing plan and acceptance gates did not change.

`prepared-v3` passed all 50 preparation commands. The physical campaign ran once:
42 processes, 36 measured cells, six initial/final control processes, 324 measured
aggregates, **55728 checked timed calls / 41202432 checked timed output elements**,
plus 12384 verified warmup calls / 9156096 warmup output elements. All output,
source-effect, identity and host gates
passed. Total collection time was 8.050592 seconds within the 180-second cap.

## Timing results, including missing comparisons

Values below are midpoint-of-forward/reverse medians in microseconds per declared
source-call interval, only where that role passed every predeclared timing gate.
An inconclusive role is not repaired by dropping its rejected samples.

| M/K/N | Calls per aggregate | Stock | 3x5 | 5x3 | Qualified comparison to stock |
| --- | ---: | ---: | ---: | ---: | --- |
| 32/32/32 | 128 | inconclusive | 14.1774 | 18.2527 | none: stock below floor |
| 33/35/37 | 128 | 10.1547 | inconclusive | 24.8937 | 5x3 / stock = 2.45145 |
| 64/7/19 | 256 | inconclusive | inconclusive | inconclusive | none |
| 1/257/65 | 256 | inconclusive | 8.7904 | 13.7781 | none: stock preemption |
| 65/257/1 | 256 | inconclusive | inconclusive | inconclusive | none |
| 128/128/128 | 8 | inconclusive | inconclusive | inconclusive | none |

Of 324 measured aggregates, 82 failed a timing gate: 65 were below the 1 ms floor,
19 observed an involuntary switch, with two overlapping both conditions. None
failed thread-CPU/wall or clock-slack checks. Fifteen of 36 directional cells and
six of 18 paired roles qualified; only one pair had a qualified stock comparator.
The 5x3 loss at 33/35/37 is present in both directions (2.45213 and 2.45077), not
a selected favorable direction. No confidence/significance claim is made.

## Reproduction identities and retained raw evidence

Raw campaign root (outside every Git worktree):
`/var/tmp/mdslc-p1-campaign-Mk2kpj/benchmark_reports/`.
The source-bound binaries/private bundles must remain at their emitted paths;
post-emission relocation is unsupported. These local files are not a remote
archive or a guarantee against future temporary-directory retention policy.

| Identity | SHA-256 |
| --- | --- |
| `prepared-v3/prepared.json` | `c2446f8446947df0d1cca2c1182e0357b81a32c14e62490cd679531cde045c63` |
| `collected-v1/collection.json` | `0805df0fd145e7536cf11e4b6ee94dcac7809e2cfa19e849b4620614c12ee828` |
| Frozen plan | `dffefb75511adf63afa8ea647da39e04fdbd18c00f6acb01f561732d3c2f8649` |
| Timed source | `3717a4de99cca90dd0754a676299e39c2435e76cbd373ef0f3b6bbacf441f7e3` |
| Stock timed executable | `3ee207891b90cbf25ddfb303f9eca9db79f8b251841b079598a4ddef2856fce0` |
| 3x5 timed executable | `dc2b4061c1c2a3b7e5213d27fb8b78bd3466f23e4d24e84b2bc64c8df9d1a970` |
| 5x3 timed executable | `afb08dd73822113ae837df16c872b62a515aa6208ee377f9264bc54cb6d3a2e9` |
| Installed stock DSO | `666accbfb66b7166aa0ced1a885c0edc51f197b92d3f8b296042d4ccfff05337` |
| Requested 3x5 DSO | `b384c320a853d4cbf32ac4913171daab38ecbe43335def7f432f6024d56e8e91` |
| Requested 5x3 DSO | `9c738ab09acd47d65a2e357666a0f3c5ce600b38f86a956dad4a4f2f5546e438` |

The preparation receipt retains all source/tool/DSO/stage identities and original
commands. The collection's complete digest catalog retains every raw stdout,
stderr and command receipt. Offline `run.py summarize --collection ...` rejoins
raw chronology, checks all outputs/metadata, recomputes normalization and pairing,
and refuses edited derived results. Root ran this audit successfully. Preparation
generated 1293587 artifact bytes and 144492 raw bytes; collection added 250790 raw
bytes. The existing SDK/toolchain/producer and retained failed preparations are
separately inventoried prerequisites, not hidden inside those per-capsule caps.

## Review and next boundary

Two independent source/profile reviews accepted the bounded harness; the hostile
receipt reviewer separately accepted the narrow alias correction and all actual
preparation receipts. Fifteen synthetic tests cover the final harness and are
registered as a Linux/Python CTest without running a compiler or benchmark in
that test. The independent collection review rechecked all 126 raw catalog files,
42 chronological commands, 74 host snapshots, immutable inputs, status/clock
derivations and paired summaries. It recomputed all six output hashes using
independent integer-prefix arithmetic and `struct` f32 encoding, not the harness's
bit encoder. No remaining high/medium source or evidence-consistency finding was
reported. Hosted integration remains pending at this documentation checkpoint;
it does not authorize merge by itself.

The next justified question is a **bounded code-generation diagnosis of the
33/35/37 stock-versus-5x3 loss**: identify actual scheduled-IR/LLVM/assembly
differences while separating structural observations from causal performance
proof. Do not infer a winning tile, new numerical permission or dispatch threshold.
Broader comparisons need a separately preregistered floor/noise contract, not
reinterpretation or retry of this frozen campaign.
