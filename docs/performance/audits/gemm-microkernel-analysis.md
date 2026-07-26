# GEMM microkernel and instruction-scheduling audit

Date: 2026-07-26

Status: Milestone 6 evidence. No production code or planner behavior changed as
part of this lane.

## Claim labels

Every technical conclusion below carries one of these labels:

- **Measured**: observed in a guarded physical-host run or exact emitted
  artifact.
- **Derived**: arithmetic or control/data-flow reasoning from measured facts.
- **Source-backed**: directly established by the reviewed MDSLC source.
- **Hypothesis**: plausible explanation that still needs a controlled
  experiment.
- **Proposed**: a candidate Milestone 7 experiment, not an accepted
  implementation decision.

## Evidence identity and scope

The audited Release objects and benchmark executable were built from clean
source commit:

```text
951239f1bee5541a4cf5ad72fab2192de07cf89d
```

The toolchain was Ubuntu Clang/LLVM 21.1.8, Release `-O3 -DNDEBUG`. The
physical host was an AMD Ryzen AI 9 HX 370, with 48 KiB L1D and 1 MiB L2 per
core, a performance governor, active boost, and a configured maximum frequency
of 5.157 GHz. The benchmark process was restricted to logical CPU 0 and every
result passed `--guard` and the independent double-precision oracle.

**Measured:** The exact object digests were:

```text
8f86dc591e62e57d56121cb0da3396d87cb2fda3b26bc6493f943949291b5471  cpu_packed_avx2.cpp.o
d1a407221f170f50cc40fb17e51fa838c7b0ece81bd9a17fd255961593e710b1  cpu_packed_avx512.cpp.o
```

Raw object dumps, LLVM-MCA output, optimization records, and guarded schema-v4
JSON are deliberately outside Git under:

```text
/home/hamza-usta/.tmp/mdslc-m6-microkernel.hHPeDS/
```

This is a bounded diagnostic sample, not the complete Milestone 6 performance
matrix. Other builds or host states require new evidence.

## Current microkernel contract

**Source-backed:** AVX2 and AVX-512 share `MR=4`, `NR=16`, `MC=128`,
`NC=256`, and `KC=256`. A 4-by-`KC` A micro-panel is 4 KiB and a
`KC`-by-16 B micro-panel is 16 KiB. The combined 20 KiB micro-panel working set
fits the host's 48 KiB L1D. The complete 128-by-256 A block and 256-by-256 B
block total 384 KiB, which fits the host's 1 MiB private L2 but is not a
universal cache optimum.

**Source-backed:** B is packed in contiguous 16-float vectors and padded with
zeros at N edges. A is packed in groups of four rows and padded at M edges.
The B workspace is 64-byte aligned, so the aligned 32-byte and 64-byte loads in
the AVX2 and AVX-512 kernels remain legal at every K step.

**Source-backed:** Both kernels compute a complete 4x16 tile for every call.
Incomplete M/N tiles use a 64-float stack buffer, copy any accumulated output
into it, compute the full tile, and copy only the legal output region back.
K tails are exact loop trip counts over a packed, padded M/N panel; no
out-of-range speculative vector load was found.

## Exact Release artifact

### AVX2/FMA

**Measured:** The exact symbol
`matcore_cpu_packed_avx2_4x16_microkernel_f32_v1` is 927 bytes. Its steady
K-loop contains:

- two aligned 256-bit B loads;
- four scalar-to-YMM broadcasts from packed A;
- eight YMM `vfmadd231ps` instructions;
- one pointer increment, one trip-count decrement, and one branch.

That is eight independent YMM accumulator chains (`ymm0` through `ymm7`),
two B registers (`ymm8` and `ymm9`), and four broadcast registers (`ymm10`
through `ymm13`). No stack access or vector-register spill occurs inside the
K-loop. The body advances by exactly one K element; Clang emitted no K
unrolling or software prefetch.

**Derived:** Each K step performs
`8 FMAs * 8 lanes * 2 operations = 128 FLOPs` and reads 64 bytes of packed B
plus 16 bytes of packed A. Its panel-level arithmetic intensity is therefore
1.6 FLOP/byte before amortized C traffic.

**Measured:** LLVM-MCA 21 with `-mcpu=znver5` predicts a four-cycle block
throughput, 4.10 instructions per cycle, and primary FP-resource pressure for
this exact loop. The model assigns four-cycle FMA latency and 0.5-cycle
throughput to a YMM FMA. Eight independent accumulators are therefore enough
to cover the modeled recurrence latency at two YMM FMAs per cycle.

**Derived:** Under that model the AVX2 hot loop reaches 32 FLOP/cycle. Merely
adding K unrolling cannot raise that modeled FP execution ceiling, although it
could reduce branch/address overhead or change real-hardware scheduling.

### AVX-512F/FMA

**Measured:** The exact symbol
`matcore_cpu_packed_avx512_4x16_microkernel_f32_v1` is 745 bytes. Its steady
K-loop contains:

- one aligned 512-bit B load;
- four ZMM `vfmadd231ps` instructions with embedded scalar broadcasts from
  packed A;
- one pointer increment, one trip-count decrement, and one branch.

Only four ZMM accumulators (`zmm0` through `zmm3`) and one B register (`zmm4`)
are live in the hot loop. No stack access or vector-register spill occurs
inside that loop. Clang emitted no K unrolling or software prefetch.

**Derived:** It also performs 128 FLOPs and reads 80 panel bytes per K step.
Moving from AVX2 to AVX-512 does not increase the register tile, total useful
work, or panel reuse; it only halves the number of vector FMA instructions.
Twenty-seven of the 32 architectural ZMM registers remain unused by the hot
loop.

**Measured:** LLVM-MCA 21 predicts the same four-cycle block throughput and
32 FLOP/cycle as AVX2 for this body. It reports four memory-operand FMAs with
11-cycle modeled latency, 1.0-cycle reciprocal throughput, load-queue pressure,
and a loop-carried dependency contribution. The four accumulators provide much
less latency-hiding distance than the AVX2 kernel's eight.

**Important model limitation:** Although `-mcpu=znver5` was requested, LLVM's
reported resource names are still `Zn4*`, and this compiler model is not
physical counter evidence for the host's Zen 5 AVX-512 datapath. It is useful
for comparing the emitted dependency structure, but it cannot establish the
actual 512-bit FMA width or frequency response.

## Per-call overhead outside the K-loop

**Measured:** Both exact objects initialize the 256-byte edge array before the
complete-tile branch:

- AVX2 emits eight aligned 32-byte zero stores;
- AVX-512 emits four aligned 64-byte zero stores.

This zero fill occurs even for a complete 4x16 tile. Both functions also
perform null, K, row-count, and column-count checks per microkernel call and
reserve a stack frame of 352 bytes (AVX2) or 384 bytes (AVX-512).

**Derived:** For a full tile at `K=256`, the zero fill is small compared with
32,768 floating-point operations, but it is a pure per-tile tax and becomes
proportionally important for small K. The validation branches and noinline
call boundary likewise repeat for every microtile even though their callers
already hold validated packed-panel contracts.

**Proposed:** Split a minimal, prevalidated full-tile hot symbol from a checked
edge wrapper. Allocate and initialize edge storage only on the incomplete-tile
path. Preserve the current checked entry point for direct testability and
fail-closed behavior; do not expose the unchecked symbol outside the internal
packed executor.

## Bounded physical-host timing

The command family was:

```sh
taskset -c 0 matcore-bench \
  --m M --n N --k K --variant VARIANT --threads 1 \
  --warmup 3 --iterations 9 --timer-floor-us 5000 \
  --hot-cache --include-packing --reuse-workspace --guard \
  --json-out RAW_EXTERNAL_PATH
```

**Measured:** Complete implementation results were:

| Shape | Packed AVX2 GFLOP/s | Packed AVX-512 GFLOP/s | AVX-512 / AVX2 |
| --- | ---: | ---: | ---: |
| 128x128x128 | 110.21 | 111.39 | 1.011 |
| 256x256x256 | 127.72 | 129.90 | 1.017 |
| 512x512x512 | 130.53 | 134.98 | 1.034 |
| 1024x1024x1024 | 123.16 | 129.50 | 1.051 |
| 511x513x515 | 123.44 | 128.26 | 1.039 |

**Measured:** At 512 square, caller-owned prepacked B increased AVX2 from
130.53 to 133.40 GFLOP/s (1.022x) and AVX-512 from 134.98 to
138.42 GFLOP/s (1.025x). B packing is therefore measurable but is not the
dominant limitation for this one repeated square case.

**Measured:** The AVX2-only diagnostic that packs complete A and B before
timing reached 127.29, 138.81, 136.43, and 137.62 GFLOP/s at squares
128, 256, 512, and 1024 respectively.

**Important comparison limit:** That diagnostic is not simply "complete time
minus packing." It uses a full-K packed layout and one microkernel K span,
whereas the complete implementation uses `KC=256`, may reload/store C across K
blocks, and includes persistent-worker dispatch. It establishes an
approximately 137 GFLOP/s current compute-path plateau, not a clean packing
duration.

**Derived:** The physical timing agrees with the artifact-level warning: on
this bounded sample, AVX-512 is only 1--5% faster than AVX2, not close to a 2x
width gain. This does not isolate whether the residual cause is accumulator
depth, actual execution width, clock behavior, or outer-loop/cache effects.

## Tail behavior

**Measured:** For AVX2 compute-only one-microtile diagnostics, a 3x15 edge tile
and a full 4x16 tile took nearly the same time at K=8, 64, and 256. The edge
tile therefore delivered about 70% of the full-tile useful throughput, close
to its useful-output ratio `45/64 = 0.703`, because it still computes all 64
lanes. At K=1 the edge call was approximately 8% slower in time, but both calls
were dominated by the roughly 2 microsecond persistent-dispatch interval.

**Measured:** Large single-axis tails (511x512x512, 512x511x512, and
512x512x511) did not show a stable material penalty relative to 512 square in
this sample. In contrast, 511x513x515 was about 5% lower in useful throughput.

**Derived:** The 513-column case crosses the `NC=256` boundary and creates a
third N macro-panel for one useful column. Its loss cannot be assigned to the
microkernel edge code alone; it also changes packing and A-panel reuse.

**Proposed:** Add masked or narrow edge kernels only after separating the edge
copy cost from macro-panel boundary repacking. A family such as 4x8, 4x4, or
masked 4x16 may improve useful tail efficiency, but a third tiny NC panel is
an outer blocking problem as well.

## Load, address, dependency, and prefetch assessment

**Source-backed:** The AVX2 K-loop has six load instructions and eight
register-register FMAs per K step. LLVM-MCA predicts FMA-resource pressure,
not AGU saturation, at the current tile. The eight accumulators avoid a
modeled FMA-latency bottleneck, and no hot-loop spill was found.

**Source-backed:** AVX-512 performs one vector load and four
memory-broadcast FMAs per K step. Its four accumulator recurrences are the
smallest credible latency-hiding set for this tile, while the wide register
file is mostly unused.

**Hypothesis:** An AVX-512 8x16 kernel (eight accumulators) or 4x32 kernel
(eight accumulators) could increase recurrence distance and reuse each loaded
operand more effectively. The better shape is host-specific: 8x16 increases A
broadcasts, while 4x32 increases B-vector loads and edge overcompute. Both need
exact object inspection, correctness tests, and complete-call measurements.

**Hypothesis:** A separately loaded/broadcast A strategy plus register-register
AVX-512 FMAs may schedule differently from the current embedded-broadcast
memory operands. LLVM-MCA alone is insufficient to choose it; benchmark both
forms with equal register tiles.

**Source-backed:** Neither kernel emits explicit prefetch instructions.

**Hypothesis:** Software prefetch is unlikely to repair the in-L1
micro-panel loop by itself because packed accesses are sequential and the
20 KiB active panels fit L1D. Prefetch may matter at the outer packed-panel or
original-matrix packing boundary, particularly for cold-cache or rectangular
problems. Treat prefetch distance as a controlled experiment; do not add an
unconditional host-tuned distance.

## Frequency evidence and counter limitation

**Measured:** The governor was `performance`, boost was active, and sysfs
reported a 5.157 GHz configured maximum. `cpupower` could not read a
hardware-derived current frequency. `turbostat` could not access MSRs and
failed in its no-MSR path on this host. `perf stat` was unavailable because
`kernel.perf_event_paranoid=4`; no system policy was changed.

**Derived:** No workload-averaged APERF/MPERF evidence exists in this lane.
Consequently, the audit cannot attribute the small AVX-512 advantage or the
older variable AVX-512 results to downclocking. Any such attribution remains a
hypothesis until a privileged, controlled frequency capture is approved.

## Ranked findings

1. **High confidence / high expected AVX-512 impact — derived:** the AVX-512
   implementation reuses a 4x16 tile and only four accumulators, leaving most
   ZMM registers idle. It does not expose enough independent work to establish
   a wide-ISA throughput advantage.
2. **High confidence / medium small-K impact — measured:** every full-tile call
   zeroes a 256-byte edge buffer and executes validation/prologue logic that is
   unnecessary after caller-side validation.
3. **High confidence / medium tail impact — measured and derived:** an edge
   call computes all 64 output lanes; useful efficiency falls in proportion to
   tail occupancy. Crossing an NC boundary compounds this with outer packing
   overhead.
4. **Medium confidence / medium impact — derived:** the one-step K loop has no
   software pipelining. AVX2 is already at the LLVM model's FP-resource ceiling,
   but AVX-512 needs wider register tiles and then K-unroll experiments to
   determine whether dependency or front-end overhead remains.
5. **Low confidence until counters / host-specific impact — hypothesis:**
   AVX-512 frequency behavior may change crossovers, but this audit has no
   APERF/MPERF evidence.
6. **Low confidence / shape-specific impact — hypothesis:** explicit outer-loop
   prefetch may help cold or rectangular panels; the in-L1 K loop does not
   present evidence for it.

## Milestone 7 experiment order

1. **Proposed:** Remove unconditional full-tile edge-buffer initialization via
   a private validated full-tile symbol; object-test that its K loop and
   prologue contain no stack traffic.
2. **Proposed:** Prototype AVX-512 8x16 and 4x32 kernels separately. Require at
   least eight accumulator chains, no spills, exact ZMM FMAs, complete tails,
   and a generic-binary ISA gate.
3. **Proposed:** After choosing the AVX-512 register tile, compare K-unroll
   factors 1, 2, and 4 with LLVM-MCA and physical complete-call timing.
4. **Proposed:** Add narrow/masked edge kernels, but evaluate NC-boundary
   packing separately so the microkernel is not credited for an outer-loop
   fix.
5. **Proposed:** Test prefetch only for cold-cache and large/rectangular panels,
   with a no-prefetch control and explicit host-bound results.
6. **Proposed:** Obtain privileged APERF/MPERF and hardware-counter evidence in
   a separately approved run before making any AVX-512 downclock claim.

## Validation performed

```text
ctest -R runtime.cpu.(packed_avx2|packed_avx512|
                     packed_avx2_object|packed_avx512_object)
4/4 passed
```

The exact artifact tests confirmed YMM/ZMM packed FMA instructions. All 33
bounded guarded timing files produced for this lane were timing-valid and
correct. No production source, public ABI, planner rule, or package behavior
was changed.

## Claims not supported

- This audit does not claim BLAS parity.
- It does not claim that AVX-512 downclocks on this host.
- It does not claim that LLVM-MCA's reused `Zn4*` resource model is an exact
  Zen 5 hardware model.
- It does not prove that one wider register tile is optimal.
- It does not compare compute-only MDSLC timing with a complete OpenBLAS call.
- It does not justify public API or ABI changes.
