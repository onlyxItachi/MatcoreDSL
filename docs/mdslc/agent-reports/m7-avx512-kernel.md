# Milestone 7 AVX-512 kernel lane

Date: 2026-07-26

Owner: AVX-512 packed F32 microkernel, its focused correctness test, exact
artifact test, and this report.

## Outcome

The Milestone 6 audit found that the original AVX-512 4x16 kernel had only four
accumulator recurrence chains and left most of the ZMM register file unused.
This lane added a private 4x32 full-tile kernel with eight accumulators. The
serial packed executor uses it only for complete 4x32 tiles and retains the
checked 4x16 implementation for every M/N edge.

The implementation:

- reuses the version-1 A and B packed layouts, blocking constants, caller-owned
  workspace, prepacked-B representation, and capability gate;
- consumes two adjacent 16-column B micro-panels without repacking;
- leaves the public C ABI and installed headers unchanged;
- exposes the callback only through the private runtime header so the parallel
  packed executor can reuse the same body;
- keeps the exact symbol ELF-hidden (it is absent from `nm -D` output);
- does not change planner selection or variant IDs.

Commits:

1. `89023a37c5d0e720d9a682d388875de77ac5aa0f` —
   `feat(cpu): widen the AVX-512 full-tile kernel`
2. `c2aa68afb53ed5b1dbdf09c299a47c5f116361f3` —
   `refactor(cpu): expose the private AVX-512 tile callback`
3. `032dd3e3da8e8c14a8aba0043c5c57172e04c2da` —
   `test(cpu): distinguish debug AVX-512 artifacts`

## Correctness and safety

The focused test now explicitly exercises:

- one 4x32 full tile;
- adjacent 4x32 tiles;
- multiple 4-row panels;
- an M edge after complete full tiles;
- a second `KC` accumulation block;
- misaligned but contract-valid inputs;
- randomized M/N/K tails;
- cross-ISA prepacked-B interoperability;
- insufficient/misaligned workspace and alias rejection.

Results:

```text
Release focused CTest: 2/2 passed
Debug focused CTest:   2/2 passed
ASan+UBSan executable: passed, including leak detection
```

The existing guards and independent double-precision oracle passed. The
workspace size, alignment, provenance, alias, and fail-before-mutation paths
are unchanged.

## Exact artifact evidence

Release source checkpoint:

```text
032dd3e3da8e8c14a8aba0043c5c57172e04c2da
```

Release AVX-512 object SHA-256:

```text
063dcd7595b3f384bd9bf05fa11537cf877e68d636b4d4a2ed65120cdf5a8ba5
```

The exact 280-byte symbol
`matcore_internal_cpu_packed_avx512_4x32_full_microkernel_f32_m7`
contains:

- eight independent ZMM accumulators;
- two aligned ZMM B loads;
- four ZMM scalar broadcasts;
- eight packed `vfmadd231ps` instructions per K step;
- 14 distinct ZMM registers;
- no stack access or spill in the steady K loop.

`readelf -Ws` reports the symbol as `GLOBAL HIDDEN`; it is not present in the
runtime shared library's dynamic symbol table. The existing checked 4x16 edge
symbol remains intact.

The artifact test accepts Debug only when debug information is actually
present and the exact eight packed FMA instructions remain. The unoptimized
Debug object spills and uses only three distinct ZMM register names, so it is
reported as `optimized_register_tile=false`; no Debug performance claim is
made. Release must retain at least ten distinct ZMM registers and reports
`optimized_register_tile=true`.

## Bounded before/after measurement

The final comparison used clean detached source trees, forced
`cpu.native-packed.avx512-fma.f32.v1`, `taskset -c 0`, one requested thread,
hot cache, transient packing included, reused workspace, two warmups, seven
normalized samples per process, a 5 ms timer floor, `--guard`, and ABBA
process order. Each table cell is the median of two independently launched
baseline or candidate processes. Both source trees reported clean provenance;
every result was timing-valid and passed the independent oracle.

Baseline:

```text
ddda3ccf628dae60bdb7f57d68d024fd02168fcb
```

Candidate used for the guarded sweep:

```text
c2aa68afb53ed5b1dbdf09c299a47c5f116361f3
```

| MxNxK | Baseline GFLOP/s | 4x32 GFLOP/s | Speedup |
| --- | ---: | ---: | ---: |
| 256x256x256 | 127.66 | 137.83 | 1.080x |
| 512x512x512 | 133.57 | 142.23 | 1.065x |
| 1024x1024x1024 | 130.81 | 140.25 | 1.072x |
| 1536x1536x1536 | 126.47 | 135.22 | 1.069x |
| 4096x4096x64 | 136.14 | 144.89 | 1.064x |
| 64x4096x4096 | 88.61 | 91.26 | 1.030x |
| 511x513x515 | 124.53 | 133.46 | 1.072x |

Median speedup was 1.069x, geometric-mean speedup 1.064x, minimum 1.030x,
and maximum 1.080x over this bounded set. The small gain for
`64x4096x4096` agrees with the audit's conclusion that the small-M transient
path is preparation-sensitive; widening the compute tile cannot remove that
cost.

Raw JSON remains outside Git under
`/home/hamza-usta/.tmp/mdslc-m7-avx512-final-isolated-abba/`.
The normalized run table SHA-256 is:

```text
0115c0bbc046fbb8d421693cafb2b1c8605fb26b659fe9e588431827d19e2f1d
```

These are host-bounded single-thread results on the AMD Ryzen AI 9 HX 370.
They are not a universal AVX-512 or BLAS-parity claim. Full Milestone 7
planner, OpenBLAS, parallel, Windows, and cross-agent performance validation
remain integration-owner gates.
