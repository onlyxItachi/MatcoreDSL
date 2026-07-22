# Milestone 4 native packed AVX2/FMA kernel lane

Date: 2026-07-22

## Scope and ownership

This lane owns only the new compiled packed-backend sources, their isolated
tests, and the exact-function disassembly gate. It does not modify the public C
ABI, existing planner, generated code, runtime dispatcher, or CMake targets.
Those integration points remain owned by the lead/runtime lane.

## Implemented contract

`cpu_gemm_backend.h` defines a versioned internal C++ backend API around the
existing `CpuGemmProblemV1`. It provides:

- explicit transient A+B and transient-A-with-prepacked-B workspace modes;
- overflow-checked workspace and persistent packed-B storage queries;
- an exact 64-byte workspace-alignment requirement;
- explicit status values for invalid problems, pointers, alignment, aliases,
  overflow, unavailable ISA, insufficient workspace, and invalid prepacked B;
- caller-owned persistent packed-B storage with a versioned borrowed view and
  deterministic provenance validation;
- no allocation, exception, copy outside caller storage, or ambient fallback.

All validation completes before the first output store. The implementation
rejects output/input overlap, partial overlap, workspace/tensor overlap,
workspace/prepacked-storage overlap, malformed prepacked metadata, insufficient
storage, and actual pointer alignment weaker than the declared problem
contract. A rejected call does not mutate output.

## Kernel architecture

The deterministic v1 blocking parameters are:

| Parameter | Value |
| --- | ---: |
| MR | 4 |
| NR | 16 |
| MC | 128 |
| NC | 256 |
| KC | 256 |
| Workspace alignment | 64 bytes |

The 4x16 microkernel holds eight YMM accumulators, two B vectors, and four
broadcast A operands without exhausting the sixteen-register AVX2 register
file. At KC=256, one A micro-panel is 4 KiB and one B micro-panel is 16 KiB.
The maximum transient packed A and B macro-panels are 128 KiB and 256 KiB,
respectively. Their 384 KiB total is a conservative L2-scale working set on
the validation host; these parameters are not claimed to be universally
optimal.

Both A and B are explicitly packed. Missing M/N tail lanes are zero padded.
The exact noinline `target("avx2,fma")` microkernel handles full tiles directly
and uses a bounded 4x16 local edge tile for partial M/N stores, so it performs
no out-of-bounds input load or output store. K tails are represented by the
actual packed block depth. Multiple KC blocks accumulate into C only after the
first block.

Persistent B uses the same NC/KC/NR panel order as transient packing. Its
storage requirement is `K * round_up(N, 16) * sizeof(float)`, checked for
overflow. Execution requires only the caller-owned transient A panel and
validates the view's source identity, shape, blocking version, storage size,
alignment, and provenance before use. Source contents must remain unchanged
while the view is live.

## Correctness and adversarial validation

The isolated test uses seeded inputs and an independent double-precision
oracle. It covers:

- tiny, square, rectangular, aligned, and 4-byte-aligned inputs;
- full 4x16 tiles and complete M/N/K tails;
- shapes crossing MC, NC, and KC boundaries;
- 40 additional seeded randomized shapes;
- persistent prepacked-B preparation and three repeated executions;
- insufficient, null, and misaligned workspace;
- exact and partial output/input overlap;
- workspace/input overlap and output/packed-storage overlap;
- declared-versus-actual tensor alignment mismatch;
- invalid and overflowing problems;
- insufficient and source-overlapping prepacked storage;
- corrupted prepacked provenance;
- input, output, workspace, and packed-storage guard preservation.

Release-like validation:

```text
/usr/bin/clang++-21 -std=c++20 -O3 -g \
  -Wall -Wextra -Wpedantic -Werror \
  -Icompiler/lib/planner -Icompiler/lib/runtime \
  compiler/lib/runtime/cpu_gemm_backend.cpp \
  compiler/lib/runtime/cpu_packed_avx2.cpp \
  compiler/tests/runtime/packed/packed_gemm_test.cpp \
  -o /tmp/mdslc-m4-kernel-audit/packed_gemm_test
/tmp/mdslc-m4-kernel-audit/packed_gemm_test
```

Result:

```text
native packed AVX2/FMA GEMM: all tests passed
```

ASan/UBSan validation used `-O1 -fsanitize=address,undefined
-fno-sanitize=pointer-overflow -fno-omit-frame-pointer` with leak detection.
The complete suite passed without a sanitizer report.

The same test passed under Clang 21 `-O0` and GCC 15.2 `-O3`, both with
`-Wall -Wextra -Wpedantic -Werror`. Clang static analysis reported no finding.

## Exact instruction evidence

The disassembly gate resolves and disassembles only:

```text
matcore_cpu_packed_avx2_4x16_microkernel_f32_v1
```

It requires a defined text symbol, at least one YMM operation, and packed
single-precision `vfmadd132ps`, `vfmadd213ps`, or `vfmadd231ps`. Both the `-O3`
and `-O0` artifacts passed. The `-O3` artifact contained eight static
`vfmadd231ps` instruction sites and YMM registers through `ymm13` in the exact
function.

## Integration notes and limitations

- The lead should compile the two backend sources into the runtime and wrap
  this C++ API behind the additive workspace-aware C ABI. The legacy one-shot
  ABI must not select this variant because it supplies no caller workspace.
- Planner legality must require x86-64, usable AVX2/FMA state, compiled backend
  availability, exact supported semantics, and sufficient aligned workspace.
- The implementation performs its own fail-closed runtime ISA check before the
  target-specific function call. Synthetic planner capabilities must never
  bypass that physical execution gate.
- Prepacked-B provenance catches stale or structurally incompatible views but
  is not a security primitive. The caller remains responsible for source and
  storage lifetime and immutability.
- No benchmark or crossover claim is made by this lane. Planner promotion must
  wait for the common Milestone 4 benchmark contract and calibrated evidence.
- Windows compilation and artifact validation were not available locally.
  The target attribute is isolated to the microkernel; the later Windows lane
  must validate clang-cl support and provide the platform-specific capability
  gate before claiming runtime support.
