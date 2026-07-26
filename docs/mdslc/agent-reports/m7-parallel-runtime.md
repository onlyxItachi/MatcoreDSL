# Milestone 7 parallel-runtime lane

Date: 2026-07-26

Branch: `mdslc/native-blas-parity-v1`

Ownership:

- `compiler/lib/runtime/cpu_parallel_gemm.cpp`
- private AVX2 callback declaration in
  `compiler/lib/runtime/cpu_gemm_backend.h` after lead handoff
- `compiler/tests/runtime/parallel/parallel_packed_gemm_test.cpp`
- this report

## Evidence-backed problem

The Milestone 6 audit established two deterministic decomposition failures:

1. the old task count was `ceil(M / 128)`, so `M=64` exposed one worker even
   when `N*K` was large;
2. `M=129` produced fixed row loads `[128, 1]`.

It also established that packed B is already one immutable caller-owned image
shared safely across workers, while each worker has an isolated transient-A
workspace. The change below preserves those ownership contracts and does not
split K or introduce a reduction.

## Implemented task formula

Let:

- `MR=4`, `MC=128`, `NC=256`;
- `row_quantum = lcm(4, 16 / gcd(N, 16))`;
- `row_groups = ceil(M / row_quantum)`;
- `macro_tiles = ceil(M / 128)`;
- `column_panels = ceil(N / 256)`;
- `max_rows = min(requested_threads, macro_tiles, row_groups)`.

The default grid is `max_rows x 1`. N partitioning is considered only when:

- the declared problem alignment is at least 64 bytes;
- the actual C pointer is 64-byte aligned;
- `N % 16 == 0`;
- there is more than one 256-column panel.

For each `r` in `[1, max_rows]`, the runtime evaluates:

`c = min(column_panels, floor(requested_threads / r))`.

It selects the grid maximizing `r*c <= requested_threads`; ties prefer larger
`r` so A packing is duplicated only when M alone cannot expose the requested
parallelism. A column-only grid is accepted only when saturated
`2*M*N*K/(r*c) >= 2^23`. A genuinely two-dimensional grid, which duplicates A
packing across column tasks, requires `2*M*N*K/(r*c) >= 2^25`. Otherwise
execution remains row-only. The stricter two-dimensional floor excludes the
measured `129x512x512` regression described below.

Row groups and column panels are quotient/remainder balanced. Task identity is
`row_task * column_task_count + column_task`. K is never split. Each task owns
one disjoint rectangle of C and traverses K in the same order as the serial
packed backend.

Representative grids:

| Problem | Threads | Old tasks | New grid |
| --- | ---: | ---: | ---: |
| `64x1024x1024` | 4 | 1 | `1x4` |
| `129x1024x512` | 4 | 2 (`128/1` rows) | `2x2`, balanced row groups |
| `1024x1024x1024` | 4 | 8 | `4x1` |

Low-declared-alignment or low-actual-alignment problems remain row-only even if
an allocator happens to provide stronger incidental alignment. This keeps
runtime behavior identical to the planner-visible contract.

## Packing and kernel integration

- B is packed once before dispatch into the existing authenticated packed-B v1
  format.
- A remains packed into non-overlapping per-worker workspace.
- Column-task boundaries are complete `NC=256` panels. In packed-B v1, every
  preceding complete panel contains exactly `256*K` floats, so the panel base
  is `packed_data + column_begin*K`.
- Within a panel, the KC block offset advances by
  `round_up(panel_columns, 16) * depth`. Checked arithmetic proves every
  interval remains inside `packed_elements` before use.
- Row-only tasks delegate to the existing checked prepacked backend, preserving
  its measured hot path.
- AVX2 full 4x16 tiles call the same private v2 full-tile body as serial packed
  execution.
- AVX-512 full 4x32 tiles call the same private full-tile body as serial packed
  execution. Remaining 4x16 and edge tiles use the checked wrapper.
- No microkernel, packing, planner, or public ABI identifier changed.

## Correctness and adversarial coverage

The focused test adds:

- aligned AVX2 `64x257x1024` four-worker 2-D execution;
- aligned AVX2 `129x129x512` balanced-wave 2-D execution;
- aligned AVX2 M/K-tail `63x193x1024`;
- low-alignment row-only execution even with a four-thread request;
- output guard regions proving disjoint C ownership;
- insufficient four-worker workspace rejection before output mutation or
  worker submission;
- AVX-512 2-D correctness for both one- and two-MC-row problems;
- double-precision independent oracles for every new execution.

## Validation

Focused Release correctness passed on the physical AVX2/AVX-512 validation
host. A bounded direct-runtime ABBA diagnostic, pinned to CPUs 4--7, compared
the old and new implementations with four requested threads:

- `M=64,K=1024,N=1024`: median-of-process-medians improved from
  approximately 1.91 ms (old one-worker path) to 1.59 ms (new `1x4` grid),
  about 1.20x.
- `M=129,K=512,N=512`: an initial `2x2` grid regressed from approximately
  0.842 ms to 0.935 ms. This result directly motivated the `2^25`
  two-dimensional work floor; the production formula now retains the old
  two-worker row path for that problem.
- `1024^3`: row-only delegation remained neutral within diagnostic noise
  (approximately 12.86 ms old and 12.84 ms new).

Every process reported the same deterministic checksum for old and new paths.
These are bounded engineering diagnostics under a process CPU mask, not final
Milestone 7 parity or universal performance claims. Clean-tree integration,
sanitizer, and complete benchmark validation remain lead-owned.

## Limitations

- N partitioning deliberately uses complete 256-column panels. It does not
  split within an NR panel.
- N partitioning is disabled for `N % 16 != 0` and low-alignment output.
- Column tasks duplicate A packing. M-first tie-breaking and the static
  work-per-task floor bound that cost; further shared-A designs require a
  different explicit lifetime/synchronization contract.
- The public report remains ABI v1 and therefore reports legacy macro-tile
  count, not the private 2-D grid. Planner diagnostics own the user-visible
  decomposition explanation.
