# Milestone 7 AVX2 full-tile kernel audit

## Scope and provenance

This lane independently audited commit
`e778c64f1d7c527ff6d509599a713c937b151c2e`
(`feat(cpu): add AVX2 full-tile hot path`) on the integrated
`mdslc/native-blas-parity-v1` tree at
`c8e0bfbe775e44609ad56a06c079a3ed45b6b768`.
The lane owned only this report. No runtime, planner, public header, test, or
package source was changed.

The exact comparison points were:

- parent: `ddda3ccf628dae60bdb7f57d68d024fd02168fcb`;
- candidate: `e778c64f1d7c527ff6d509599a713c937b151c2e`.

Both benchmark binaries were built from clean detached worktrees. Raw JSON,
disassembly, symbol listings, and build trees remain outside Git under
`/home/hamza-usta/archives/MatcoreDSL-m7-avx2-audit-20260726/`.

The validation host was an AMD Ryzen AI 9 HX 370, Linux
7.0.0-27-generic, with the performance governor and boost enabled. Builds used
Clang/LLVM 21.1.8 and OpenBLAS 0.3.32 was discovered coherently through
pkg-config.

## Source and contract review

The change isolates the already existing 4x16 AVX2/FMA calculation in
`matcore_cpu_packed_avx2_4x16_full_microkernel_f32_v2`. Serial packed
execution now calls that symbol directly for complete 4x16 tiles. The checked
v1 edge wrapper remains responsible for partial tiles and continues to stage
them through its guarded 4x16 stack buffer.

The fast symbol deliberately does not validate pointers, K, or output extent.
That is safe under the current private contract:

- descriptor, alias, alignment, workspace, and positive-dimension validation
  occurs before packed execution;
- the serial executor calls it only when both tile dimensions are complete;
- the parallel executor calls it only for a complete 4x16 tile and valid packed
  panel extents;
- K is never split into a zero-depth block;
- partial rows or columns retain the checked edge wrapper.

The added `{8, 513, 32}` packed test shape exercises complete 4x16 tiles,
multiple KC panels, a one-element K tail, and the accumulate path between
panels. The existing matrix also covers complete tiles, M/N/K tails,
misalignment, prepacked-B reuse, workspace failures, aliases, and seeded
randomized shapes.

No high-severity correctness, memory-safety, or ABI defect was found.

## Exact artifact evidence

The Release backend archive defines the exact production symbol:

```text
00000000000000e0 T matcore_cpu_packed_avx2_4x16_full_microkernel_f32_v2
```

The isolated Release function has:

- 52 decoded instructions;
- eight `vfmadd231ps` instruction sites;
- YMM registers `ymm0` through `ymm13` in use;
- eight independent 256-bit accumulator registers;
- two aligned packed-B vector loads per K step;
- four scalar-to-YMM broadcasts per K step;
- no `rsp` or `esp` reference;
- no stack allocation or spill;
- one `vzeroupper` before return.

The parent’s checked wrapper had the same eight packed-FMA sites but decoded to
213 instructions, allocated `0x160` bytes of stack, and contained 45
stack-referencing instructions because the full and edge paths shared one
entry. The new symbol removes that wrapper/prologue cost for production full
tiles; it does not change MR=4, NR=16, the packed layout, or the eight-FMA
inner-loop schedule.

Debug correctness is supported, but Debug is not a performance artifact. Its
exact full symbol retained eight packed FMA sites yet decoded to 767
instructions, used only `ymm0` through `ymm2`, and contained 645
stack-referencing instructions. All performance conclusions below therefore
apply only to the Release artifact.

## Hidden/public boundary

The symbol is global inside the private static backend archive so the serial
and parallel runtime translation units can share it. In the linked shared
runtime, `readelf -Ws` classifies it as `LOCAL`. It is absent from all 15
symbols returned by `nm -D --defined-only`.

The declaration resides only in
`compiler/lib/runtime/cpu_gemm_backend.h`; that header is not installed.
`compiler/include/matcore/runtime_c.h` does not declare the symbol. Linux links
the private archive with `--exclude-libs,ALL`, while the Windows DLL disables
automatic symbol export. The full microkernel is therefore an internal
implementation callback, not a stable C ABI addition.

## Focused correctness and sanitizer evidence

All results below were produced from the integrated tree:

| Configuration | Command scope | Result |
|---|---|---:|
| Release | `runtime.cpu.packed_avx2` | 1/1 passed |
| Release | `runtime.cpu.packed_avx2_object` | 1/1 passed |
| Debug | `runtime.cpu.packed_avx2` | 1/1 passed |
| Clang ASan+UBSan Debug/O1 | `runtime.cpu.packed_avx2` | 1/1 passed |

The sanitizer build used:

```text
-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
```

Independent direct executions of the exact parent and candidate Release test
binaries both printed:

```text
native packed AVX2/FMA GEMM: all tests passed
```

## Guarded ABBA performance comparison

A first ABBA attempt was discarded before review because an unrelated
multi-core `torch-xdna` pytest and an integration benchmark sweep were active
during sampling. None of those contaminated numbers is used as evidence.

The retained run began after both processes exited. Immediately before the
run, `mpstat -P 0,12 1 2` reported CPU 0 95.94% idle and its SMT sibling CPU
12 97.00% idle. The system used the performance governor; boost was enabled;
the recorded policy range was 605264--5157000 kHz. Both exact binaries ran on
CPU 0 under `taskset`; all BLAS/OpenMP thread-control variables were fixed to
one even though the forced native variant does not call BLAS.

Each row used A-B-B-A process order, five warmups, eleven measured aggregate
samples, a 5 ms timer floor, hot cache, allocation-excluded reused workspace,
packing included, 64-byte alignment, one thread, and `--guard`. Each JSON
record authenticated the requested native AVX2 variant, exact clean source
commit, valid timing, and double-oracle correctness. The table reports the
arithmetic mean of the two process-level medians for each revision:

| MxNxK | Parent median (ms) | Candidate median (ms) | Parent/candidate | Candidate delta | Max within-revision spread |
|---|---:|---:|---:|---:|---:|
| 4x16x8 full tile | 0.002441 | 0.002501 | 0.9760x | -2.40% | 4.05% |
| 128x128x128 | 0.037588 | 0.038541 | 0.9753x | -2.47% | 0.98% |
| 512x512x512 | 2.013767 | 2.026428 | 0.9938x | -0.62% | 0.18% |
| 64x1024x1024 short-wide | 1.169265 | 1.179181 | 0.9916x | -0.84% | 0.58% |
| 1024x64x1024 tall-narrow | 1.111887 | 1.216147 | 0.9143x | -8.57% | 11.07% |
| 255x257x259 tail-heavy | 0.291170 | 0.296594 | 0.9817x | -1.83% | 0.38% |

The tall-narrow comparison is too unstable to quantify because the two
candidate medians differ by 11.07%; it is retained as a diagnostic only. The
other comparisons are stable and show no complete-call speedup. The candidate
is 0.62--2.47% slower on all five stable shapes.

This result does not contradict the assembly improvement: it shows that
removing the checked wrapper's stack prologue is not a material end-to-end
cost at the tested KC depth. The caller also gained a full-versus-edge branch
and a different out-of-line call/code-layout path. The exact causal split was
not measured here, so it would be incorrect to attribute the small regression
to one instruction in isolation.

The fast symbol can still be useful as the shared prevalidated callback for
parallel execution. However, the serial routing change must not be cited as a
Milestone 7 performance improvement. It should either be retained for that
internal reuse with the regression documented, or independently reworked and
remeasured before promotion.

## Findings

### Medium: the registered artifact test names the checked wrapper

`runtime.cpu.packed_avx2_object` still disassembles
`matcore_cpu_packed_avx2_4x16_microkernel_f32_v1`. It proves AVX2/FMA exists in
the edge-capable wrapper, but it does not lock the newly selected production
full-tile symbol against future scalarization or spills. This audit directly
verified the correct symbol, but the durable CTest gate should inspect
`matcore_cpu_packed_avx2_4x16_full_microkernel_f32_v2` and require the Release
properties recorded above. This is a coverage issue, not a defect in the
current machine code.

### Medium: the serial hot-path change did not improve complete calls

The retained ABBA comparison shows a small but consistent 0.62--2.47%
regression on every stable representative shape. The instruction-level
cleanup is real, but its intended end-to-end benefit is not. This commit alone
does not advance the native-BLAS parity target, and no speedup claim should be
made from it.

### Low: Debug is intentionally unsuitable for performance evidence

The target-specific Debug body is correct and still contains packed AVX2/FMA,
but its extensive stack traffic makes it unsuitable for an ISA-performance
claim. Release remains the artifact acceptance configuration.

## Verdict

Correctness, sanitizer, source-contract, exact machine-code, and hidden-symbol
checks pass. The new Release symbol is a legitimate private AVX2/FMA
microkernel and is safe for its prevalidated call sites.

Performance promotion fails: the retained complete-call evidence shows no
serial speedup and small stable regressions. No production edit was made by
this audit lane, but integration should address the stale artifact gate and
must not count `e778c64` as measured parity progress.

## Integration resolution

The integration branch resolved both medium findings after this independent
lane froze its evidence:

- `a7f2eff` registered the exact full-tile symbol in
  `runtime.cpu.packed_avx2_object`, while retaining the checked edge-wrapper
  inspection. The Windows distribution validator also inspects both symbols.
- `2b5173f` removed the unproven serial caller routing. The private exact
  full-tile symbol remains available to the parallel runtime, where the caller
  has already authenticated complete tile bounds.

Accordingly, the artifact coverage finding is closed and the rejected serial
promotion is not included in Milestone 7 performance claims.
