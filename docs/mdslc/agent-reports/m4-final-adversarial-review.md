# Milestone 4 final adversarial review

Date: 2026-07-22

Review base: `6a5b931baa6ec1136fb3c0471f515bd666a23981` (`main`)

Reviewed integration tip: `2cb97cc07b46a6db89c8981b9f94683d1ffa239f`

Review branch: `mdslc/m4-final-review`

## Verdict

**Accepted for the validated Linux host scope.** The final diff has no
unresolved high- or medium-severity finding. The standalone native frontend,
typed IR, five-candidate CPU planner, additive workspace/prepacked-B ABI,
OpenBLAS adapter, packed AVX2/FMA implementation, benchmark contract,
installation, and existing `.mdsl` artifact path all passed independent
review and execution.

This verdict is deliberately narrower than production portability or a claim
of universal performance. Windows remains modeled but unvalidated; AVX-512,
parallel execution, topology/NUMA planning, BF16, INT8, and AMX belong to
Milestone 5.

## Ownership and method

This lane was read-only for production sources. It reviewed the complete diff,
reproduced failures before remediation, inspected remediation commits, and
modified only this report after the candidate stabilized. The review attempted
to reject the milestone for unfair timing, hidden allocation, incorrect CBLAS
mapping, uncontrolled provider threads, illegal or falsely labeled ISA code,
out-of-bounds vector access, workspace overflow/aliasing, nondeterministic
planning, ABI leakage, package path leakage, and generated-artifact tracking.

## Findings and resolutions

### M4-R1 — excessive OpenBLAS thread plans (medium, resolved)

Before `49cb162`, a forced OpenBLAS plan with `--threads 1000` was reported
legal and selected even though execution rejected the provider policy. This
made planning and execution disagree. The fix adds the discovered provider
ceiling to implementation resources, rejects requests above it before output
mutation, reports `actual_threads=0` for illegal candidates, and checks the
same condition in the direct adapter and C ABI query.

Independent reproduction at the final tip:

- `matcore-plan ... --threads 1000 --variant cpu.external.openblas.f32.v1`
  exits 1;
- `matcore-bench` with the same request exits 1;
- both name `OpenBLAS requested thread count exceeds provider maximum`;
- the validation host reports a ceiling of 24;
- runtime tests prove rejected queries preserve output and caller-owned report
  structures.

### M4-R2 — misleading packing-excluded mode (medium, resolved)

The original `--exclude-packing` option either rejected the packed candidate
because A packing remained necessary or misleadingly labeled complete
reference/OpenBLAS calls as packing-excluded. `74277a0` replaces that behavior
with one explicit packed-compute-only diagnostic. Complete A and B micro-panels
are prepared before timing; the interval invokes the exact production 4x16
AVX2/FMA microkernel, tail handling, and output stores without allocation or
packing. JSON and human output say `complete_implementation_comparison=false`
and `comparison=diagnostic-only`. Other variants reject the mode, including
OpenBLAS whose provider-internal packing is opaque.

The final 33x35x37 adversarial run was correct, timing-valid, used 12,480 bytes
of declared workspace, and carried the explicit
`packed-compute-only: A and B packing prepared before timing` scope. A forced
reference request with `--exclude-packing` exited 1 actionably.

### M4-R3 — double-live one-shot buffers and understated cap (medium, resolved)

The original `--include-allocation` path retained reusable output/workspace
while allocating another one-shot pair, and its estimator omitted some
`AlignedBuffer` alignment overhead. The old 512-cubed case could therefore
stay under a nominal 4 MiB cap while more memory was simultaneously live than
reported.

`74277a0` removes the unused reusable output/workspace in one-shot mode,
destroys the preceding pair before the next allocation, and accounts for each
live aligned buffer as payload plus twice its requested alignment plus
`alignof(max_align_t)`. It also includes prepacked storage and the cold-cache
payload. Unit tests accept the exact computed boundary and reject one byte
below it.

Independent final-tip evidence:

- 512x512x512 packed one-shot with a 3 MiB cap exits 1 before allocation;
- the 4 MiB run succeeds and passes correctness;
- its reported estimate is 3,539,520 bytes, including 393,216 bytes of
  workspace, below the 4,194,304-byte cap;
- compute-only and prepacked-B modes cannot be conflated with one-shot
  allocation.

### M4-R4 — instrumented timing-smoke flakiness (medium, resolved)

The first ASan/UBSan review run exposed a new flaky acceptance test: a slow
first probe could exceed a 100 microsecond floor while later correct samples
fell below it. The core and CLI tests consequently failed despite correct
outputs. The failure reproduced 6 times in 25 core-test runs. A first attempt
using a 1 microsecond CLI floor still failed in both Release and the
OpenBLAS-disabled build for a 2x3x2 reference call.

Commits `344c42c`, `4abacb0`, and `2cb97cc` separate schema/correctness smoke
coverage from performance acceptance, retain dedicated timer-floor rejection
coverage, and use representative 33x35x37 reference and 127x129x131 packed
tail shapes for CLI timing-valid smoke checks. At the final tip the independent
ASan/UBSan core loop passed 25/25, the CLI loop passed 10/10, and the focused
instrumented suite passed 8/8. The integration owner's additional repetition
evidence was core 50/50 and CLI 25/25.

## Implementation review

### OpenBLAS adapter

The adapter is compiled only after pkg-config discovery, `cblas.h` compilation,
LP64 `blasint` verification, and link authentication. The reviewed row-major
call uses `CblasNoTrans` for both inputs, `M/N/K` in semantic order, and leading
dimensions `K`, `N`, and `N`. It uses documented local OpenBLAS thread
control, verifies the provider-reported setting, restores the prior setting,
and performs all known legality checks before calling the void CBLAS entrypoint.
The optional-off build does not advertise or link the variant and forced use
fails without fallback.

### Workspace and C ABI

The original `matcore_runtime_gemm_f32_v0` symbol and layouts remain intact.
Six additive functions expose resource query, workspace execution, prepacked-B
query/preparation/execution, and the existing plan report. No C++ type crosses
the public header. Required size/alignment is queryable, storage is owned by the
caller, and packed execution performs no hidden allocation. Range arithmetic,
workspace/tensor overlap, packed descriptor provenance, source identity,
mutability, rank, shape, layout, dtype, memory-space, and output aliasing are
validated before output mutation.

The Release DSO defines exactly seven public dynamic symbols: the original
GEMM entrypoint plus the six additive planning/workspace/prepack functions.
Private backend C++ symbols do not leak through the DSO boundary.

### Packed AVX2/FMA engine

The implementation uses per-function `target("avx2,fma")`; the entire runtime
is not compiled for AVX2. Reviewed parameters are MR=4, NR=16, MC=128, NC=256,
KC=256 with 64-byte caller workspace. Packing and requirement arithmetic use
checked multiply/add/round-up operations. Boundary tiles use a local guarded
4x16 tile, avoiding speculative out-of-range matrix loads and stores. Seeded
random, rectangular, tail, low-alignment, 64-byte-alignment, alias, insufficient
workspace, malformed prepack, and repeated-use tests passed under ASan/UBSan.

Exact-symbol disassembly of
`matcore_cpu_packed_avx2_4x16_microkernel_f32_v1` contains YMM operations and
eight packed single-precision FMA instruction sites. The stable variant ID is
therefore backed by the claimed ISA body on this build.

### Deterministic planner and performance claims

The five fixed stable IDs are reference, tiled, compiler-vectorized AVX2/FMA,
OpenBLAS, and native-packed AVX2/FMA. Every candidate records legality, reason,
cost, priority, workspace, alignment, and actual thread policy. Forced illegal
variants fail rather than falling back. Automatic selection remains a static,
deterministic calibrated rule; no runtime autotuning was introduced.

The external calibration archive at
`/home/hamza-usta/archives/MDSLC-m4-calibration-20260722T162854` was checksum
verified. Excluding its checksum manifest it contains 368 files and 879,322
bytes; including the manifest it contains 369 files and 947,598 bytes. The
sanitized report records combined median regret 1.000, p95 1.124, and maximum
1.132 on the declared pinned validation-host matrix. Independent spot runs
reproduced the stated qualitative behavior: packed AVX2 materially beats the
scalar and compiler-vectorized implementations on supported medium/large
shapes; OpenBLAS is generally fastest; the measured M=1 provider regression is
kept out of automatic selection. These are host/provider-specific observations,
not a BLAS-parity or universal-performance claim.

## Independent validation at `2cb97cc`

Toolchain and host:

- Ubuntu Clang/LLVM 21.1.8, CMake 4.3.2, Ninja 1.13.2;
- AMD Ryzen AI 9 HX 370, 12 physical cores / 24 logical CPUs, one socket;
- one physical NUMA node;
- OpenBLAS 0.3.32 pthread LP64 provider via pkg-config.

Validation results:

- final Release build: success;
- complete Release CTest suite: 27/27 passed in 64.64 seconds;
- complete Debug CTest suite: 27/27 passed in 66.09 seconds;
- Release and Debug each include the native frontend, 63-case integration
  matrix, install/consumer, IR, planner, runtime, ABI, object, and benchmark
  tests;
- ASan+UBSan focused suite: 8/8 passed with leak detection and halt-on-error;
- ASan+UBSan benchmark CLI repetition: 10/10; core repetition: 25/25;
- OpenBLAS-disabled focused suite: 5/5, CLI repetition 5/5, and no OpenBLAS
  dynamic dependency;
- exact AVX2 artifact check: YMM present, eight packed FMA sites;
- repository hygiene and `git diff --check`: passed;
- installed-file absolute source/build path scan: passed;
- fresh install contains all four tools, both public headers, versioned runtime,
  and relocatable CMake package; the external installed consumer passed inside
  both complete suites;
- runtime dependency inspection resolves the coherent pthread OpenBLAS DSO and
  normal C/C++ libraries, with no Python or nanobind dependency.

An independent final artifact proof produced an ordinary ELF64 x86-64
relocatable `gemm_v0.o`, linked it with ordinary Clang 21 into an ELF PIE, and
executed:

```text
host-before
MDSLC CPU GEMM PASS
```

`nm -C` showed the generated call-site/backend symbols and an undefined stable
`matcore_runtime_gemm_f32_v0` C boundary. `--save-temps` produced the expected
host, IR, sites, stubs, backend, and object artifacts outside Git.

## Remaining limitations and claim boundaries

- Windows x64 is represented by the platform record but is not compiled,
  packaged, or runtime-validated in this milestone.
- TSan was not treated as applicable: Milestone 4 introduces no MDSLC worker
  pool or shared parallel execution state. Persistent parallel execution and
  its race gate belong to Milestone 5.
- The host has one NUMA node; no real multi-node behavior is claimed.
- OpenBLAS thread reporting is the provider-reported configured local thread
  count, not an operating-system measurement of workers active for each tiny
  problem.
- Packing-excluded output is a diagnostic microkernel interval and must never
  enter end-to-end planner-regret or OpenBLAS comparisons.
- Calibration thresholds are static and validation-host-specific. Runtime
  autotuning and universal crossover claims remain out of scope.
- No CUDA, GPU, AVX-512, BF16, INT8, AMX, GEMV, GEVM, fusion, or heterogeneous
  placement support is implied by this review.

## Final decision

Milestone 4 CPU Performance Foundation: **passed for the validated Linux host
scope**, with no unresolved high or medium finding.
