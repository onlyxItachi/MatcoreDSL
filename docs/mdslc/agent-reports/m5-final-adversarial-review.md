# Milestone 5 final adversarial review

Date: 2026-07-22

Review base: `e4dc0affff6c540a65435ba25c5cefa4d69cb562` (`main`)

Production and test checkpoint reviewed:
`95fdce281fda8f449f65689e2f7e14c14d7d89c6`

The subsequent commits inspected by this review are documentation-only:
the independent Linux acceptance matrix, the EOF normalization, and the
Windows x64 portability audit. The Windows audit explicitly reports an
unsupported and unvalidated platform; it does not change the Linux product
code or broaden support claims.

## Verdict

**Accept the Linux Milestone 5 implementation for the validated host scope.**

No unresolved high- or medium-severity finding remains. I independently read
the complete production, test, package, workflow, and evidence diff; rebuilt
the product from a clean external build directory; reran the complete Release
suite; reran focused ASan/UBSan and TSan matrices; exercised the selected
parallel AVX-512 path on the physical host; repeated the persistent executor;
authenticated both packed-ISA artifacts; tested explicit provider absence;
and inspected an installed prefix.

This approval is deliberately narrower than production readiness. It covers
the Linux, synchronous CPU GEMM architecture and implementations actually
compiled and executed on the declared one-node AMD validation host. It does
not establish Windows support, physical multi-node NUMA behavior, accelerated
BF16/INT8, AMX, any GPU backend, or universal performance.

## Rejection-oriented scope

The review attempted to reject the implementation on each of these surfaces:

- CPUID claims without OS extended-state checks;
- AVX-512 dispatch before exact hardware, OS, compiler, implementation, and
  runtime-validation evidence exists;
- an AVX2 or AVX-512 stable ID masking a scalarized body;
- AMX usability inferred only from flags rather than Linux permission state;
- duplicated or contradictory planner legality;
- malformed capability, topology, placement, and workspace records;
- hidden workspace allocation, packing, tensor copies, or fallback;
- output mutation after a pre-execution rejection;
- workspace arithmetic overflow and tensor/workspace overlap;
- tail loads or stores outside A, B, or C;
- persistent-worker lifetime races, shutdown deadlock, output races, false
  sharing, executor recreation, and nested provider oversubscription;
- inaccurate physical/logical-core interpretation or synthetic NUMA evidence
  presented as physical validation;
- C++ ABI leakage through the public C interface;
- unstable exported symbols or broken existing ABI layouts;
- unfair benchmark intervals, unauthenticated timed output, biased registry
  order, stale source provenance, and raw benchmark artifacts entering Git;
- OpenBLAS dimension/thread mistakes and silent provider fallback;
- build-tree-only success, absolute installed paths, Python/nanobind leakage,
  or untracked generated files;
- documentation claims exceeding direct hardware and test evidence.

## Static review result

### Capability and ISA safety

Capability v2 separates hardware, OS-enabled state, compiler support,
implementation availability, and numerical runtime validation. x86 AVX and
AVX-512 usability checks include OSXSAVE/XCR0 state. AMX remains unavailable
on this host and the Linux permission query fails closed. Synthetic records
remain injectable and validated; unknown or inconsistent domain bits do not
become legal planner evidence.

The AVX-512 F32 kernel is isolated behind a per-function target attribute; the
runtime as a whole is not compiled for AVX-512. The call path independently
checks runtime usability before entering the target function. Full M/N/K tails
use bounded edge storage and no speculative out-of-range load was found.

### Persistent and parallel execution

The executor serializes submissions, retains a submitter-owned callback record
only on active workers, and includes active workers in the lifetime barrier.
The final shutdown state machine makes a stopping worker drain an unseen active
submission before exit, closing the previously documented deadlock. Worker
results are cache-line separated. Parallel GEMM assigns disjoint output row
bands, packs B once into shared read-only workspace, gives each worker a
separate aligned A-pack region, and reports exact shared/per-worker workspace.

The public C interface exposes only opaque ownership and POD descriptors. It
does not expose the internal callback API. Context creation performs exact
numerical validation on the same bound workers later used for execution.
Provider nesting is rejected rather than silently oversubscribing native and
OpenBLAS workers.

### Planner, topology, and NUMA

The eight-entry registry is fixed and deterministic. Every candidate records
legality, reason, runtime validation, required feature domains, workspace,
thread count, cost, and priority. Automatic ties resolve by priority then
registry order. Forced illegal variants return no selected fallback.

Topology discovery and affinity records are deterministic, validated, and
restricted to the inherited process mask. Incomplete placement evidence fails
closed for parallel planning. Single-node discovery and affinity are physically
validated on this host. Multi-node policy is synthetic-only and the plan
diagnostic never claims hidden page placement or interleaving.

### Numeric semantics and ABI

Typed IR verification admits only the implemented F32, BF16-to-F32, and
I8-to-I32 contracts. BF16 conversion specifies round-to-nearest-even and
canonical NaN behavior. I8 accumulation deliberately uses modulo-2^32
semantics before bit reinterpretation as I32. Independent tests cover these
contracts and overflow behavior.

The shared runtime exports exactly 15 `matcore_runtime_*` C symbols. Existing
descriptor sizes and offsets remain pinned. Callers retain tensor and
workspace ownership; pre-execution validation covers pointers, rank, dtype,
layout, stride, mutability, tensor overlap, workspace alignment, workspace
size, and workspace/tensor overlap before execution begins.

### Benchmark and evidence integrity

Schema v4 distinguishes complete-call measurement modes from microkernel-only
diagnostics. The timed region, allocation/packing mode, persistent-context
boundary, cache mode, actual threads, affinity, workspace, checksum, timer
floor, CPU/compiler/provider state, and exact source provenance are emitted.
Final timed output is authenticated, and equal-cardinality untimed oracle
replays bracket the balanced forward/reverse timing passes. Guard mode rejects
unknown or dirty tracked-source provenance.

The committed performance report is a sanitized, host-specific summary. It
does not claim universal crossover rules, real multi-node performance,
accelerated BF16/VNNI/AMX, or Windows support. Raw benchmark JSON remains
outside Git.

## Independent validation

All builds used Clang/Clang++ 21.1.8, Ninja `-j2`, and external build/temp
directories. The Release provider was authenticated OpenBLAS 0.3.32 via
pkg-config.

### Fresh Release, OpenBLAS required

```text
cmake -S compiler -B <release> -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=ON -DMDSLC_REQUIRE_OPENBLAS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build <release> -- -j2
ctest --test-dir <release> --output-on-failure -j1
```

Result: 89/89 build actions and **42/42 tests passed**. The matrix includes the
native frontend and driver pipeline, integration, relocated consumer, C17 ABI,
IR, platform/capability/topology/affinity, numeric types, AVX2 and AVX-512,
persistent and parallel runtime, workspace, OpenBLAS, public context ABI,
planner CLI, exact artifact checks, and benchmark/provenance tests.

### ASan and UBSan

A fresh Debug/OpenBLAS-disabled build used `-O1 -g`,
`-fsanitize=address,undefined`, and `-fno-omit-frame-pointer`. With leak
detection, halt-on-error, strict string checks, and UBSan stack traces, the
focused capability/topology/affinity, typed numeric, AVX2, AVX-512, executor,
parallel, planner, public C context, and benchmark matrix passed **15/15** with
no sanitizer report.

### TSan

A fresh Debug/OpenBLAS-disabled build used `-O1 -g -fsanitize=thread`. The
executor, parallel GEMM, planner-resource, and public-context matrix passed
**4/4** with `halt_on_error=1` and no TSan report.

### Runtime and stress evidence

- A guarded 127x129x131 automatic run passed the independent double-precision
  oracle and selected legal single-thread packed AVX-512 execution.
- A guarded 512x512x512, four-thread run selected
  `cpu.native-parallel.avx512-fma.f32.v1`, used four actual workers and explicit
  shared/per-worker workspace, and passed correctness. Its observed 197.6
  GFLOP/s is evidence for this run only, not a portable claim.
- The persistent execution-context regression passed **100/100** repeated
  Release processes, including its internal alternating active-worker and
  callback-shutdown stress.
- With OpenBLAS disabled, a forced external-provider request exited nonzero,
  reported `selected=none`, and identified the unlinked adapter; it did not
  choose a native fallback.

### Artifact, install, and hygiene evidence

- Exact AVX2 symbol inspection found YMM operations and eight packed-FMA
  instruction sites.
- Exact AVX-512 symbol inspection found 21 ZMM operands and four packed-FMA
  instruction sites.
- Installation into a prefix containing spaces produced all four tools,
  public headers, the versioned runtime, and relocatable CMake package files.
- `libmatcore_runtime.so.0.0.0` has the expected OpenBLAS dependency in the
  provider-enabled build and exactly 15 public Matcore C exports.
- Installed `matcore-plan` uses the prefix-relative `$ORIGIN/../lib` runpath.
- Binary and text scans found no reviewed checkout or build-directory path in
  the install prefix.
- Repository hygiene passed, no generated binary/cache/log bundle was found in
  the tracked diff, and `git diff --check` is clean after the dedicated EOF
  normalization commit.

The separately committed Linux acceptance lane also records fresh Debug
42/42, ASan/UBSan 31/31, TSan 3/3, OpenBLAS-disabled 37/37, installed consumer,
C17 ABI, native `.mdsl -> .o -> executable`, and legacy frontend 14/14 evidence.
That report was inspected for consistency but was not used as a substitute for
the independent executions above.

## Findings and disposition

No new high- or medium-severity defect was reproduced.

The integration already contains focused fixes and regressions for the serious
defects found during earlier review rounds: creator-thread rather than
bound-worker runtime authentication, loss of exact validation evidence on
rejected plans, inactive workers retaining borrowed submissions, callback-side
shutdown deadlock, benchmark timed-output masking, registry-order drift,
caller/worker CPU overlap, and stale build provenance. The current code and
independent stress/sanitizer runs confirm those fixes remain present.

One low-severity formatting finding was raised during this review: an extra
blank line at EOF in `m5-topology-planner-hardening.md`. It was corrected by
the documentation-only EOF normalization commit, and the final diff check is
clean.

## Honest remaining limitations

- Windows is a portability audit only. No clang-cl/LibTooling build, COFF/PE
  artifact, runtime DLL/import library, package consumer, hosted workflow, or
  distribution ZIP has been produced.
- Physical topology is one socket and one NUMA node. Multi-node NUMA behavior
  has synthetic legality coverage only.
- AVX-512 F32 is physically runtime-validated here. AVX-512 BF16, VNNI, and
  AMX optimized variants are not implemented or claimed; BF16 and I8 have
  reference semantics only.
- Planner calibration and performance numbers are specific to this AMD Ryzen
  AI 9 HX 370 host, toolchain, OpenBLAS provider, governor, and measured
  contract. They are not global optimality claims.
- The runtime implements GEMM only for this milestone. It does not establish
  GEMV, GEVM, fused operations, MLIR lowering, GPU execution, or heterogeneous
  placement.

## Final recommendation

Merge the Milestone 5 Linux implementation only after the integration owner
confirms its final exact-tip hosted checks and publication gates. Preserve the
Windows work as the explicitly deferred compatibility phase. Within the
validated Linux CPU scope, the advanced backend acceptance gate has no
unresolved high or medium review finding.
