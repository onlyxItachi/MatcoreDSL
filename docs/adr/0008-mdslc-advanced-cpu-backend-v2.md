# ADR-0008: MDSLC Advanced CPU Backend v2

Status: Accepted and locally validated for the declared Linux host scope;
hosted publication and Windows validation pending

Date: 2026-07-22

## Context

Milestone 4 established a deterministic single-thread CPU planner, an optional
OpenBLAS candidate, caller-owned packed-GEMM workspace, and an AVX2/FMA native
engine. Milestone 5 extends that proven path without changing the valid-C++
source contract, trusted LibTooling frontend, typed Matcore IR v1, or existing
C ABI symbols.

The advanced backend must distinguish what the processor advertises from what
the operating system enables, what this binary implements, and what has
actually executed successfully. It must also avoid per-call thread creation,
make workspace and thread policy explicit, and report topology/NUMA decisions
without moving memory implicitly.

## Decisions

### Capability record v2

The versioned CPU capability record separately represents:

- hardware feature discovery;
- operating-system architectural-state enablement;
- compiler support used to build a function body;
- implementation availability in the current binary; and
- runtime-validation status on the current physical host.

Feature names have a canonical order. Unknown future bits fail closed for a
known record version. Tests may inject synthetic records, but synthetic support
is never reported as physical runtime validation. X86 AVX and AVX-512 legality
requires both CPUID feature bits and the relevant XCR0 state. AMX additionally
requires the documented Linux per-thread extended-state permission before any
tile instruction may execute.

### Variant registry and planner v3

Planner v3 is additive to planner v2 and retains stable v1 variant IDs. Its
core F32 registry adds:

- `cpu.native-packed.avx512-fma.f32.v1`;
- `cpu.native-parallel.avx2-fma.f32.v1`; and
- `cpu.native-parallel.avx512-fma.f32.v1`.

Each candidate records semantic legality, capability requirements,
implementation/runtime-validation state, thread count, affinity/topology
policy, workspace, estimated cost, and a deterministic rejection or selection
reason. Automatic planning uses static calibrated rules only. Runtime
autotuning is out of scope.

AVX-512 is not intrinsically preferred over AVX2. It is eligible only when the
exact function is compiled, hardware and OS state are usable, execution is
permitted, workspace is sufficient, and the calibrated rule selects it.

### Persistent execution context

An additive C ABI owns an opaque execution-context handle. The handle contains
no public C++ ABI and exposes explicit creation, execution, diagnostics, and
destruction operations. Creation fixes a maximum worker count and affinity
policy. Workers persist across GEMM calls; execution never creates a thread per
operation. The internal C++ context's `shutdown()` operation is repeat-safe and
joins workers. The opaque public C handle is consumed exactly once by its
destroy function and is invalid afterward; a dangling handle cannot be passed
again safely.

Each invocation reports requested and actual threads. Output macro-tiles are
assigned deterministically. Per-worker workspace is caller-visible through a
query and never allocated inside the existing one-shot ABI. Native and
OpenBLAS pools are mutually exclusive for one execution, preventing nested
oversubscription.

### Topology and NUMA

Topology v1 models logical CPUs, physical cores, packages, NUMA nodes,
CPU-to-node mapping, reliable cache-sharing groups, and discovery completeness.
Linux discovery reads sysfs outside the execution hot path. Portable and
Windows backends remain isolated behind the platform layer.

Initial affinity choices are `none`, `compact`, and `scatter`; NUMA placement
is `single-node` or conservative `local-first`. No policy migrates, interleaves,
or allocates user pages implicitly. The validation host has one NUMA node, so
multi-node behavior is synthetic-only unless separately exercised on physical
multi-node hardware.

### BF16 and INT8 semantics

Typed reference paths precede accelerated variants:

- BF16 inputs convert exactly to F32, multiply and accumulate in F32, and
  produce F32 output.
- I8 inputs multiply exactly and accumulate modulo 2^32 into I32 output. The
  explicit two's-complement bit result avoids signed-overflow undefined
  behavior and matches non-saturating integer dot-product semantics.

Public descriptors use fixed-width C representations. C++ templates,
containers, exceptions, and implementation-specific ABI objects never cross
the runtime boundary. AVX-512 BF16, VNNI, or AMX variants are advertised only
after exact instruction inspection and the required physical-runtime gate.

### Benchmark and portability policy

Milestone 4's benchmark contract remains authoritative. Milestone 5 adds
single/parallel thread sweeps, speedup, efficiency, planner regret, topology,
affinity, ISA, and per-thread workspace fields. Raw runs remain ignored; only
reviewed summaries are committed.

Linux remains the local implementation and performance-validation host.
Windows x64 validation follows Linux Milestones 4 and 5 as a focused
compatibility phase using clang-cl/MSVC ABI, COFF/PE, a runtime DLL/import
library, installed consumer, paths containing spaces, and a CI-produced ZIP.
Windows limitations are reported separately and do not convert Linux evidence
into Windows claims.

## Implemented candidate and evidence boundary

The Milestone 5 integration branch implements the decisions above without
changing the valid-C++ source contract or removing an existing C ABI symbol.
Its F32 planner-v3 registry contains all five Milestone 4 candidates plus
packed AVX-512, parallel AVX2, and parallel AVX-512. The additive public
execution-context and typed-reference interfaces bring the installed runtime
surface to exactly 15 exported C functions.

| Area | Implemented and focused-validation state |
| --- | --- |
| Capability v2 | Hardware, OS state, compiler, implementation, and physical runtime-validation domains are distinct and fail closed. |
| AVX2/FMA F32 | Packed and persistent-parallel variants executed on the declared Linux host; exact YMM packed-FMA artifact checks pass. |
| AVX-512F/FMA F32 | Packed and persistent-parallel variants executed on the declared Linux host; exact ZMM packed-FMA artifact checks pass. |
| Execution context | Persistent workers, requested/actual thread reporting, deterministic row bands, explicit per-worker workspace, affinity policy, and mutually exclusive native/OpenBLAS pools are implemented. |
| Topology/NUMA | One physical Linux NUMA node is discovered and exercised; multi-node planning is synthetic-only and performs no page placement or migration. |
| BF16 and I8 | BF16-to-F32 and I8-to-I32 typed reference semantics, IR contracts, oracles, and C entry points are implemented. |
| Accelerated low precision | AVX-512 BF16, AVX-512 VNNI, AMX-BF16, and AMX-INT8 are not implemented or runtime-validated. The host exposes no AMX. |
| Windows | Frontend, COFF/PE artifacts, runtime DLL/import library, planner, package/consumer, and ZIP are unvalidated and unproduced. |

This table records an implementation-complete Linux candidate and focused
evidence, not final milestone acceptance. Fresh exact-tip Release/Debug and
sanitizer gates, the whole-diff independent review, hosted checks, normal merge,
issue closure, and the immutable tag remain required. The public opaque handle
is still destroyed exactly once; only the internal C++ `shutdown()` operation
is repeat-safe.

## Rejected alternatives

- Global `-mavx512*` runtime compilation: it would make generic binaries unsafe.
- CPUID-only dispatch: it ignores OS extended-state legality.
- A thread pool created per GEMM: it defeats the persistent-runtime objective.
- Hidden runtime workspace or page migration: it violates the explicit resource
  contract.
- OpenBLAS inside native workers: it creates nested oversubscription.
- Synthetic or compile-only support labeled runtime-validated: it is not
  physical evidence.
- Runtime autotuning: deterministic static planning remains the milestone
  contract.

## Consequences

The generic runtime remains usable without AVX-512. New functionality is
additive and inspectable, but planner/runtime reports and the package gain
versioned advanced-CPU records. Hardware-gated paths can remain compile-only or
unavailable without weakening the correctness of supported variants, provided
their status is explicit.
