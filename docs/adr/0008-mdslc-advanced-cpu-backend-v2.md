# ADR-0008: MDSLC Advanced CPU Backend v2

Status: Accepted and published for the declared Linux host scope; focused
Windows x64 compiler/runtime/package validation passed

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
| Windows | Hosted Windows Server 2025 with clang-cl/LLVM 21.1.8 validates the native frontend, COFF/PE artifacts, runtime DLL/import library, planner, runtime AVX2 variants, package/consumer with space and Unicode paths, focused runtime/generated-code ASan, and a CI ZIP. AVX-512 is compile/disassembly-only on this host, OpenBLAS is omitted, and multi-node NUMA is synthetic-only. |

This table records the accepted Linux implementation and the separately
bounded hosted Windows evidence. Linux passed fresh Release/Debug, sanitizers,
whole-diff review, hosted checks, normal merge, and the immutable
`mdslc-cpu-backend-v2` tag. The focused Windows candidate passed Release (35
passed plus one explicit AVX-512 hardware skip), Debug (26 passed plus the same
skip), package relocation, external consumer, exact artifact inspection, and
independent review. The public opaque handle is still destroyed exactly once;
only the internal C++ `shutdown()` operation is repeat-safe.

Windows AddressSanitizer evidence is intentionally narrower than Linux. The
runtime DLL and generated host/stub/backend translation units are instrumented
and executed. The native LibTooling executables use the already validated
Release tools because the authenticated LLVM archive's LLVMSupport allocator
conflicts with the Windows static ASan allocator thunk. Windows UBSan is not
claimed. The distribution records the Microsoft Visual C++ 2015--2022 x64
Redistributable as an external prerequisite; a clean-machine installation was
not performed. The Windows ZIP contains 17 installed files, passed recursive
import and absolute-path-leak checks, and has SHA-256
`b2c633192d3084585198f24eedba3957a85552c5d483d3b656bfdeda60480cd2`.

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
