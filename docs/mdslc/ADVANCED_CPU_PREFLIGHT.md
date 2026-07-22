# Advanced CPU Backend Preflight

Captured: 2026-07-22 (Europe/Istanbul)

Milestone: MDSLC Milestone 5 — Advanced CPU Backend

Repository base: merge commit
`e4dc0affff6c540a65435ba25c5cefa4d69cb562`
(`mdslc-cpu-performance-foundation-v1`)

## Operating system and toolchain

- Ubuntu 26.04 LTS (Resolute), Linux 7.0.0-27-generic x86-64.
- Clang and Clang++ 21.1.8, target `x86_64-pc-linux-gnu`, POSIX thread model.
- LLVM 21.1.8.
- CMake 4.3.2 and Ninja 1.13.2.
- 24 logical CPUs reported by `nproc`.
- 14 GiB physical RAM and 32 GiB swap; 6.9 GiB memory was available at
  capture time. Builds remain limited to Ninja `-j2`.
- The repository filesystem had 354 GiB available.

## Physical CPU and topology

- AMD Ryzen AI 9 HX 370, family 26/model 36/stepping 0.
- One socket, twelve physical cores, two threads per core, 24 online logical
  CPUs; SMT is enabled.
- One physical NUMA node. Node 0 contains CPUs 0-23. Multi-node NUMA behavior
  can therefore be synthetic-tested but not physically performance-validated
  on this host.
- L1 data: 576 KiB total over 12 instances; L1 instruction: 384 KiB total over
  12 instances; L2: 12 MiB over 12 instances; L3: 24 MiB over two groups.
- CPU 0 shares its 16 MiB L3 with CPUs 0-3 and 12-15. The second LLC group is
  represented separately by sysfs and must not be collapsed into a fictional
  uniform LLC.
- CPUs 0-3 have a reported maximum of 5.158 GHz; CPU 4 and the compact-core
  group report 3.289 GHz. Planner evidence must retain this heterogeneous-host
  caveat.
- `amd-pstate-epp` is active, governor `performance`, boost enabled. Frequency
  metadata is observational and is not treated as a fixed clock guarantee.

## ISA inventory and validation boundary

Linux reports AVX, AVX2, FMA, AVX-512F/DQ/BW/VL, AVX-512 VNNI, and AVX-512
BF16, among other extensions. It does not report AMX-TILE, AMX-BF16, or
AMX-INT8.

These flags are inventory only. Milestone 5 dispatch also checks OS extended
state, compiler implementation availability, exact function machine code, and
runtime correctness. The report does not by itself authorize AVX-512
execution. AMX runtime is unavailable on this physical host. Linux headers do
provide `ARCH_GET_XCOMP_PERM` and `ARCH_REQ_XCOMP_PERM`; no AMX permission is
requested unless both a hardware implementation and matching runtime test
exist.

## External provider

- OpenBLAS 0.3.32, Debian/Ubuntu package `0.3.32+ds-5`.
- LP64 pthread development surface from
  `libopenblas-dev`/`libopenblas-pthread-dev`.
- `pkg-config` resolves the coherent pthread header/library pair and `-lopenblas`.
- OpenBLAS remains optional and its pool is never nested inside native workers.

## Validation scope

Physically eligible for validation on this host:

- AVX2/FMA single-thread and parallel F32;
- AVX-512 F32 when XSTATE and exact artifact checks pass;
- AVX-512 BF16 and VNNI only after typed reference semantics and a complete
  optimized implementation exist.

Not physically validatable here:

- AMX-BF16 and AMX-INT8;
- real multi-node NUMA placement;
- Windows/MSVC ABI, COFF/PE, DLL, and Windows NUMA behavior.

Windows receives a focused hosted compatibility phase only after the Linux
Milestone 5 implementation and acceptance gates pass.
