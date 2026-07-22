# MDSLC CPU performance preflight

> Historical Milestone 4 pre-implementation snapshot. The dependency and host
> inventory remains useful evidence, but the implementation blockers recorded
> here were subsequently addressed by Milestones 4 and 5. Current acceptance
> status belongs in `STATUS.md` and the final milestone reports.

Audit date: 2026-07-22
Repository checkpoint: `main` at
`6a5b931baa6ec1136fb3c0471f515bd666a23981`
Audit behavior: read-only repository and system inspection; no packages changed

## Validation host

- OS: Ubuntu 26.04 LTS (Resolute)
- Kernel: `7.0.0-27-generic`, x86-64
- CPU: AMD Ryzen AI 9 HX 370 with Radeon 890M
- Topology: one socket, 12 physical cores, 24 logical CPUs, SMT enabled
- NUMA: one node, CPUs 0-23, local distance 10
- Process affinity during audit: CPUs 0-23, NUMA node 0
- Cache line: 64 bytes
- Per-core caches: 48 KiB L1 data, 32 KiB L1 instruction, 1 MiB L2
- Last-level cache groups:
  - cores 0-3 and SMT siblings 12-15 share 16 MiB L3;
  - cores 4-11 and SMT siblings 16-23 share 8 MiB L3.
- Maximum policy frequencies:
  - cores 0-3: approximately 5.157 GHz;
  - cores 4-11: approximately 3.289 GHz.
- Frequency driver: `amd-pstate-epp`
- Governor and energy preference: `performance` on every policy
- Boost: supported and enabled
- Memory at audit: 14 GiB total, 6.6 GiB available
- Swap at audit: 32 GiB total, 5.8 GiB used
- `/tmp` free space at audit: 3.6 GiB

The host is heterogeneous in frequency and LLC membership. A defensible
single-thread calibration run must pin a declared logical CPU. CPU 0 is the
initial high-frequency-core choice; results from the lower-frequency LLC group
must be labeled separately. Affinity, governor, boost, and cache group remain
environment facts, not universal planner assumptions.

## CPU instruction inventory

The operating system reports AVX2 and FMA. It also reports AVX-512F,
AVX-512DQ, AVX-512BW, AVX-512VL, AVX-512VNNI, and AVX-512BF16. No AMX feature
was reported.

This is an inventory only. Milestone 4 may use the already validated AVX2/FMA
path. AVX-512 and BF16 remain unsupported until Milestone 5 verifies hardware,
OS architectural state, compiler support, implementation availability, exact
instructions, and runtime correctness. AMX must fail closed on this host.

## Compiler and measurement tools

- `clang-21` / `clang++-21`: Ubuntu Clang 21.1.8 (`6ubuntu1`)
- `llvm-config-21`: 21.1.8
- LLVM include directory: `/usr/lib/llvm-21/include`
- LLVM library directory: `/usr/lib/llvm-21/lib`
- LLVM CMake directory: `/usr/lib/llvm-21/lib/cmake/llvm`
- CMake: 4.3.2
- Ninja: 1.13.2
- ccache: 4.12.3
- perf: 7.0.12
- taskset: util-linux 2.41.3
- numactl: present; reports the single node above
- cpupower: present
- turbostat: 2026.02.14

Release benchmark builds use Clang 21 with `-O3 -DNDEBUG`. The MDSLC native
frontend continues to use the coherent LLVM/Clang 21.1.8 development tuple.
Milestones 4 and 5 do not need MLIR and must not change the system MLIR setup.

## Coherent CBLAS provider

A complete optional OpenBLAS development provider is installed:

- packages: `libopenblas-dev:amd64` and
  `libopenblas-pthread-dev:amd64`;
- package version: `0.3.32+ds-5`;
- pkg-config module/version: `openblas`, 0.3.32;
- header root: `/usr/include/x86_64-linux-gnu/openblas-pthread`;
- shared implementation:
  `/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.32.so`;
- SONAME: `libopenblas.so.0`;
- threading layer: pthread (`OPENBLAS_THREAD=1`, `USE_OPENMP=0`);
- integer ABI: LP64 (`USE_64BITINT=0`);
- configured maximum: 128 threads;
- runtime default on this host: 24 threads;
- dynamic runtime record:
  `OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=128`.

The library exports `cblas_sgemm`, provider configuration queries, and
`openblas_set_num_threads_local`. A direct Clang 21 compile/link/run probe used
row-major `cblas_sgemm`, constrained the provider to one thread, and produced
the expected 2x2 result `19,22,43,50`. `ldd` resolved the probed executable to
the pthread provider above.

The distribution's `OpenBLASConfig.cmake` computes include/library locations
that do not exist in this installation. MDSLC should authenticate OpenBLAS via
the working pkg-config target and a configure-time header/link/API probe rather
than relying on that config file. Required legality includes 32-bit `blasint`
dimension and leading-dimension bounds. Single-thread comparisons must call
the provider's local thread-control API and record the actual count; ambient
default use of 24 threads is not a valid one-thread comparison.

No package installation is required for the Milestone 4 OpenBLAS adapter.

## Existing benchmark baseline

The pre-milestone opt-in `matcore_cpu_planner_benchmark`:

- accepts only `[--guard] [M K N]`;
- uses two warmups and nine measured executions;
- caps A+B+two-float-output storage at 256 MiB;
- times compute after allocation and planning;
- emits text only;
- does not report GFLOP/s, packing, workspace, provider, cache, affinity, or
  thread behavior;
- compares with the same float reference implementation rather than an
  independent double oracle;
- links the header-only planner rather than exercising the installed runtime
  ABI.

An isolated Release audit build passed the three focused runtime tests,
including the exact YMM/FMA object check. Diagnostic medians were collected at
16, the 33x35x37 tail, 128, and 256 sizes. They are intentionally not retained
as performance claims: the smallest intervals were close to call/timer
overhead, the process was not pinned, and the old contract does not represent
allocation or packing consistently.

## Milestone 4 measurement gates

`matcore-bench` v1 addresses the baseline gaps through:

- explicit one-shot, reused-workspace, prepacked-B, and compute-only modes;
- stable JSON schema and environment capture;
- a configurable pre-allocation memory cap with checked arithmetic;
- full or sampled independent double-precision correctness checks;
- minimum/median/nearest-rank-p95 statistics;
- hot-cache aggregation to a declared timer floor;
- explicit rejection of sub-floor cold-cache samples;
- exact requested and actual thread counts;
- workspace, packing, alignment, provider, capability, compiler, flags, and
  source-commit metadata;
- quick, standard, and opt-in full profiles in M,N,K order.

Raw runs belong beneath ignored `benchmark_reports/`. Only reviewed summaries
may be committed to `docs/performance/cpu/`. Performance claims are limited to
the declared host, build, affinity, provider, timing mode, and shape matrix.

## Blockers at the time of this audit

There is no dependency blocker for the Linux Milestone 4 benchmark or optional
OpenBLAS adapter. Current memory pressure makes low-parallelism builds (`-j2`)
and explicit benchmark memory caps mandatory. At the time of this preflight,
AVX-512, BF16, persistent parallelism, topology-aware planning, and Windows
runtime validation were separate Milestone 5 or deferred Windows gates. The
Linux Milestone 5 implementation later addressed the first four; Windows
runtime validation remains a separate focused phase.
