# Milestone 6 CPU performance preflight

Captured: 2026-07-26 (Europe/Istanbul)  
Repository base:
`951239f1bee5541a4cf5ad72fab2192de07cf89d` (`main`)  
Mutation: none; the audit and baseline build used external directories

## Repository and baseline

- The canonical checkout was clean on `main`.
- Local `main` and `origin/main` resolved to the same commit.
- No merge, rebase, cherry-pick, revert, or bisect operation was active.
- No open pull request raced with the audit.
- Repository hygiene, `git diff --check`, and `git fsck --full --strict`
  completed without an integrity failure.
- The Windows x64 workflow remains present at
  `.github/workflows/mdslc-windows.yml`.

A fresh external Release configuration used Clang/Clang++ 21.1.8, Ninja, the
native frontend, and required OpenBLAS. All 95 Ninja actions completed with
`-j2`; the declared suite passed **44/44** in 82.27 seconds. The suite included
frontend, IR, planner, runtime, OpenBLAS, AVX2/AVX-512 artifact, strict C17 ABI,
install, and external-consumer gates.

## Physical host

- OS: Ubuntu 26.04 LTS.
- Kernel: Linux 7.0.0-27-generic.
- CPU: AMD Ryzen AI 9 HX 370.
- Topology: one socket, 12 physical cores, 24 logical CPUs, SMT enabled.
- NUMA: one physical node.
- Cache totals: 576 KiB L1d, 384 KiB L1i, 12 MiB L2, 24 MiB L3.
- Maximum-frequency domains:
  - CPUs 0-3 and 12-15: approximately 5.158 GHz;
  - remaining CPUs: approximately 3.289 GHz.
- Frequency policy: `amd-pstate-epp`, governor `performance`, energy
  preference `performance`, boost enabled.
- Memory snapshot: 14 GiB RAM with 6.1 GiB available; 31 GiB swap with 28 GiB
  available.
- Repository filesystem free space: 299 GiB.

The host is not uniform. Reports must retain logical CPU, placement, LLC group,
and frequency-domain context rather than presenting one core as representative
of every core.

## Toolchain and analysis tools

- Clang/Clang++/LLVM: Ubuntu 21.1.8.
- Matching development packages: `1:21.1.8-6ubuntu1`.
- CMake: 4.3.2.
- Ninja: 1.13.2.
- ccache: 4.12.3.
- GNU binutils: 2.46.
- `llvm-objdump-21`, `llvm-mca-21`, `perf` 7.0.12, `numactl` 2.0.19,
  `cpupower`, and `turbostat` are installed.
- LLVM's host scheduler model is `znver5`.

No package is missing for the functional, disassembly, static-scheduling,
build, package, or benchmark work. No package was installed during preflight.

## External BLAS

- Provider: OpenBLAS `0.3.32+ds-5`.
- Threading layer: pthread.
- Integer ABI: LP64.
- Coherent implementation:
  `/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.32.so`.
- Provider configuration:
  `DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=128`.
- `cblas_sgemm`, provider metadata, and local thread-control symbols are
  present.
- The ambient provider default was 24 threads. Fair runs must keep using an
  explicit provider thread count and record the actual value.

## Counter-collection limitation

`perf_event_paranoid` is `4`. Unprivileged `perf stat` failed for both software
and hardware events. `turbostat` also lacked the unprivileged MSR/RAPL access
needed for a reliable run.

This does not block correctness, wall-clock benchmarking, disassembly,
compiler remarks, sysfs frequency observation, or static `llvm-mca` analysis.
It does block direct hardware-event claims such as measured cache misses,
TLB misses, IPC, and stalled-cycle fractions. Those fields must be reported as
unavailable rather than estimated from static models. The milestone does not
change system policy to bypass this restriction.

