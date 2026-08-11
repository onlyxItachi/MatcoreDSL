# CPU beta final-candidate local validation

Date: 2026-08-11

Candidate: `6796fd85963f985fb652eb8242d37538b29f0765`

Branch: `mdslc/semantic-compiler-foundation-v1`

Base: `e5069758ad04bdb459de2026cad8498b47fda707`

## Verdict

**Passed for the complete declared local Linux Milestone H scope.** Validation
used a fresh `--no-local` clone, detached at the exact candidate, so later
documentation work could not change any source-copying package test. The clone
was clean before and after the matrix and had no non-ignored untracked files.

This run supersedes the earlier `69d099e` local matrix as final-candidate
evidence. The later candidate includes composition-root verifier hardening,
exact source-inaccessible producer binding, and fail-closed external-provider
conformance routing. No prior build tree, linked artifact, install tree, or test
result was reused. Compiler caching remained enabled and was keyed by
compilation inputs; no zero-hit claim is made.

Raw logs and the immutable validation clone remain outside the repository:

```text
/home/hamza-usta/.cache/mdslc-h-logs-6796fd8-aM084y/
/home/hamza-usta/.cache/mdslc-h-matrix-6796fd8-VB03Wo/
```

This report does not claim hosted GitHub Actions, Windows execution,
native/OpenBLAS parity, a public API/ABI/backend-contract freeze, or GPU/NPU
support.

## Authenticated environment

- CPU: AMD Ryzen AI 9 HX 370 with Radeon 890M
- topology: one socket, 12 physical cores, 24 logical CPUs, SMT enabled, one
  physical NUMA node
- compiler and LLVM: Ubuntu Clang/LLVM 21.1.8
- MLIR: isolated Ubuntu MLIR 21.1.8 development prefix, coherent with LLVM
- provider: OpenBLAS 0.3.32 pthread when enabled
- generator: Ninja
- build policy: `nice -n 10`, at most two build jobs, serial CTest

The validator audited the host for competing compiler/build processes before
each build. An ASan launch encountered a conservative idleness-race rejection
before any action; it was repeated unchanged after the host was confirmed idle.

## Full configuration matrix

| ID | Build | MLIR | Configured default | OpenBLAS | Configure | Build | Tests | CTest |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| R1 | Release | ON | `matcore-mlir` | required | 1.85 s | 128/128, 50.23 s | 63/63 | 184.23 s |
| R2 | Release | ON | `matcore-mlir` | OFF | 1.59 s | 128/128, 20.96 s | 63/63 | 176.83 s |
| R3 | Release | OFF | `capture-v0` | required | 1.48 s | 106/106, 6.03 s | 58/58 | 143.28 s |
| R4 | Release | OFF | `capture-v0` | OFF | 1.23 s | 106/106, 5.87 s | 58/58 | 140.10 s |
| C1 | Release | ON | `capture-v0` | required | 1.68 s | 128/128, 17.91 s | 63/63 | 172.77 s |
| D1 | Debug | ON | `matcore-mlir` | required | 1.69 s | 128/128, 39.61 s | 63/63 | 285.79 s |

All six full suites passed, for 368/368 registered test executions. The matrix
authenticates both Linux product defaults, the complete MLIR/OpenBLAS Release
toggle grid, and full Debug with the semantic default and provider present.

The MLIR-enabled suites include semantic core, composition/domain verification,
recovered-GEMM analysis, explicit CPU lowering, native object/executable
integration, package consumers, strict C17 ABI, and source-inaccessible
installation. The MLIR-disabled suites include the fail-closed semantic-
unavailable gate. OpenBLAS-disabled artifacts have no provider dependency;
MLIR-enabled installed artifacts do not link the aggregate shared `libMLIR`.

## Sanitizers

ASan+UBSan used Debug, MLIR enabled/default `matcore-mlir`, OpenBLAS disabled,
and the exact hosted in-process allowlist. The build passed 128/128 steps in
86.40 seconds and all 20/20 tests passed in 4.63 seconds, including
`platform.fp_environment.v1`.

TSan used the bounded MLIR-off/OpenBLAS-off runtime profile. Its four requested
targets built in 31/31 steps and all 4/4 shared-state tests passed in 8.62
seconds:

```text
runtime.cpu.execution_context.v1
runtime.cpu.parallel_packed.v1
runtime.cpu.planner_v3_resources
runtime.c_abi.public_context_v1
```

## Install, ABI, and source-inaccessible evidence

The fresh R1 install contained 15 files or symlinks, including all five tools,
the versioned runtime shared library, two public headers, and the CMake package.
The runtime retained exactly 15 public Matcore C exports and SONAME
`libmatcore_runtime.so.0`.

The installed package declared Matcore MLIR available and defaulted to
`matcore-mlir`. Installed consumer, strict C17 ABI, source-inaccessible
consumer, and source-inaccessible safety tests all passed. The package test
registration captured:

```text
expected-source-commit=6796fd85963f985fb652eb8242d37538b29f0765
configured-source-clean=ON
```

It cloned that exact commit and executed after removing access to the staged
producer source/build trees. The installed leak scan found no absolute source,
build, or private MLIR development path. `matcore-plan` retained only the
intended relative `$ORIGIN/../lib` RUNPATH.

## Legacy, integrity, and sanity gates

- legacy Python frontend contract: 14/14 passed
- `git diff --check`: passed
- `tests/check_repository_hygiene.sh`: passed
- `git fsck --full --strict`: exit 0; no corruption was reported
- final detached validation tree: exact candidate, clean, no untracked files

One guarded, pinned, build-idle 256-cubed GEMM run selected single-thread
OpenBLAS, passed correctness, and emitted valid exact-clean provenance and
timing metadata. Its median was 0.221198 ms and diagnostic throughput was
151.694 GFLOP/s. This is only an execution/planner/timer sanity sample. It is
not native-BLAS parity, planner-regret calibration, or a general performance
claim.

## Supported and withheld conclusions

Supported locally:

- the exact candidate passes the declared Release, compatibility, Debug,
  sanitizer, package, ABI, artifact, legacy, and hygiene gates;
- the explicit semantic CPU route is a real executable path, not an unused
  sidecar;
- absent MLIR/provider capabilities fail closed where required;
- forced non-provider v2 requests avoid provider conformance work while
  retaining truthful linked-but-uninspected diagnostics; and
- context creation remains the explicit all-variant validation boundary.

Not supported by this local lane:

- hosted Linux or Windows validation of the pull-request head;
- Windows Matcore-MLIR execution;
- native-BLAS parity or Milestone 7 completion;
- executable map/domain or recovered-loop replacement;
- multi-node NUMA or AMX runtime claims;
- a merged/tagged beta or public release; or
- public API/ABI/backend-contract freeze or accelerator work.
