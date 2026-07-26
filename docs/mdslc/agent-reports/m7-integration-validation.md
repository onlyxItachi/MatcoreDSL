# Milestone 7 integration validation

Date: 2026-07-26

## Identity and verdict

- Base: `ddda3ccf628dae60bdb7f57d68d024fd02168fcb`
  (`mdslc-cpu-performance-audit-v1`).
- Frozen implementation and test checkpoint:
  `2863253d1f2b06a943c2028ae298d0381d15ddf4`.
- Branch: `mdslc/native-blas-parity-v1`.
- Local integration verdict: implementation gates passed; Milestone 7
  performance envelope **partially passed**.

The implementation can be reviewed and merged as bounded CPU-kernel,
parallel-runtime, diagnostics, and evidence hardening. It must not be described
as completed native BLAS parity.

## Exact local build and test matrix

All build trees were fresh and external to Git. Clang/LLVM 21.1.8 and Ninja
were used with build parallelism two.

| Configuration | Result |
| --- | ---: |
| Release, OpenBLAS 0.3.32 required | 50/50 CTest passed in 133.34 s |
| Debug, OpenBLAS 0.3.32 required | 50/50 CTest passed in 199.43 s |
| Release, OpenBLAS explicitly disabled | 50/50 CTest passed in 122.23 s |
| ASan+UBSan supported scope | 19/19 passed in 20.16 s |
| TSan shared-state scope | 4/4 passed in 7.40 s |
| Installed package/consumer verbose rerun | 4/4 passed in 19.66 s |
| Exact Release ISA artifact rerun | 3/3 passed |

The OpenBLAS-disabled forced-provider request exited 1, selected no variant,
reported `OpenBLAS CBLAS adapter is not linked`, and left no OpenBLAS
dependency in `libmatcore_runtime.so`.

ASan/UBSan used `detect_leaks=1`, fail-fast behavior, strict string checks, and
UBSan stack traces. TSan used fail-fast and deadlock-stack reporting. No
sanitizer diagnostic was emitted.

## Native compiler/artifact proof

The exact Release `mdslc++` compiled `compiler/examples/gemm_v0.mdsl` with
`--save-temps -c`. Ordinary Clang 21 linked the resulting object against the
runtime. Execution printed:

```text
host-before
MDSLC CPU GEMM PASS
```

`file` and `readelf` authenticated an ELF64 x86-64 `REL` object. `nm -C`
showed defined `main`, the generated call-site wrapper, and the unresolved
stable `matcore_runtime_gemm_f32_v0` C boundary. `ldd` resolved the versioned
runtime and optional OpenBLAS provider from the expected locations.

Saved outputs included host, overlay, Matcore JSON IR, sites, stubs, backend,
three component objects, combined relocatable object, and executable.

## Exact ISA evidence

Release artifact checks reported:

- AVX2 checked edge: 100 YMM operands and 16 packed-FMA sites;
- AVX2 full tile: 14 distinct YMM registers, eight packed-FMA sites, no stack
  reference;
- compiler-vectorized body: function-local YMM packed FMA;
- AVX-512 checked edge: 21 ZMM operands and four packed-FMA sites; and
- AVX-512 full tile: 14 distinct ZMM registers, eight packed-FMA sites, zero
  vector stack spills.

The generic runtime remains capability-gated rather than globally compiled for
AVX2 or AVX-512.

## Package, ABI, and repository boundaries

- Relocated installed consumer: configure/build/run, source/header-triggered
  rebuild, and no-op rebuild passed.
- Strict installed C17 ABI: compile/link/run passed and authenticated exactly
  15 public exports.
- Source/build-inaccessible package: configure/build/run passed at the exact
  source checkpoint.
- Source-inaccessible deletion safety: passed.
- Public headers, exported C symbols, and runtime SONAME remain unchanged from
  the Milestone 6 base.
- `git diff --check`: passed.
- `bash tests/check_repository_hygiene.sh`: passed.
- Legacy `tests/test_frontend_contract.py`: all 14 contracts passed.
- No raw benchmark, build, cache, binary, or temporary artifact is tracked.

## Performance collection and partial verdict

The parity runner, summarizer, and their adversarial contracts pass in both
Release and Debug. Complete physical performance collection could not be
finished on the shared host: unrelated Codex sessions repeatedly launched
multi-process compiler, pytest, checksum, and physical-device validation
workloads after quiet prechecks.

Four uninterrupted forward candidates were rejected by the interference
guard. A later authenticated resumable manifest reached 258/368 forward cases,
but there was no complete forward manifest and no reverse manifest. The
incomplete manifest is not used for quantitative parity acceptance.

Retained bounded ABBA evidence shows a real 1.715--1.879x AVX2 and
1.720--1.809x AVX-512 improvement for cooperative B preparation on
`64x4096x4096`. The measured four-thread speedups, 2.14x and 1.82x, remain
below the 3.0x criterion. Full native/OpenBLAS ratios and full-envelope planner
regret remain unestablished.

## Windows and hosted status

The existing Windows x64 portability contract is present and passed inside
the local Linux CTest matrix. The true hosted Windows clang-cl/COFF/PE/DLL,
package, consumer, and ZIP workflow must be rerun on the pull request. No new
Windows performance claim is made.

## Publication constraints

- Issue #15 and GitHub milestone #5 must remain open.
- No native-parity completion tag may be created.
- The pull request must state the partial performance result.
- All hosted Linux/OpenBLAS-disabled/Windows/hygiene checks must pass before a
  normal merge.
- Public API/ABI/backend-contract freeze must not begin automatically.
