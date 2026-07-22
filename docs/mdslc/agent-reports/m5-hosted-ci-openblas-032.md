# Milestone 5 hosted-CI OpenBLAS provider repair

## Scope and ownership

This lane owns only `.github/workflows/mdslc-native.yml` and this report. No
compiler, planner, runtime, ABI, package, or test source was changed.

## Failure and contract

PR #11 job `89032271429` installed Ubuntu 24.04's OpenBLAS 0.3.26 development
package. That release does not provide the `openblas_set_num_threads_local`
surface required by MDSLC's provider-local thread policy, so the existing
coherence probe correctly rejected it. The repair preserves that fail-closed
probe instead of weakening the API or silently accepting an older provider.

The required matrix lane now downloads the official OpenBLAS 0.3.32 source
release from:

```text
https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.32/OpenBLAS-0.3.32.tar.gz
```

It verifies SHA-256
`f8a1138e01fddca9e4c29f9684fd570ba39dedc9ca76055e1425d5d4b1a4a766`
before extraction. The isolated build under `RUNNER_TEMP` is Clang 21,
x86-64, pthread, LP64, shared-library, CBLAS-only, single-precision, generic
ISA, and capped at 64 provider threads. The generated pkg-config and loader
paths are exported only in the required matrix job. CI checks both
`cblas_sgemm` and `openblas_set_num_threads_local`, requires pkg-config version
0.3.32, and authenticates the exact runtime dependency path.

The explicitly disabled job no longer installs the Ubuntu OpenBLAS development
package, never enters the conditional source-build step, configures with
`MDSLC_ENABLE_OPENBLAS=OFF`, and retains the negative `ldd` assertion.

## Local reproduction

The official archive and build were reproduced outside the repository at:

```text
/home/hamza-usta/archives/MatcoreDSL-CI-OpenBLAS-0.3.32/
```

Observed evidence:

- archive checksum: passed;
- source build and install: passed;
- installed pkg-config version: `0.3.32`;
- installed symbols: `cblas_sgemm` and
  `openblas_set_num_threads_local` present;
- required MDSLC configure: reported
  `MDSLC OpenBLAS variant: 0.3.32 via pkg-config`;
- required MDSLC build: 89/89 build edges completed;
- required runtime dependency: resolved to the isolated 0.3.32 prefix;
- required tests before committing this workflow change: 40/42 passed; the
  two expected failures were benchmark `--guard` rejecting the intentionally
  dirty source-worktree provenance;
- disabled MDSLC configure: reported provider unavailable;
- disabled MDSLC build: 89/89 build edges completed;
- disabled runtime dependency: no OpenBLAS `DT_NEEDED`/`ldd` entry;
- disabled provider/planner/runtime focused tests: 5/5 passed;
- workflow YAML parsed with `yq`; `git diff --check` passed.

The clean-commit full-suite rerun is performed after this focused commit and
handed to the integration owner as terminal evidence.
