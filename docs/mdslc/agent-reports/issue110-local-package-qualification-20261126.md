# Issue 110: real Linux installed-package controls

Harness implementation: `883c89dda7c7eb245cafffb8ae926a77205a9b37`, with
independent review at `ba18ae805e60ccedc33678e749364a830fc8cc51`.
The integration owner repeated the focused safety suite: **25 passed,
2 explicit Windows-only skips**, plus diff and repository-hygiene checks.

## Actual installation and execution

Both corrected runners were run twice against the existing Release producer
at `/home/hamza-usta/mdslc-work/issue-hunt-20261112/build-null-maps-release`:
source `0fd5f767dd9ddcf15a19a4223897edd4279eb41e` (the independently reviewed
#119 branch), coherent Clang/LLVM/MLIR 21.1.8, native and bootstrap frontends,
experimental regions ON, OpenBLAS OFF, default `capture-v0`.
This deliberately reuses one producer build: **cross-tree harness integration**,
not a claim that #110's whole branch was separately built or qualified.

Commands used `/usr/bin/python3 -B` and `/usr/bin/cmake`:

- `compiler/tests/package/run_installed_c17_abi_probe.py`, with the producer
  above, `--c-compiler /usr/bin/clang-21`, and external owned container
  `.../issue-hunt-20261112/package-probes-20261126/c17`.
  **2/2 compile/link/run PASS**, each authenticating all **15 public exports**.
- `compiler/tests/consumer/run_installed_consumer_test.py`, with the producer
  above, `--source-dir` this branch's `compiler/tests/consumer`,
  `--cxx-compiler /usr/bin/clang++-21`, `--matcore-mlir-available ON`,
  `--default-semantic-pipeline capture-v0`, and external owned container
  `.../issue-hunt-20261112/package-probes-20261126/consumer`.
  **2/2 PASS** for relocated configure/build/run, source/header/runtime-header
  rebuilds, no-op behavior and the semantic-capability override.

Each pair used the same container in order; the first invocation began with
a nonexistent container. The printed retained children are:

| Runner | First | Repeated |
| --- | --- | --- |
| C17 ABI | `run-ybpo2uzs` | `run-r_pawkxn` |
| CMake consumer | `run-5neeeipb` | `run-8xz8b609` |

No reset, cleanup, ownership-marker synthesis or adoption of old test outputs
was performed. The disposable focused suite separately checks prior-run
contents/metadata remain unchanged; these real integration runs establish
installation/execution and retention, not a hostile-filesystem guarantee.

## Remaining gate

Native Windows (including reparse/junction and external-consumer behavior),
same-head hosted regressions, and merge qualification remain pending.
PR #125 separately addresses the reproduced Windows baseline toolset and
empty-snapshot contract failures. Neither historical green CI nor a Linux
synthetic Windows-path test substitutes for native execution of this change.
