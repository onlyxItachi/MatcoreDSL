# Private archive observations across compiler profiles

Scope: test-oracle correction from PR #58 head
`c3ffafb556d7a693430e2648088b922b8fa18ac3`. No runtime, candidate, DSO export,
link policy, compiler flag, frontend or execution-authority change.

## Hosted counterexample

The uninstrumented Debug/OpenBLAS-OFF
[job 102118643373](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34243275177/job/102118643373)
finished its 131-test suite with exactly two failures:
`private_runtime.archive_negative.assembler` and `.named`. Both exited with
their declared substituted-cleanup code, 71 or 72, already on the ordinary
invocation. The separate DSO cleanup tests passed. There was also one unrelated
AVX-512 capability skip. This is not an isolated-runtime failure: the old
archive oracle incorrectly required the optimized Release observation that
only its allocation-failure path selected the substituted cleanup.

Returning from the ordinary fixture still destroys owning results. Its old
comment claiming that return avoided cleanup was incorrect.

## Corrected bounded oracle

The expected exit is exactly 71 for the assembler spelling or 72 for the named
spelling. Both invocations run before classification.

| Ordinary invocation | Failure-sweep invocation | Interpretation |
| --- | --- | --- |
| 0 | Exact expected exit | Historical optimized Release failure-only interception. |
| Exact expected exit | Same expected exit | Ordinary-path interception; neither ordinary success nor completion of the OOM sweep is claimed. |
| 0 | 0 | Allowed only when reproduction is explicitly optional for the instrumented profile; an observation, not archive ownership proof. |
| Anything else | Anything else | Test failure, including unrelated exits, crashes and launch errors. |

Sixty classifier falsifiers run inside each existing archive test. They include
wrong cleanup codes, unrelated numeric exits, textual crash/launch errors,
partial reproduction, and absence of required reproduction. No CTest names or
counts change. The isolated-DSO tests still require exit zero in all profiles;
they do not use this negative-control classifier.

## Actual focused validation

- Existing Release archive binaries: both classified as failure-only, exits
  `(0,71)` and `(0,72)`.
- Existing ASan+UBSan archive binaries: both classified as optional
  non-reproduction `(0,0)`, retaining their narrower evidence status.
- A local diagnostic compiled unchanged `closed_host_v1.cpp` and the fixture
  with Clang 21.1.8, `-O0 -g`, production candidate macros, strict FP flags,
  the existing issued strict GEMM object and existing Runtime DSO. Both
  ordinary and sweep invocations selected 71/72. The old oracle failed both;
  the corrected oracle classified both as ordinary-path interception.
  This reused the Release-issued leaf and Runtime: it is a bounded local
  O0 ownership diagnostic, not a fresh homogeneous full Debug product build.
- All four existing isolated-DSO cleanup executables (two spellings in Release
  and ASan+UBSan) returned zero. Sanitized positives used
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.
- Actual `/usr/bin/false`, `/usr/bin/true` with reproduction required, and a
  missing executable were each rejected. The sixty classifier checks and
  `git diff --check` passed.

Scratch diagnostic: `/tmp/mdslc-archive-observation.kCvFGg` (not required by tests).

| Artifact | SHA-256 |
| --- | --- |
| Unchanged `closed_host_v1.cpp` | `102d88f0d0430dfca16394b549df8824c2fe54d5beed44900f9c1d51ab6d3cf6` |
| New archive classifier | `5066e2188673cb889d5d1966473088cdaf55a86a326c8961026ff8b18e0b0a5c` |
| Local O0 assembler fixture | `b397b14e8d069b139f9a7f539ad443db87eca301d9276ab5408afeb8a0ccf755` |
| Local O0 named fixture | `f694b3c631d04d388337fb08aa68290cb6841e84f06788ae0c8a18ad9fd72814` |
| Reused issued strict leaf | `8a5334297a2b680f2df2cea28945ce773b4b250abfa567b19f313c446b50b847` |

Final exact-head integration and hosted checks remain the integration owner's
gate. This focused correction does not claim those future checks passed.
