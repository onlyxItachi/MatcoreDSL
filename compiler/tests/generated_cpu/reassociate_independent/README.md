# Independent reassociate issuer controls

These fixtures are independently authored against issuer `1c03306` and its
test-only classifier correction `62379dd`. They do not change production code.
The integration owner should add `add_subdirectory(reassociate_independent)`
inside the existing generated CPU test directory, after the issuer target exists.
The included CMake file registers two tests.

`mutations.cpp` checks 47 assertions with structurally valid MLIR/LLVM mutants.
It explicitly separates narrow-envelope checks from exact-replay checks, checks
wrapper data-pointer binding and non-invented pointer guarantees, and preserves
the permitted diagnostic-location/module-comment differences. The deliberate
renamed-buffer positive is self-consistency, never source authentication.

`classifier_test.py ../expect_reassociate.sh` has 14 synthetic controls. Ten
initial refusal cases, the later-allocation-stack owner adversary, two valid
widths and a preserved skip-77 case test classification only. No synthetic text
is presented as evidence that generated instructions actually executed. The
allocation-stack case fails against the old `1c03306` classifier and passes
against `62379dd` without weakening its expected outcome.

The independent report records real generated-leaf execution separately from
these mutations and synthetic diagnostic controls. No generated artifacts or
logs are committed.

## Connected source and fresh installed package

`connected.mdsl` is an independently authored ordinary source fixture: full
8x8 `gemm(v,v)`, shifted overlapping publications/late reads, an insufficient
second-publication capacity, owning observations, FP/errno restoration, and a
same-Value FMA discriminator. `installed_identity.cpp` is a separate private
adapter consumer. Public `Result` has no candidate identity telemetry.

`connected_test.py --driver /absolute/build/bin/mdslc-region --output /fresh/path`
runs five source policies. Add `--provider /exact/configured/libopenblas.so` for
the sixth policy. Add `--sdk /fresh/installed/prefix` for the installed source
matrix and private DSO owner/report consumer. `--sanitized` expects the actual
ASan/UBSan driver installation and adds sanitizer flags only to the ordinary
private consumer's final link; source sanitization belongs to the installation.
The output directory must not already exist, including on repeat CTest runs.
The default ordinary compiler is `/usr/bin/clang++-21`, overridable with `--cxx`.

The runner verifies exact output counts, all input/produced-executable hashes
before and after every subprocess, empty positive stderr,
source executable instrumentation profile, direct private/runtime dependencies,
local issued leaf/wrapper definitions, and the configured provider dependency
for the manual consumer. It removes loader override environment variables.
Four source rounding-label swaps and one installed-owner swap are deliberate
failing-executable controls; without `--sdk`, the two build-source controls run.
An exact diagnostic-free hardware-unavailable exit 77 propagates as suite SKIP,
not success; CTest registration must set `SKIP_RETURN_CODE 77`. Wrong skip text,
other exit codes and sanitizer/crash diagnostics cannot excuse a failure.
`runner_test.py` supplies synthetic classifier and changed/missing-file controls,
separately from actual source execution.
No hidden-source/package-relocation or benchmark claim is made. See the separate
`generated-reassociate-connected-independent-v1.md` evidence report.
