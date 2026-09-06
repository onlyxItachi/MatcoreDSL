# Opt-in region build/install integration

## Provenance and scope

The independently reviewed promotion implementation
`9e8fbd773b32bea11ca41dd995941391dd4501e0` and its evidence commit
`6caccb78e2945f96db66bbf5b8ef8345b2810e4a` are retained through normal merge
`91fd0abf5edceb248e3426c97ea50932aada4c7d`, composed with opaque-Value integration
head `fd360cd0c4f1c2bae886d4d9e4b2a05055d6e4e7`. Focused installed-consumer and
CI additions are `3b58676437d471f7f0ddccfadb07999d488f60f4`. After the clean-head
tests below completed, normal merge `4513c8b58fa1ac080e867dbc2022c5ef0a3458c5`
inherited canonical PR #52 (`3374ffbb2100dd68fdd34a46ee93495d1c3c4137`); that
merge changed only two documentation files. Branch:
`mdslc/experimental-build-install-integration-v1`.

No runtime, frontend, semantic, emitter or public/private header implementation
bytes change in this integration. The production CMake owner is byte-identical
to the original reviewed promotion. The sole merge conflict reconciles the old
uninstalled-header check: keep it when the feature is OFF; feature ON intentionally
installs the complete experimental support package and checks that contract.
The feature defaults OFF, requires the exact native Linux x86-64 21.1.8 tuple,
and does not add a source driver or execution authority to source intrinsics.

## Mechanical integration checks

- Fresh feature-ON installs contain the reviewed public declarations, private
  revision-coupled adapter header and production candidate archive with exactly
  one issued generated leaf. LLVM/MLIR and private targets do not leak into
  exported consumer dependencies; no injection or embedded provider symbols
  are admitted in the archive.
- Installed Result, candidate and independently authored Value/OOM/lifetime tests
  compile against installed include paths, archive and runtime DSO only. Normal
  producers and `_GLIBCXX_DEBUG=1` / `_GLIBCXX_USE_CXX11_ABI=0` consumers exercise
  both Result and Value ownership across translation-unit configuration changes.
- The installed archive itself passes matching-artifact execution controls and
  rejects a renamed-revision artifact, including stale constructor COMDATs first
  and last at O0/O2. Failed links must name the actual missing revision; unrelated
  linker failures cannot satisfy the test.
- ASan installs instrument the actual issued leaf and run its intentional
  out-of-bounds negative control. No test-library or parent-only instrumentation
  substitutes for that generated-code evidence.
- Feature-OFF installs exclude the experimental headers/private package. The
  established legacy runtime and optional provider retain ownership and behavior.

CI enables this opt-in package in the existing Linux MLIR Release/Debug/sanitizer
lanes, preserves feature OFF in non-MLIR and Windows configurations, and retains
all earlier regression selections. The sanitizer set is exactly **62 + 1 = 63**,
adding the installed package execution/control test. A Release provider-OFF lane
also builds and executes installed consumers with `BUILD_TESTING=OFF`.

## Exact local validation

All rows ran with a clean tree at implementation/integration head `3b586764...`.
Linux x86-64; Clang/LLVM/MLIR 21.1.8; Ninja with at most two compile jobs and
serial CTest execution. Build roots are under
`/home/hamza-usta/MatcoreDSL-wt-experimental-build-install-integration-v1`.

| Configuration and scope | Observed outcome |
| --- | --- |
| Release, feature ON, OpenBLAS OFF, `BUILD_TESTING=OFF`, `build-package-production` | Full 97-target build PASS; fresh installed Result 37 checks, candidate 121,215 checks, independent Value 85 checks plus 32,000 owning cycles; mixed Result 8,000 cycles, mixed Value, and actual installed revision/stale-COMDAT controls PASS |
| Release, feature OFF, MLIR OFF, OpenBLAS OFF, `BUILD_TESTING=OFF`, `build-package-off` | Full 56-target build and fresh legacy-only installation PASS; no closed-region/test target required |
| Release, feature ON, required OpenBLAS 0.3.32, `build-package-release` | Full 243-target build PASS; smallest package checks 2/2 PASS, 5.59 s; affected regression scope 38/38 PASS, 28.99 s |
| Same Release build, complete CTest registry | **116/116 PASS, 283.47 s** |
| Debug ASan+UBSan, feature ON, OpenBLAS OFF, `build-package-asan` | Full 245-target build PASS; smallest installed package check 1/1 PASS, 4.22 s |
| Same sanitizer build, independently enumerated exact affected scope | **63/63 PASS, 57.99 s**, including installed generated-leaf OOB refusal |
| Repository hygiene and `git diff --check` | PASS |

Sanitizers use `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`
for C/C++ and matching executable/shared linker flags. Environment:
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`,
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, `DEBUGINFOD_URLS=`.
The 63-test result is the affected sanitizer scope, not all registered subprocess
tests. Full Release includes the legacy installed/C17/source-inaccessible package
checks, runtime/provider/planner surfaces and both admitted-source variants.
Benchmark-named contract tests do not establish a new performance result.

## Independent review and remaining boundary

The root integration reviewer independently accepted normal merge `91fd0ab` and
focused change `3b586764`: feature-ON/OFF ownership is coherent, the 63-test union
retains the prior set, and installed mixed/revision consumers defend the actual
opaque-Value package. A second static reviewer found no blocker in the installed
consumer additions; that review did not independently rerun these builds.
These are engineering reviews, not fabricated GitHub approval events. Hosted
results and any canonical merge belong to the later PR/checkpoint record.

SHA-256 identities:

- Production CMake owner: `66d1bd4943bd5d8c1888b3b73db844fb0cf1a7a77f6a4cf2482f3ccb0e947cef`.
- Root CMake: `ba9993e5a21eb94780be0bea3a1b34e6965e04e6ffe1175b5ccb2be9afe436c2`.
- Installed-consumer script: `81ff70ebc43427809260344bd33f27f01d76123fdb18c78b9d7717d826228e47`.

The next justified boundary is the separately reviewed authenticated named-region
source-to-executable driver using these exact installed artifacts. This PR does
not implement that driver, generated fusion or additional targets; it does not
close Native BLAS Parity, freeze any API/ABI, or establish zero-copy/performance
claims. Windows feature-ON execution remains unsupported and was not tested.
