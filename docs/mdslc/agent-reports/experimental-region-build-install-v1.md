# Experimental region production build/install validation

Implementation checkpoint: `9e8fbd773b32bea11ca41dd995941391dd4501e0`, branch
`mdslc/experimental-region-build-install-v1`, based on the composed runtime and
frontend checkpoint `de0911eebc788b2d05266efadb9d3902ebc2b7c7`. The implementation
changes CMake and build documentation only. No runtime/source semantics, driver
option, ABI promise, issue disposition or canonical merge is changed here.

## Observed validation

Host: Linux x86-64; Clang/LLVM 21.1.8; exact MLIR 21.1.8 at the local extracted
`mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir` package. Builds use Ninja and
two compile jobs. Build roots are under the isolated worktree
`/home/hamza-usta/MatcoreDSL-wt-experimental-region-build-install-v1`.

| Configuration / command surface | Exact result |
| --- | --- |
| `build-regions-offtests`: Release, feature ON, `BUILD_TESTING=OFF`, OpenBLAS OFF; full default build | 97/97 build steps succeeded |
| Same build, directly run `experimental_regions_install.cmake` against a fresh prefix | 37 owning-result assertions and 121,215 candidate assertions passed; exactly one issued leaf definition, no injection exports, no public LLVM/MLIR dependency |
| `build-regions-tests`: Release, feature ON, tests ON, required OpenBLAS ON (actual 0.3.32) | Full 236-step default build succeeded |
| Same build, smallest affected region/runtime/source/generated/package regex | 38/38 CTests passed, 43.94 s |
| Same clean checkpoint, all CTests except separately run source-inaccessible package test | 111/111 CTests passed, 218.34 s; includes legacy CPU/native/provider, frontend, IR/MLIR, generated leaf/object/ASan negative control, admission, source execution, package consumers and benchmark-contract tests |
| `build-regions-feature-off`: Release, feature OFF, MLIR OFF, tests ON, OpenBLAS OFF; full default build | 136/136 build steps succeeded |
| Feature-OFF selected package/legacy/frontend/FP/C-ABI tests, clean checkpoint | 12/12 passed, 23.62 s |
| Feature-OFF `package.installed_source_inaccessible`, exact clean checkpoint | 1/1 passed, 31.40 s |
| `build-regions-config-asan`: Debug, feature ON, tests OFF, OpenBLAS OFF, sanitizers set **only** in `CMAKE_CXX_FLAGS_DEBUG=-g -fsanitize=address,undefined -fno-omit-frame-pointer` | Full 97-step build succeeded |
| Same config-only sanitized build, installed consumers under `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`, `UBSAN_OPTIONS=halt_on_error=1`, `DEBUGINFOD_URLS=` | 37 + 121,215 assertions passed; installed actual leaf heap-out-of-bounds negative control was detected inside the generated kernel |
| Unsupported configuration probes | Missing native frontend, missing MLIR, 22.1.8 compatibility tuple, and cross-compilation each rejected with the experimental-feature diagnostic |
| Repository hygiene | `git diff --check` passed; implementation checkpoint was clean for provenance-dependent package tests |

The installed consumers deliberately include only installed public/private
headers and link the installed registry archive plus the existing runtime DSO.
They do not use LLVM/MLIR includes, libraries or exported CMake targets. This
tests runtime/primitive packaging, **not** a new installed source compiler.

## Negative results retained

- Initial directory split dropped the always-ASan object's generating rule:
  Ninja reported `strict-asan.o` missing. Giving each generated object an
  explicit custom target in its production owner directory fixed the
  cross-directory dependency; subsequent normal/always-ASan execution and the
  negative control passed.
- Initial configure dependency spelling contained both canonical paths and
  `../..` spellings of the same header; Ninja rejected duplicate phony outputs.
  Canonicalizing the owner directory fixed this without touching header bytes.
- Independent CMake review identified the inherited global-flags-only sanitizer
  selection gap. The implementation now considers active single-config flags;
  the fresh Debug-config-only sanitized build and **installed generated-code**
  OOB control demonstrate the correction, not just compiler instrumentation.
- The first feature-OFF consumer run correctly failed benchmark provenance
  because the CMake worktree was still dirty when built (11/12 other selected
  tests passed). After committing, reconfiguring and rebuilding the provenance
  target, the same 12/12 surface passed. No guard was weakened.

## Artifact identities

These are local build artifacts, not portable/reproducible-build promises:

```text
Production CMake owner:
66d1bd4943bd5d8c1888b3b73db844fb0cf1a7a77f6a4cf2482f3ccb0e947cef
Top-level CMake:
1cc5c4ab6e74fc791fb6cb49a1fa9f80532f1e48ac5aaea17cd36466c4cde928
Installed-package test script:
a1ec264ad1d23f3f254744a32511f0e7ea995f50afa818dff6d92e388d577441
Release/OpenBLAS-OFF production registry archive:
9c12ca15b4e2f2d673e7fc05c48e58f7292512e245ef22e8e450d068307bbdf9
Release actual strict-normal.o:
8a5334297a2b680f2df2cea28945ce773b4b250abfa567b19f313c446b50b847
Debug-config-only ASan actual strict-normal.o:
7c7a6986a264cde615c804badc296cac1c681c6756334148ee3dc6ee84137892
```

The sanitized manifest says `address_sanitizer=function_attributes`; the
resulting object imports the expected `__asan_report_load4`, `load8`, `store4`
and initialization functions. The normal manifest says `address_sanitizer=off`.

## Integration handoff

See [build/install contract](../EXPERIMENTAL_REGION_BUILD_INSTALL_V1.md) for exact
target and installed-path variables. Root owns final independent CMake review,
integration with the separate opaque private-Value revision and connected
source-driver/compiler targets, hosted CI and any canonical merge. CPU-agent
review was partial (it found the sanitizer issue) before being reassigned;
this record does not fabricate a final independent approval.

Windows execution was not performed locally. Feature OFF preserves its explicit
header/package boundary; feature ON is rejected outside native Linux x86-64.
No performance, parity, zero-copy, accelerator or new region-language execution
claim follows from this build/install checkpoint.
