# Restore the pinned Windows compiler environment

2026-11-26; base `1c1221e89272eb2ad6ff058b8e26de4653f3a93d`.
This is a separate CI prerequisite, not part of the #110/#115/#119 source fixes.

## Observed failure and inferred cause

[PR #123 run](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45827924148)
and the independently dispatched
[unchanged main run](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45828441882)
both fail the ordinary and delayed-template-parsing RapidJSON compile probes
before building MDSLC. The runner selects VS18 / MSVC 14.52.37025 with the
authenticated Clang 21.1.8 archive. The logs contain the generic CMake refusal,
not the underlying compiler diagnostic. Do not report STL1000 as observed.

The released [14.52-family STL header](https://github.com/microsoft/STL/blob/cde2f43b42e079cfb8a1b3f9d0b4676b01367e74/stl/inc/yvals_core.h#L912-L927)
requires Clang 22, whereas the [14.44-family header](https://github.com/microsoft/STL/blob/1f6e5b16ec02216665624c1e762f3732605cf2b4/stl/inc/yvals_core.h#L897-L911)
admits Clang 19 and newer. This strongly supports an incompatible default
STL/compiler pairing; those sources are not byte identities of the hosted copy.
The observed [runner image manifest](https://github.com/actions/runner-images/blob/win25-vs2026/20261116.315/images/windows/Windows2025-VS2026-Readme.md)
includes the side-by-side `Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64`
component.

## Bounded change

Require that installed component, initialize the environment using Microsoft's
documented [vcvarsall toolset selector](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#vcvarsall-syntax),
and verify the selected `VCToolsInstallDir` is in the 14.44 servicing family.
Record its actual STL-header hash/version diagnostics. A missing component or
ignored selection fails closed. The family permits installed servicing updates;
it is not a frozen full MSVC patch identity. The runner, SDK selection, pinned
LLVM/RapidJSON bytes, tests, packaging and sanitizer scopes stay unchanged.
There is no compiler upgrade, automatic fallback, mismatch-override macro,
suppressed compiler error, dependency installation or product-source edit.
Print bounded CMake probe error context on Release configuration failure.

## Qualification gate

The initial local diff/YAML checks are not Windows execution. Acceptance needs
independent review and the complete existing native Windows workflow on this
candidate, including Release/Debug, relocated external consumer, C17 ABI,
artifact checks, focused ASan and distribution upload. A successful configure
alone cannot establish restoration. Until then, the three repair PRs must not
be merged on the strength of older green Windows results.

## First hosted correction result and second prerequisite

Source `6be2cd0451edae06752191a77b4cebdb6b5bf8f4`, with report-only review
commit `53399315c13d0da56eba13eabe6f335dad154897`, configured and built on
[the same Windows lane](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45829291153).
Release passed **51/52**, including the installed consumer/C17 ABI. The only
failure was `support.platform.v1`: its pre-existing assertion requires empty
snapshot input to fail with a diagnostic and no existence/identity result.
The log does not identify which conjunct failed. No later Debug/ASan or
distribution qualification follows from this partially completed run.

`capture_file_snapshot_v1` previously delegated the empty-path decision to
`std::filesystem::absolute`, unlike the neighboring normalizer's explicit
empty-path refusal. Enforce the existing test contract before any filesystem
call. This follow-up is a separate focused production-guard commit, not a
test waiver or a change to the admitted path/numerical/execution model. Extend
the negative assertion to require an empty normalized path/identity chain and
check that a following valid snapshot succeeds and clears the diagnostic.
The original workflow-only change remains intact in history. Independent review
and a fresh complete Windows run are still required for this new source head.

Local follow-up: configured a small Release bootstrap-only build with the
installed Clang 21.1.8, native frontend/MLIR/regions/OpenBLAS OFF, and built only
`matcore_platform_support_v1_test` with Ninja `-j1` (five compile/link steps).
`ctest -R '^support\.platform\.v1$' -j1` passed **1/1**, zero skips on Linux.
This tests the shared guard and valid-path regression, not the Windows native
implementation or the hosted failure witness. Diff and repository-hygiene
checks passed. An initial local YAML assertion script assumed every workflow
step had a `name`; it failed on an unnamed `uses` step. Correcting that local
inspection to use `get` passed without changing the workflow or its checks.
