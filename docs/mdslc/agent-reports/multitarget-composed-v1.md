# CPU/GPU correctness composition v1

This integration branch composes existing bounded implementations; it adds no
source syntax, target, automatic-selection policy, numerical permission or
performance claim. Neither pending branch is made canonical by this local merge.

## Exact pre-merge inputs

- First parent (GPU): `65b6e9eacc145bd514e416bced3513655ebd7db0`.
- Second parent (CPU): `665cf3756c58196b5f58acc2f784e92f7b94b52c`.
- Shared ARM dependency: `8157eabed52009038b5dfe584d9d7c3d52996b97`.
- Integration branch: `mdslc/multitarget-composed-v1`; normal merge, no squash or
  rewritten implementation history. The original CPU/GPU worktrees are untouched.

The planned canonical landing order is CPU first. Existing Candidate and
ClosedCpuPolicy values 0–5 and Implementation values 0–8 are unchanged. CPU
AVX/AVX2/AVX512F retain Candidate/Policy 6/7/8 and Implementation 9/10/11. GPU
NVVM/ROCDL append as Candidate/Policy 9/10 and Implementation 12/13. Both branches
originally occupied the same new slots, so retaining both unpublished assignments
would be impossible. Explicit enumerator values and compile-time regressions in
the existing candidate/source tests protect the selected composition order. This
is not a public/wire ABI freeze or permission to mix unmatched private artifacts.

## Conflict and automatic-merge review

Five conflicted files retain both implementations: the policy declaration and
symbolic emitter mapping, runtime enum declarations/dispatch/report names, and
driver parsing/usage. Runtime dispatch keeps complete independent preprocessor
blocks. The GeneratedMemref declaration remains available to STRICT_ISA-only
synthetic tests. GPU candidate interfaces keep their documented host-pointer,
synchronous staging contract rather than claiming device-resident source values.

Automatically merged CMake retains the three ISA objects, capability discovery,
GPU image/runtime dependencies, hidden private DSO and native-only exclusions.
The driver retains GPU dependency snapshots, symbol ownership and explicit
link/RUNPATH handling. CPU hardware/OS-state and GPU availability/shape guards
remain before allocation and empty/zero-K shortcuts; default selection is
unchanged. Hosted workflow composition retains CPU count gates 12/13/175 and
the optional vendor-independent GPU issuer build. No test selection is removed.

## Validation boundary

Prior separate CPU and GPU results belong to their exact recorded heads, not
this new composition. Fresh combined Release validation uses coherent
LLVM/Clang/MLIR 21.1.8, the established GPU SDK/driver tuple, OpenBLAS 0.3.32,
isolated SSD build/TMPDIR, two compile jobs and the shared heavy-link lock.
Independent composition review, focused enum/registry/guard/source/package checks
and a frozen full standalone run are required before this branch is eligible
for further integration. No source edits are permitted during that full run.

Before freezing the merge, independent staged review found no source-level
blocker. The fresh combined build completed 465/465 actions. Initial registry,
ISA issuer/facts/direct execution, GPU issuer/qualification/registry/object and
physical-adapter checks passed 27/27 (12.39 seconds), zero skips or failures.
The baseline strict LLVM/object and all three normal ISA objects are byte-identical
to the prior CPU build. Both GPU fill/GEMM LLVM outputs and their cubin/hsaco images
are byte-identical to the prior GPU build. These comparisons establish unchanged
leaf artifacts, not by themselves correctness of the composed runtime connection.

The combined Release configuration schedules 379 tests, including 13 closed
candidate, 43 strict-ISA and 148 generated-GPU tests. The existing region-driver
selection still contains 42 tests with OpenBLAS. Fresh frozen source/package and
full-suite outcomes must be recorded against the resulting merge commit.

## Frozen combined execution and configuration correction

Normal composition commit: `a67276a072e56531c3e4810f3d172f7782d72a7c`, exact
parents above, tree `3361c037783519e7640ef3d668bf5a88b102ae5b`.
Independent immutable source review accepted all five conflict resolutions and
automatic merges. Focused source/ownership/installed-package execution passed
**28/28**, no skips (294.59 s), including all nine ISA source controls and all
four GPU built/installed-source paths, each with forced hidden-device refusal.
The clean frozen full suite then passed **379/379**, no skips (797.71 s).

That full run explicitly configured `/usr/bin/python3`; it did **not** establish
unseeded test registration. Hosted evidence exposed an empty-interpreter test
command in a fresh configure. The narrow correction
`4fc4560ce16e5c4d9be070fcb465ce2ced317dd2` adds own-scope interpreter discovery
before GPU test registration and records the failure, without changing compiler,
runtime or kernel code. See [counterexample and reproduction](gpu-test-discovery-fix-v1.md).

At clean fixed head `4fc4560`, tree
`e45e2f4259ae9df583aebcf5b6f81e8c35a909ff`, the combined profile reconfigured
and rebuilt successfully, then passed a second frozen full **379/379**, zero
failures/errors/skips (675.84 s). Independent review parsed all 379 JUnit cases,
including the actual CPU ISA paths, both GPU physical adapters, all four GPU
source/installed-source paths and package/source-build-inaccessible controls.
The closed installed-source/build-inaccessible test took 121.417 s. No source
edits occurred during either full run.

Separately, a new **unseeded**, SDK-free build at `4fc4560` discovered Python 3.15,
built the driver in **98/98** actions and actually ran both compiled-out GPU
source tests: **2/2**, zero skips (6.08 s), independently replayed **2/2**
(6.06 s). This is clean-configuration and refusal evidence, not device execution.

Task evidence root:
`/home/hamza-usta/mdslc-work/multitarget-v1/builds/combined-integration/`.

| Preserved artifact | SHA-256 |
| --- | --- |
| `full-a67276a.xml` | `3e4bc7b77dfebf624a706c75c27f600e53b69cf1f4b03a270c88a5213310698f` |
| `full-a67276a.log` | `5c8fcb5888d08e9f3c8349f140a432611d3899591e0a682f0fe32535f3902d67` |
| `full-4fc4560.xml` | `82f82b7c4be5ab747e38f6d5b4fb47c4667e19abb5a654f2078167cebd765351` |
| `full-4fc4560.log` | `14ecd764e08aa91eaa9f106760869ba9775af89bb7ec3e30e38844ca1330d244` |
| `configure-4fc4560-cache.txt` | `7a4910c2548bb2db6154d3d91796b5210636047ccaf30b10d44ec658044ada72` |
| Fixed-head `bin/mdslc-region` | `35b040e1fd6f24a1f6f762071c5e1c751114624d78d79cf4c50d812a9d8a71c5` |
| Fixed-head private candidate DSO | `e6314e0fd410889690951c1a75f7c61d45230c58ea7bee8e649520ccf13c1ed7` |

All three strict ISA normal objects retain their recorded CPU hashes; both
fill/GEMM images per GPU retain their recorded original GPU hashes. Host/runtime
identity changed consistently with the enum composition. Artifact equality alone
was not used to waive the combined source/runtime tests or hosted qualification.

## Qualified canonical checkpoint

GPU engineering [PR #72](https://github.com/onlyxItachi/MatcoreDSL/pull/72) was
normally merged on 2026-09-11 after exact-head independent review and all complete
qualifying PR workflows passed:

- Pre-merge head: `4fc4560ce16e5c4d9be070fcb465ce2ced317dd2`.
- Canonical merge: `0fe537b4246510c57d801577a8d915876e40a941`.
- Canonical parents: `a2bd66d7dd643a9da4eb2f55428739f69357070f` and `4fc4560`.
- Hosted checkout: `34f057b6e1bcab96c8f124f398db6f8b674a8fc5`, parents
  `665cf3756c58196b5f58acc2f784e92f7b94b52c` and `4fc4560`; its tree is exactly
  the frozen local tested tree `e45e2f4259ae9df583aebcf5b6f81e8c35a909ff`.
- Canonical differs from that tree only in the preceding ARM/CPU checkpoint
  documents: `CURRENT_STATE.md` and their two integration reports. The complete
  `compiler/` subtree remains `866b76ec3e15150ea9e3e706d10ae3a1d9bcb035`.

[Native run 34608176113](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34608176113)
completed all eight jobs successfully at the authenticated hosted checkout:

| Configuration | Actually passed | Explicit skips | Test time (s), where recorded |
| --- | ---: | ---: | ---: |
| Release, MLIR OFF, OpenBLAS OFF | 83 | 0 | — |
| Release, MLIR OFF, OpenBLAS ON | 83 | 1 | 240.20 |
| Release, scalar MLIR, OpenBLAS OFF | 233 | 0 | 1230.34 |
| Release, row MLIR, OpenBLAS OFF | 234 | 0 | 1099.97 |
| Release, MLIR ON, OpenBLAS ON | 235 | 0 | 1265.88 |
| Debug, MLIR ON | 233 | 0 | 1071.93 |
| Global ASan + UBSan scope | 161 | 14 | 1288.07 |
| TSan runtime scope | 4 | 0 | — |

The ASan/UBSan skip set is exactly 13 AVX512F hardware/OS-gated tests (four normal,
six instrumented, three source) and one legacy packed eligibility test. The latter
combines hardware/OS/compiler facts; its log does not isolate one cause. This is
not global-ASan AVX512F execution qualification. The other complete MLIR
Debug/Release jobs had no skips, and local physical instrumented ISA execution
is separately recorded. The production `BUILD_TESTING=OFF` scalar and row jobs
also executed installed checks: Result 37, candidates 247255, private Value 85
plus 32000 ownership cycles, mixed result 8000 concurrent handle cycles, and
mixed value PASS. These counts belong to those exact profiles.

[Native ARM run 34608176097](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34608176097)
passed scalar **97/97** (747.90 s) and row **99/99** (755.26 s), zero skips.
Installed source/build-inaccessible execution passed in 178.15/178.20 s.
Artifacts: scalar ID `10267229089`, SHA-256
`00ef2c79355de6bdd332bf0687d108ab7c779c2c585491f9b81ea6b939a3bcda`;
row ID `10268041511`, SHA-256
`3f04a1ad836836afed28800f3dcdaf69d3e37d63f22607e1d4c5743a255abf00`.
This is native ARM strict CPU evidence, not GPU or AVX execution on ARM.

[Windows run 34608176102](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34608176102)
passed **32 tests + one legacy packed eligibility skip** (18.07 s), with its
separate ASan GEMM **1/1** (0.04 s). This preserves the Windows standalone route,
not Windows region/GPU support. [Legacy CI 34608176121](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34608176121)
and [hygiene 34608176143](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34608176143)
passed. Redundant push/obsolete-head runs were cancelled to avoid duplicating
qualification; no incomplete or cancelled run is reported as a passing gate.

Two target specialists reviewed the shared issuer, worker/failure boundary,
source/DSO ownership and tests; independent final review checked conflict
resolutions, artifact hashes, all 379 local JUnit cases, unseeded configuration
and exact hosted checkout/results. No remaining blocker was found. These are
independent agent engineering reviews, not a separate GitHub-account approval
or a formal security audit. Hosted GPU scopes establish SDK-independent
issuance/refusal; the actual physical GPU qualification remains the recorded
local execution on the two named devices. All prior negative evidence survives.
