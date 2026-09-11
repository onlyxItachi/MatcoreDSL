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
