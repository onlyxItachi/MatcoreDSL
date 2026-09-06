# Installed region driver: integration HOLD

2026-09-07. This is a research/integration checkpoint, not canonical capability
or a completed foundational compiler claim. The driver must not merge yet.

## Exact implementation points

- `0d880002402f3d04de0f96a77749dbcd1419a90a`: executable driver with original
  Clang host preservation, named-function ABI thunk, compiler-owned helper
  capture, source/header/artifact rechecks, no-clobber output, pinned Clang and
  linker, ELF output verification and exact runtime/provider export reservations.
- `2e4a59eb7fb71ab342fb863458cdbaa948c3caf6`: additionally uses upstream LLVM
  internalization on the independently compiler-issued helper module before host
  linking. Only its entry and required LLVM appending metadata retain external
  definitions; remaining externally replaceable definitions fail closed. External
  runtime declarations remain external, and ordinary host linkage stays intact.
- Independent failing fixtures and report:
  `mdslc/experimental-driver-independent-v1` at
  `4c689c6a57dfce3a2e700be92d909adc7d67b688`.

The branch contains reviewed prerequisite implementations but has not yet
reconciled its complete CMake/CI registration with the final canonical/package
integration heads. No full driver regression or hosted-driver acceptance claim.

## What actually ran

At the first point, focused Release source/driver/artifact tests passed 3/3,
17.03 s. Focused Debug ASan+UBSan source/driver/artifact/ABI tests passed 4/4,
30.30 s. An earlier sanitizer invocation was interrupted while llvm-symbolizer
waited on debuginfod; it was not counted. The passing rerun explicitly cleared
`DEBUGINFOD_URLS` and used the established sanitizer environment.

A fresh Release, required OpenBLAS 0.3.32, `BUILD_TESTING=OFF` production build
completed 109 Ninja steps. An actual installed driver in a prefix containing
spaces and a comma compiled the ordinary sample into a generated executable.
Both rectangular GEMM output and the second-operation failure-prefix path ran
successfully on the local x86-64 CPU. The same installed compilation and execution
also passed with the complete original source/build worktree hidden by bwrap.
This proves bounded installed execution, not the missing hostile-link contract.

At the helper-internalization point, affected Release tests passed 3/3, 17.06 s.
The added real weak-function interposition oracle then passed the ABI suite 1/1,
0.91 s: the host keeps its own return value 91 while the compiler helper retains
its issued return value 7. Updated ASan+UBSan source/driver/ABI tests passed 3/3,
29.18 s. This correction still lacks final independent review.

## Falsification results that prevent acceptance

1. **Observed and corrected:** pinned Clang originally selected a caller-PATH
   fake linker and published ASCII after its successful exit. The independent
   reviewer reproduced the failure, then confirmed the pinned-linker/ELF-kind fix
   produced an actual executable with correct math under the same hostile PATH.
2. **Observed and corrected within a bounded helper contract:** weak helper
   definitions could participate in ordinary host LLVM ODR selection. The new
   upstream internalization step preserves independent private implementation
   bodies; its concrete real-execution oracle and sanitizer checks pass locally.
3. **Observed and unresolved:** whole-archive linkage does not protect the weak
   private `ObservationAbiV2` cleanup destructor. A host assembler-label definition
   exits 71 on observation allocation failure. An ordinary private namespace/type
   definition without assembler syntax exits 72. The ordinary successful execution
   path passes, demonstrating why arithmetic-only tests were insufficient.
4. **Rejected interpretation:** a ban on assembler-label definitions alone does
   not fix the named-type adversary; raw linker-name C functions are another
   reason not to build an ever-growing syntax blacklist.
5. **Rejected interpretation:** checking shared-library exports alone leaves
   static private-archive weak definitions unprotected. Conversely, introducing a
   private DSO alone leaves earlier helper LLVM weak definitions unprotected.

Other useful preserved negatives: replacing the original source body as text
changed later source-column observations; mixed static/shared LLVM duplicated
APFloat identities and corrupted ordinary float IR; pointer identity of distinct
LLVM identified sret types was not C++ ABI equivalence. The branch retains their
focused corrections and tests. See `decisions/REGION_HOST_LINKAGE_V1.md` and the
ABI/artifact independent reports.

The exact libstdc++ `<iostream>` declaration
`.globl _ZSt21ios_base_library_initv` is accepted only as exact non-defining bytes
and only if its name is not in the reserved artifact set. General module/inline
assembly remains outside this bounded driver. Tests cover suffix directives,
labels, quoted spellings, reserved targets and actual standard-header inclusion.

## Single next implementation boundary

**Finish intrinsic private implementation ownership across both links.** Retain
the helper isolation and prototype a private candidate-runtime DSO with hidden
implementation, local binding of its own definitions, and a minimal explicit
unfrozen export contract for required Session/Value/Result/Observation entrypoints.
The existing legacy Runtime DSO must remain the sole provider-policy owner.
Avoid any change to mathematical, numerical, storage or failure semantics.

The DSO approach was selected for prototyping over syntax blacklists and fragile
C++ mangling-prefix classification. It is not yet implemented or proven. Require
the hostile cleanup programs either to fail closed for a precise ownership
reason or to execute with the genuine implementation, preserved failure prefixes
and no forged exit. Check constructor/cleanup/helper data/COMDAT/sanitizer paths,
allocator contracts, actual provider identity, mixed-STL owning handles, installed
relocation, feature-OFF compatibility and exact final-head CI before acceptance.

Do not treat `-c` as authority for a later arbitrary manual link. Deployment also
requires a stable trusted toolchain/runtime/provider/standard-library environment;
the driver is not a sandbox or a self-authenticating executable format.

## External interruption

Both independent reviewer agents exhausted the service usage allowance before
the encapsulation design could receive independent review. The driver has not
been accepted or merged. Preserve this branch, the negative fixture branch and
the separately reviewed package PR. Resume with live canonical/branch/CI checks,
then complete and independently falsify the ownership boundary. The foundational
compiler terminal condition is **not** satisfied by this checkpoint.
