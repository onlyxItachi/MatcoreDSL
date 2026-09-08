# Independent private-runtime and connected-driver review

2026-09-08. Reviewer: independent borrow/publication lane. **ACCEPT within the
bounded compiler contract**, after the residual-helper correction below. This is
not the final integration, installed-driver regression, or hosted-CI gate.
No runtime, frontend, driver, build, or test implementation was edited by this
reviewer during this review. The reviewer added only this report in a separate
worktree based on `849082e6c9e8f7e40ffe90e75babc2705ee8acc1`.

## Exact scope

- Canonical reconciliation input: `199a841e5049130fffe5a3067c75ba1016613903`
  (operator PR #57); the research branch incorporated it normally at
  `466a58d`. This review does not call the research branch canonical.
- Private candidate DSO: `3730149`, with exact-export and profile-specific
  controls at `103dfd2168785a09ca8c039e21e74c672c52f251`.
- Connected driver: `e9a1d40197d5efe08fe42bd3081fdea904ed6217`.
- Helper LLVM isolation: `2e4a59e`; bounded correction:
  `9f692c2527cf9ce9e6157b0103070d7b4503bd2d`.
- Source ownership/role fixtures: `24595d2cd027d21adcff11673b6fa837562c6413`.
- Installed-driver test addition: `849082e6c9e8f7e40ffe90e75babc2705ee8acc1`.

Reviewed runtime/ownership source identities (SHA-256):

| File | SHA-256 |
| --- | --- |
| `compiler/lib/runtime/closed_host_v1.cpp` | `102d88f0d0430dfca16394b549df8824c2fe54d5beed44900f9c1d51ab6d3cf6` |
| `compiler/lib/runtime/closed_host_v1.h` | `d8f3d0312c4cd71e869f32e350d0aa3baa5bbc139b7bdef948099ad6b7a3db7f` |
| `compiler/lib/regions/private_runtime_v2.exports` | `eb2fd69387eabba04d26abb054a2a032d5f1d94d2edec12f873e604d1aca35a0` |
| `compiler/lib/codegen/AuthenticatedHostThunk.cpp` | `3bcdb1c6985de007b83b4d8e65fd84f3a9babe9a5623724313d3b3b69a777aee` |
| `compiler/lib/codegen/ArtifactSymbolOwnership.cpp` | `84d435c9d80ebe596c1fc10034204be4cb01123d846edb72b97d0dbe39221537` |
| `compiler/lib/codegen/ExperimentalRegionCompiler.cpp` | `43bf153f4b2cfdebb078b27f58f04afeb28862e041b85dce5a3df02fd2600f2b` |
| `compiler/tools/mdslc-region/main.cpp` | `490d883d8b03324d7833407455e267753a1c5693efde289791762107a3bfeae5` |

## Ownership and observable behavior

**OBSERVED:** the DSO change leaves the runtime implementation, public/private
record layouts, numerical permissions, candidate selection, allocation ordering,
immutable ownership, publication and failure behavior unchanged. It compiles the
same private adapter and issued strict leaf. It links the existing Runtime DSO,
not another embedded CPU backend or OpenBLAS policy-state owner.

The exact 47 exported symbols are out-of-line Value/Session/Result/Observation
interfaces, diagnostics and the private revision gate. Implementation records,
STL bodies, cleanup, internal data and generated leaf symbols are not exported.
The actual ELF export equality and SYMBOLIC flags were checked independently.
The two distinct mechanisms are the version script's local scope and local
binding of references to definitions within the DSO. These match the upstream
[GNU ld version-script contract](https://sourceware.org/binutils/docs/ld/VERSION.html)
and [`-Bsymbolic` contract](https://sourceware.org/binutils/docs/ld/Options.html).

**PROVEN WITHIN A BOUNDED CONTRACT:** original-host ELF export ownership,
compiler-helper LLVM isolation, and runtime implementation localization are
separate checks. None substitutes for the others. The driver binds immutable
Runtime/candidate bytes to compiled-in build or exact installed-image digests;
requires exactly one candidate artifact at execution compilation; checks the
original host before helper insertion; and links both DSOs explicitly with
their two search directories. Header/source/artifact identities are rechecked
before publishing a new output without overwriting an existing file.

The original frozen C++ translation unit is replayed without textual body
replacement or early LLVM optimization. Only the sealed region body becomes a
checked ABI thunk. Original direct calls, function pointers, source coordinates,
host cleanup, and remaining ordinary host definitions retain their identities.
Typed aggregate ABI equivalence is structural before and after LLVM remapping;
it is not identified-StructType pointer equality. Canonical C++ record meaning
is independently the frontend's responsibility. Retirement of sealed Value
helpers rejects remaining nonhelper uses; Shape helpers are not swept away.

## Counterexamples and correction

**OBSERVED, archive implementation:** the Release whole-archive controls perform
ordinary successful mathematics, then select source-defined cleanup exits 71/72
on allocation failure. The isolated DSO controls preserve checked errors and
earlier publication instead. The optimized Release DSO also lacks the exact old
`ObservationAbiV2D2` symbol; therefore those executions alone are not causal
proof that a retained destructor was rebound. The independent exact-export and
local-binding checks supply the structural ownership evidence. ASan archives
do not reproduce that particular weak cleanup lowering; their passing sweep is
recorded as an observation, not an archive ownership proof.

**OBSERVED, internal LLVM utility counterexample:** LLVM 21 preserves
`available_externally` bodies during internalization. The original postcondition
also skipped them with `isDeclarationForLinker()`. An independently constructed
helper returning 7 consequently selected a same-named host body returning 91
after LLVM linking; `lli-21` exited 91. Scratch evidence is preserved at
`/tmp/mdslc-available-externally-review.AdIYqa`; composed IR SHA-256:
`63fbad22f45d6d3a29ae31944165991320f2cc72fd4c23cdd354e67187684391`.
Clang 21 emits this linkage before LLVM passes for the GNU-inline specimen at
`-O1`; the fixed current `-O0` helper profile emitted only a declaration. This
was **not** an observed admitted-MDSL source failure. The relevant upstream
behavior is explicit in
[LLVM 21.1.8 Internalize.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Transforms/IPO/Internalize.cpp#L94-L105).

`9f692c2` correctly rejects residual nonlocal bodies using `isDeclaration()`.
It does not forcibly internalize or grant authority to this linkage. The
permanent fixture changes a real Clang helper to that linkage and requires the
specific refusal. The implementer reported 63 checks/one failure before the
fix; the reviewer independently reran the corrected 63-check suite successfully.

The `24595d2` source fixtures separately exercise four generated execution/OOM
prefix cases and twelve export/record refusals in object/executable modes. Their
source and exact diagnostic assertions were reviewed, but their whole driver
CTest was not independently rerun in this review. The private named Session
record is correctly rejected at admission; the assembler-label constructor case
instead retains the ordinary host call while excluding it from issued helper
selection. These are different successful outcomes, not interchangeable claims.

## Independent execution evidence

No rebuild was performed. Tests used the implementer's coordinated stable
`build-isolated-release` and `build-isolated-asan` directories.

| Scope | Actual independent result |
| --- | --- |
| Release, OpenBLAS ON, `^private_runtime\.` | 8/8 pass, 0.09 s; 151358 candidate checks |
| Debug ASan+UBSan, OpenBLAS OFF, same scope | 8/8 pass, 0.13 s; 121215 candidate checks |
| Owning handles in both preceding scopes | 85 Value checks + 32000 ownership cycles; 37 Result checks |
| Release connected source + artifact + corrected ABI | 3/3 pass, 5.15 s; 11 source, 82 ownership, 63 ABI checks |
| Earlier stable Release driver contract at `e9a1d40` | 1/1 pass, 13.76 s; native/generated/automatic math, failure prefix, source traversal, no-clobber/refusals |

An earlier three-suite invocation had two passes and one **NOT RUN** because the
ABI executable had not yet been built. That is not counted as a successful gate;
the later complete rerun above supersedes it. Sanitizer execution used empty
`DEBUGINFOD_URLS`, `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`, and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

The reviewed stable Release driver SHA-256 was
`5159adc3f960c90c551ba6e790068c64e107f8f8b2f59eb294bf5d606547b8e6`.
Candidate DSO hashes were Release
`00c4f784c614d0a6f55b3baed1ce08382ff5a5622c9e62cf041d98bc6aaaccbd`
and ASan
`298b2c2c52705fca03e6c9ef38778252f103d29de7d63474e19500c9f055c93d`.
These identify tested artifacts, not a final-head rebuild or installed images.

## Remaining gates and limits

The `849082e` installation change is accepted as a test design: feature OFF
excludes the driver; feature ON requires its DSO; actual installed-source
compilation/execution has a separate oracle from archive consumers. The reviewer
did **not** execute that new install test or a fresh final-head driver here.
Final installation, source-ownership, full regression, sanitizer and hosted-CI
evidence remains the integration owner's gate, not supplied by this report.

This contract requires trusted native Linux x86-64/Clang-MLIR 21.1.8 toolchain
and library loading, canonical records, valid declared host memory/lifetimes,
no conflicting concurrent access, thread-confined Sessions, and conforming
allocator/deallocator behavior preserving FP state. DSO identity at compilation
does not authenticate future dynamic loading, arbitrary imported libc/allocator
interposition, or a later manual link of `-c` output. No sandbox, crash atomicity,
whole-region rollback, public ABI freeze, new accelerator support, performance
or BLAS-parity claim follows. The old archive is not retroactively isolated.
