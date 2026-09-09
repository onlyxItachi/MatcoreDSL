# Independent authenticated multi-source review v1

## Verdict and scope

**ACCEPT within the existing Linux x86-64 source/compiler/runtime contract** at
`3ef0a78a88a29e41de92dc95b6762e56919b6d4a`, after implementation
`a65a825a2f401c3f74a25dcbd0020f99ccd1aef0` and focused public-host capture fix
`15f16654f310cec1d4d5645a7c0aab8b6bfa11d4`. Independent reviewer authored only
the new tests and this report, not the implementation or its corrections.

Reviewed original-source freezing/replay across all TUs, canonical foreign
declarations, structural ABI and effect comparisons, compiler-issued declaration
and call witnesses, strong/weak/alias/COMDAT and retired-helper ownership before
linking, original region thunks, final no-clobber artifact publication, and the
[bounded public contract](../MULTI_SOURCE_PROGRAMS_V1.md).

No arbitrary LLVM import, public mathematical module format, cross-region
optimization, or whole-program transaction is introduced. Region outputs and
owning observations remain independently observable across ordinary host calls.
The full integration/hosted gates remain separate from this local review.

## Counterexamples and corrections

### Real source: foreign weakref carries a call-only effect promise

The independent fixture first causes Clang to create the genuine region target
through a clean declaration/call, then introduces a differently named static
`weakref` carrying `pure` or `const`. The weakref resolves to the existing target
without changing that target's function attributes; the false memory promise
appears on its call site. Checking target definitions alone is insufficient.

This follows actual pinned Clang behavior: an already existing target is reused
by [GetWeakRefReference](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/lib/CodeGen/CodeGenModule.cpp#L3614-L3636),
and the weakref does not require its own global definition. LLVM also gives
[call-site memory effects](https://llvm.org/docs/LangRef.html#function-attributes)
independent significance; clean callee attributes do not cancel a false caller
promise.

**OBSERVED before correction:** driver SHA256
`67b07b2bb4be6af1cb4a5dd7f34e3ee100ba418053b7d00151b1a1d9e283774b`
accepted the actual source, printed the prelink validation and two cross-TU
link messages, and published an executable. That executable SHA256
`f99b784a3132210ed6b111c3fffbd776f9079c94310ccd52aa15da103f49f006`
passed the existing 37-check mathematical program. This demonstrates admission
of an unauthorized effect promise, **not an observed numerical miscompile**.

The correction examines explicit source alias/weakref/resolver targets, rejects
LLVM aliases/resolvers to protected entries, and compares actual direct calls
against compiler-issued clean call witnesses. Final source fixtures require
valid ordinary C++ followed by checked refusal before any cross-TU link or output
publication. An unrelated truthful weakref to an ordinary utility still works.

### Expressiveness: unrelated overloads and private prelude pollution

Original name-only reservation rejected an ordinary `int first(int)` overload
whose linker symbol cannot replace the sealed region. Exact protected-symbol
matching now admits that overload without permitting alternate source spellings
for the genuine entry. The ordinary utility executes and supplies its expected
37 result to the host program.

Separately, public-region and ordinary-host input capture force-included an
inspection fixture that declares `namespace mdsl_probe`. A valid user variable
`int mdsl_probe` was therefore rejected despite not being reserved by the public
language. The independent single-source fixture is valid with ordinary Clang,
but old ASan driver
`1940ffd64f3bfa643642dc6602c3cf011da9a3d10e104c97f7400f242aad131f`
returned exit 1 with `redefinition of 'mdsl_probe' as different kind of symbol`
and published nothing.

The focused correction removes that unrelated prelude only for original public
region and ordinary host capture. Existing inspection/compatibility defaults
remain. Independent final executions cover the variable in a header-free
utility, in a selected region's physical main TU, and in the single-source driver.
The implementation's separate snapshot tests retain original bytes, input path,
and argument binding rather than compensating with altered source coordinates.

### Synthetic valid LLVM: called pointer address space

The new direct-call comparator originally compared function type, calling
convention and all attributes but omitted the called pointer's address space.
An independently constructed, LLVM-verifier-valid `addrspacecast` of that pointer
was accepted by the comparator: **36 checks, one failure** at `a65a825`.
This is a comparator-level incompleteness, not a demonstrated source-language
escape or raw-IR execution path. The one-line `sameType` check in `3ef0a78`
rejects it while retaining separately parsed, structurally equivalent sret/byval
types in the same owned LLVM context.

## Independent validation

Both final driver/source identities were checked before and after execution.

| Profile | Source suite | Real-Clang ABI/effect unit | Driver SHA256 |
| --- | ---: | ---: | --- |
| Release | 39 / 39 | 36 / 36 | `61da18ddbbf5ef47563d92a3b0bf92657e136d91712731dfcd1fc9f700c1952e` |
| ASan + UBSan | 39 / 39 | 36 / 36 | `060d31e67e4476d74045d77202397682fc32f8c2e1ec20dd26db0da5e2f6fc88` |

The source suite runs six successful multi-TU executables, each requiring exact
`PASS authenticated_multi_tu 37 checks` and empty stderr, plus an independent
single-region namespace/math execution. It reuses the implementation lane's
37-check mathematics fixture; that underlying oracle is not falsely attributed
to this reviewer. Additional checks cover protected weakref-pure/const, a data
definition with the region's assembler label, ordinary function-pointer calls,
existing-output and dangling-symlink no-clobber, and eight synthetic refusal
classifier controls. A sanitizer/crash exit is not accepted as a checked refusal.

The call unit consumes real Clang output with both clean and weakref-altered
calls to the same clean target. Eleven additional **valid-LLVM** mutations cover
calling convention, nofree/memory/nounwind effects, sret/byval alignment and
layout, an invented parameter noalias claim, dead-on-unwind, and called pointer
address space. All refuse. Separately parsed equivalent calls remain admitted.
Both instrumented tests and the instrumented thunk archive were used in the
sanitizer profile; ASan leak detection and fail-fast UBSan were enabled.

Reproduction entrypoints:

- `compiler/tests/program_independent/run.py --driver DRIVER --clang clang++-21
  --compiler-source COMPILER_ROOT --expected-driver-sha256 SHA --output NEW_DIR`;
- build `call_interface_test.cpp` against `matcore_authenticated_host_thunk`
  and shared `LLVM-21`, with its production header and LLVM include paths;
  run `call_interface.cmake` with `CLANG`, `COMPILER_SOURCE`, `TEST`, `WORK`.
  The ASan variant uses the instrumented archive and `-fsanitize=address,undefined`.

Final raw source records under `/tmp/mdslc-multi-source-review.oCbkLO/`:

- `final-corrected-release-2/evidence.json`:
  `b5aadf360437e4c019e881c85afff5dd3f3fddc364a40837659babed748dfe2f`;
- `final-corrected-asan/evidence.json`:
  `56a9e3ff35e2f6c6b99e65369001c07e895d5903bf6ae2aa2de036cee7b8c444`.

The source runner SHA256 is
`31c2d04d5063a5306bad1f497116fe9e603084654aa6e31088a236205e91a0c3`;
the call unit is
`20684ed62fbccfdbb25d213ddc9a496be49b6cc32313cedec45f805ff184c530`.
Reviewed final production hashes:

- `ExperimentalProgramCompiler.cpp`: `881e3b9f91831e8a56277026aa3f86e8dfb0e28d4819062e895ebe499efdcd81`;
- `ProgramHostInterface.cpp`: `22c01f47ea7a553828378f6dea67eff734ecd793ed2b3711da476ac1067cb661`;
- `AuthenticatedHostThunk.cpp`: `62f040f6db1a01242b076e237d6c6dd041ced67f05c5b1e595ab701818cda87a`;
- `ClosedRegionHostInputs.cpp`: `46205507a242062aa2719cea044dac3b084a3844f1903f5f4e6f767948db1b96`.

## Retained negative history and limits

Earlier 19-check and tightened 30-check source runs passed before adding the
namespace controls; they are historical scopes, not the final 39-check identity.
The first namespace fixture incorrectly included the selected entry definition
from another file and correctly failed selection. Its correction puts the
definition in the physical main source; it does not relax production admission.
The initial call-unit draft used unrelated LLVM contexts and an invalid
`memory(none)`/writable-sret combination. Those test-setup errors were corrected
before identifying the one real comparator omission; they are not twelve
production findings.

This is not a sandbox or a general whole-program host-effect proof. An arbitrary
ordinary wrapper/indirect resolver supplying false C++ effect promises remains
outside the valid host contract. Trusted memory, allocator, compiler, loader,
provider and lifetime preconditions remain those of the existing route. No
cross-region fusion, rollback of completed earlier calls, public IR/ABI freeze,
accelerator support or multi-source performance claim is established. No
independent hosted CI or merge result is claimed here.
