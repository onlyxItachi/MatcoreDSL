# Authenticated multi-TU research v1

2026-09-09. Bounded research result: **one-invocation source ownership is a useful
next multi-file capability without independently sealed semantic modules**.
Actual ordinary host main plus two separate mathematical source TUs compile and
execute through the existing generated candidate route. This does not authorize
production integration, imported objects, opaque mathematical bodies or arbitrary
multi-file support.

## Exact stack and final evidence

Research branch `research/authenticated-multi-tu-v1` began at accepted H1
`486d3e5d74e17fbf305c73fca3e3760ef7da59b6`, then fast-forwarded to its normal
H1/private-output-fact composition `8fbdc61258173c46f199f9142092fc1bbec6cb04`.
No production files, product builds or release artifacts were changed by this
lane. The completed research branch was subsequently authorized for normal push;
that is not a product merge or an installed capability. Reused H1 builds were
frozen at that compiled source;
their later worktree head `75c78f586801a108ec3cc49b4003b83c34a9cb26` was report-only.

Final research C++ source SHA256:
`147ce53727563e4129b3f091a04bdd4a1c26d55d2067697fba0ada0d8b022353`.

| Local research profile | Driver SHA256 | Actual result |
| --- | --- | --- |
| Release dependencies, OpenBLAS ON, research compiler `-O0` | `d0e68f949fbd739e2fdb5cbeb625e8ecd0f823fb440b9937dad0e2ba63ffb4ee` | 14/14 cases, 16 commands; executable 37 checks |
| Debug/O1 ASan+UBSan dependencies and instrumented research compiler, OpenBLAS OFF | `b78453fa03cf9f274c5cc157588c9fe72cd5ea73d42d40a52e8644a69b2f096c` | 14/14 cases, 16 commands; executable 37 checks |

The final [Release raw record](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/release.json)
is byte-identical to local `build-multi-tu-evidence-09/evidence.json`, SHA256
`34bb5fc9474fae59b48643c2eb26cd4aacac2c530f22b8f0d933ad73c17b95d1`.
The final [ASan/UBSan raw record](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/asan-ubsan.json)
is byte-identical to `build-multi-tu-evidence-asan-05/evidence.json`, SHA256
`c4e257c8d53cd7671b50d14f7a252463f6fcaf4b69fc88d258d110c49167fa5e`.
Separate [Release build](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/build-release.json)
and [sanitized build](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/build-asan-ubsan.json)
records retain compiler commands and before/after input hashes. Those are compiler
build identities, not timing results. No benchmark was run.

## Authority and execution

[The research driver](../../../compiler/experiments/authenticated_multi_tu_v1/driver.cpp)
reuses existing frozen host admission, immutable include/query closure,
unoptimized Clang LLVM capture, exact region/helper issuance, `sameAbi`, artifact
ownership and no-clobber ELF publication. It directly includes internal `.cpp`
implementations to avoid copying their algorithms; that research shortcut is not
a proposed installed API or production factoring.

Each region retains its own authenticated Program/source-site table and paired
helper. An ordinary host TU is separately captured/replayed; it is not admitted as
mathematics. A clean declaration witness is compiler-emitted from canonical
header bytes and already sealed parameter kinds, frozen and compiled through the
same host machinery. Foreign canonical declarations are compared to this
declaration counterpart using unchanged `sameAbi`; the compiler does not strip
attributes until a comparison passes. Producer-definition checks remain separate.

Before cross-TU LLVM linking or LLVM optimization, the complete source set must
agree on target/layout, canonical region types, declaration promises and issued
symbol ownership. Source definitions cannot compete with an issued region, even
when weak, COMDAT or unused inline; foreign uses of retired source-only Value
helpers are refused. One actual main and its global symbol owner are required.
Only then are existing per-region checked thunks/helpers compiled and ordinary
LLVM linking performed. The final native program uses the exact private
candidate/runtime artifacts, not an interpreter or user-supplied object substitute.

The positive uses independently compiled `main.cpp`, `first.mdsl` and
`second.mdsl` with ordinary shared declarations. Main's common fixture body is in
a normal included header; the two mathematical definitions remain different
physical source files and compiler invocations. Its 37 checks cover noncommuting
GEMMs, a cross-TU ordinary host utility, owning observations after host writes,
and a later shape-failing invocation with earlier observations intact. No global
transaction or rollback across separate Result calls is claimed.

## Concrete falsifiers and corrections

The 13 final refusals are: packed Result/Storage ABI; direct pure and const
promises; asm-label and literal extern-C alternate spellings of an issued region;
weak-any and linkonce-ODR/COMDAT definitions; an unused inline definition;
foreign retired-Value-helper address use; duplicate main; an asm main substitute;
missing main; and a cross-TU Value helper whose body is unavailable. Actual weak
linkage `4`/no COMDAT and linkonce-ODR linkage `3`/COMDAT were checked, not inferred
from source spelling. Every final refusal returned exactly `1` before validation
or link markers, with no output or sanitizer/crash marker. Five classifier unit
tests reject wrong exits including 134/139 after a matching diagnostic, wrong
phases, artifacts and sanitizer/crash text. Historical looser runner outcomes are
not retroactively labeled as this stricter validation.

Two particularly important counterexamples explain why a C++ linker is not the
semantic owner:

1. The same canonical header under `#pragma pack(1)` changes the caller ABI
   without changing the C++ region symbol. Physical header identity alone is
   insufficient; the compiled declaration witness rejects the actual alignment
   and layout mismatch before linking.
2. An alternate C++ name with `asm("issued_mangled_name")` and `pure` passed the
   first prototype's ABI check and published an executable. The
   [preserved failed oracle](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/asm-pure-counterexample.json)
   records exit 0 and publication, not a successful rejection. `sameAbi` is not
   an effect-summary comparator. The correction separately binds source spelling
   and symbol, rejects unadmitted source attributes, and compares complete foreign
   LLVM function attributes with the clean declaration witness. The final source
   cases refuse both asm and extern-C alternate names. Clang documents that an
   asm label controls object naming in its
   [pinned mangling implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/lib/AST/Mangle.cpp).

An earlier direct declaration-versus-definition ABI comparison also rejected a
valid program: Clang adds sret `noalias` and definition-side target facts not
necessarily present on a declaration. The compiler-owned declaration counterpart
addresses that distinction without weakening `sameAbi`. An initial research
error also destroyed retained LLVM modules after their context; ordering the
context before all owning Units corrected that lifetime. Neither initial failure
is evidence against the language's region semantics.

## Sanitizer boundary: failure retained, no suppression

Initial instrumented prototype builds genuinely failed before the first region
admission completed. The [raw allocator failure](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/allocator-counterexample.json)
shows `memcpy -> SmallVector<pair<TypeID,void*>>::operator= -> Dialect::addType ->
BuiltinDialect::initialize -> MLIRContext -> admitHost`. An instrumented weak
allocator slow path poisoned a slab then upstream's unsanitized fast path reused
it. Matching `-O1`, using the existing out-of-line TU anchor, and changing from
CRTP to upstream's dynamic visitor were individually insufficient.

Disassembly located the remaining instantiation in this experiment's inline
`CXXRecordDecl::getDefinition -> Redeclarable<TagDecl> -> LazyGenerationalUpdatePtr
::makeValue -> BumpPtrAllocator`. The correction uses the out-of-line
`TagDecl::getDefinition`, then checked CXXRecordDecl casting and the same physical
header FileID. The [before/intermediate/after symbols](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/allocator-symbols.txt)
show the poisoning calls disappear, matching the already-passing product driver.
No sanitizer option, source admission, declaration predicate or runtime check was
disabled. Final ASan+UBSan executes the entire 14-case suite, including actual
generated source execution. This is compatibility with the recorded prebuilt
tuple, not a claim that all upstream LLVM/MLIR is instrumented. Relevant upstream
owners are the [dynamic visitor](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/include/clang/AST/DynamicRecursiveASTVisitor.h)
and [TagDecl definition query](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/include/clang/AST/Decl.h).

During root's independent H1 integration rebuild, an old research executable
also refused the changed candidate DSO by its compiled artifact identity. That
run was an artifact-mismatch refusal, not a semantic regression or a successful
multi-TU execution. Final runs used the subsequently frozen coherent build.

## Recommendation and remaining bounds

Use one invocation owning all source TUs and their transitive frozen contexts,
plus whole-program public/private symbol ownership and canonical foreign
declaration validation. Keep source-visible Value mathematics in admitted H1
bodies; separate TUs initially compose host-visible Result/Storage boundaries.
This is genuine separate source compilation, not merely renaming headers as
modules. Normal [LLVM linking/LTO](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/docs/LinkTimeOptimization.rst)
can consume the resulting checked LLVM; it does not invent mathematical source
authority or authorize incompatible call promises.

The experiment requires every host TU to include canonical region headers;
header-free utility TUs remain an explicit expressiveness check, not proved
support. It accepts 2..8 source TUs, one selected region per mathematical TU and
ordinary unannotated foreign prototypes. No arbitrary C++ ODR checker, external
object/module importer, independently sealed semantic interface/body format,
opaque cross-TU Value expansion, whole-region optimization, performance claim,
Windows/GPU route, installer gate or hosted product CI is established. The
smallest product follow-up should remove unnecessary host-only restrictions and
factor internal APIs normally, then integrate these ownership/ABI tests; it need
not invent an independent module format first.

## Separate ordinary-host expressiveness supplement

After core research commit `7e094b88f6d1594b624a4a25606fe0ac58d0c0cc`, the
integration owner requested two additional falsifiers. The research driver,
existing fixtures and all product code/artifacts remained unchanged. Separate
[fixtures and runner](../../../compiler/experiments/authenticated_multi_tu_v1/host_expressiveness.py)
produced these same results in both recorded profiles:

| Additional source set | Actual observation |
| --- | --- |
| Main + two region TUs + ordinary `host_helper(int)` TU, with canonical header included | Executed 11 checks, including ordinary utility result and both GEMMs |
| Same utility body with no region-header include | Checked exit 1 before program link: `experimental canonical header physical FileID or exact bytes disagree`; no output or sanitizer/crash marker |
| Main + two region TUs + two ordinary helper TUs defining the same `static local_value()` spelling, returning 31 and 47 | Executed 12 checks; both results remain distinct, plus both GEMMs |

The [Release record](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/host-expressiveness-release.json),
SHA256 `112634614c8adf55c8f4486ed54fb8d05d353452aa1ea19c44d127b444ba4d93`,
and [ASan/UBSan record](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/host-expressiveness-asan-ubsan.json),
SHA256 `f427df559e37eb4e39bf2f04ab0d7ed0487bf130ee379e64c3b667c803877477`,
each preserve five commands with exact exits `0,0,1,0,0`. Their status is
`OBSERVATIONS_MATCHED`, not three successful source programs. Local originals are
`build-multi-tu-host-release-01/evidence.json` and
`build-multi-tu-host-asan-01/evidence.json`. The
[ELF symbol readback](../../../compiler/experiments/authenticated_multi_tu_v1/evidence/host-internal-symbols.txt)
also shows two distinct local functions after normal LLVM linker renaming.

Thus header-free utility support is a real, unnecessary restriction of this
prototype, not a reason to require semantic-module imports. A product design
should capture/replay every ordinary TU without force-including a region header;
authenticate canonical types and declaration promises whenever that TU actually
declares an issued region; and always retain the program-wide LLVM symbol-owner
checks. Merely skipping all interface checks when a header is absent would not be
a valid correction. Independently sealed opaque mathematical modules remain a
different, unimplemented authority boundary.
