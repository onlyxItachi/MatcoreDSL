# Authenticated host function ABI thunk — bounded implementation

Base: `85d871fc6bb2bc762de1bb5030eb7a2c46854996`; branch
`mdslc/authenticated-abi-thunk-v1`. This is an internal linker transformation,
not a frontend, raw-IR execution interface, or semantic-authority issuer.

## Boundary

The caller supplies two freshly compiler-produced, unoptimized LLVM modules in
one context, exact authenticated original/helper symbols, and an optional exact
list of sealed Value-helper symbols to retire. The utility checks both modules,
target triple/data layout, calling convention, scalar/aggregate function shape,
all return/parameter attributes, recursively typed `sret`/`byval` layouts,
target features and the required no-throw boundary. It checks structural layout
before and after linking, with exact non-type attributes and exact final LLVM
function type. Typed memory attributes require recursive layout equivalence,
not identified LLVM struct pointer identity (see the correction below).

The result owns an isolated context: in-memory bitcode copies prevent LLVM
Linker's context-owned type renaming from mutating either input module. It
preserves the original function identity, linkage, visibility, ordinary calls,
function-pointer uses, debug subprogram and CFI identities. Only the body becomes
a forwarding call. Old body-derived function and direct-call memory facts are
removed; return/parameter ABI facts remain checked. Unrelated strong global
overrides fail; standard weak ODR and LLVM's recognized appending globals use
the ordinary linker. This is not a private C++ type linker.

Value helpers are removed only when their entire remaining use graph is internal
to the explicit retirement set. Escaped pointers, globals, aliases and other
host calls reject removal. Shape helpers are untouched. Shared helpers must be
retired after every region using them has been replaced, not prematurely.

## Falsification and validation

LLVM/Clang **21.1.8**, Linux x86-64. Standalone test configuration:

```sh
cmake -S compiler/tests/abi_thunk -B build-abi-thunk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm
cmake --build build-abi-thunk --target matcore_authenticated_host_thunk_test -- -j2
ctest --test-dir build-abi-thunk -R codegen.authenticated_host_thunk_v1 --output-on-failure -j1
```

Observed: **51 Release checks, zero failures**. The real Clang fixture returns a
nontrivial `Result` containing `std::vector` through `sret`, accepts aligned
`Storage` through `byval`, and accepts `Shape` as a scalarized argument. Both a
namespaced direct call and a volatile function-pointer call execute the new
implementation. Original host RAII/destructor counts, source file/line and
same-line `__builtin_COLUMN()` remain identical to the original executable.

Negative cases include different target/layout/CC/features/noexcept/alignment,
missing `sret`, nested record size and same-size field permutations, `byval`
field permutations, real `_GLIBCXX_DEBUG` nested-Result ABI mismatch despite
identical facade-header bytes, cross-context inputs, escaped/nonhelper Value
uses, strong-global collisions and missing/duplicate retirement bindings.
Positive controls cover original aliases, multiple CFI identities, repeated
linking, input immutability and populated result move-assignment lifetime.

Observed: **55 Debug ASan+UBSan checks, zero failures**. Compiler and tests use
`-fsanitize=address,undefined -fno-omit-frame-pointer`; child C++ compilation and
final LLVM-IR compilation receive these flags too. A deliberately out-of-bounds
helper load, reached through the actual thunked executable, must produce
`AddressSanitizer: stack-buffer-overflow`. Therefore this lane demonstrates
actual helper-load instrumentation, not merely an instrumented test runner.
Strict leak/halt/string/initialization-order and UBSan halt options also pass.

The first const-input clone implementation failed its input-immutability test:
normal LLVM linking renamed types shared with the helper input. Owned context
isolation corrects that counterexample. Textual source-body replacement is not
used: preserving source line numbers alone cannot preserve same-line column
observations.

## Correction: identified memory type identity is not ABI identity

The first connected public-region ASan run was safely rejected after linking:
the original and generated functions had the same recursive ABI and LLVM
function type, but distinct identified `sret` type objects. A new independent
fixture reproduces that failure using actual unmodified Clang C++ input modules:
an opaque owning Result with nested status, owner pointer and source-location
records. No synthetic IR or source/record substitution is needed. Simpler
opaque records and the original vector-owning fixture did not expose it.

Against the original utility, the new ASan fixture produced **58 checks, one
failure**, exactly `host/helper exact ABI differs after LLVM type remapping`.
This falsifies a linker-uniquing assumption, not the source/host architecture.
The corrected comparison removes only typed Attribute object-identity equality.
It still checks complete recursive structure before and after linking, all
non-type attributes exactly, target triple/data layout/calling convention, and
exact final LLVM FunctionType identity. The last restriction is necessary for
ordinary SSA argument/return forwarding; no first-class aggregate conversion is
generated. Original host ABI attributes stay on the original function; the new
call carries the callee's own typed ABI attributes.

LLVM 21's [structure-type contract](https://releases.llvm.org/21.1.0/docs/LangRef.html#structure-types)
distinguishes identified identity from literal structural uniquing; its
[parameter contract](https://releases.llvm.org/21.1.0/docs/LangRef.html#parameter-attributes)
defines `sret`'s type as its in-memory type and requires call/callee ABI attributes
to agree. The exact [llvmorg-21.1.8 linker implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Linker/IRMover.cpp)
performs isomorphic type mapping and opportunistic name/type reuse; Matcore must
not assume that every compatible typed attribute becomes pointer-identical.

After correction: **57 Release checks, zero failures (1/1, 0.79 seconds)**;
**62 Debug ASan+UBSan checks, zero failures (1/1, 1.86 seconds)**. The latter
explicitly asserts that the two isomorphic sret types remain pointer-distinct,
then compiles and executes the real thunk, checks its result through the original
function pointer and verifies one owning destructor. The preexisting direct and
function-pointer controls, actual instrumented out-of-bounds negative control,
wrong layouts (including same-size permutations), alignment, target, convention,
noexcept and genuine debug-stdlib-layout refusals all remain in the same test.
The sanitizer options remain the strict set described above. This focused
utility result does not replace the integration owner's actual full driver test.

## Limits and ownership

ABI equivalence is not mathematical equivalence or declaration authenticity.
Equal opaque-pointer layouts cannot distinguish semantically swapped fields.
Canonical C++ record/configuration authentication and exact generated-helper
authority remain caller/frontend responsibilities. The deliberately unsafe
same-ABI sanitizer control makes this distinction explicit.

Inputs must precede LLVM optimization: clearing stale facts cannot restore host
effects already optimized away. Clang constant evaluation likewise cannot be
repaired by a thunk; frontend admission must exclude constexpr region execution.
Varargs, special target types, coroutine/GC/naked/returns-twice functions,
address-taken blocks and `inalloca`/`preallocated` arguments fail closed. No
Windows ABI, cross-target executable, installed frontend, performance, fusion
or new semantic execution authority is claimed by this utility test.

LLVM owns ABI lowering, type remapping and machine lowering. Relevant upstream
contracts: [parameter attributes](https://llvm.org/docs/LangRef.html#parameter-attributes),
[function attributes](https://llvm.org/docs/LangRef.html#function-attributes).
API behavior is compiled against the exact installed 21.1.8 headers/libraries.

Next connection belongs to the authenticated source consumer: frozen original
C++ -> original LLVM module, sealed generated orchestration -> helper module,
this checked thunk -> ordinary final object/link. It must not accept user IR or
treat a matching ABI as a new execution-authority issuer. Full repository/hosted
validation remains the integration owner's responsibility; no hosted CI is
claimed here.
