# Experimental owning-result region frontend

This is a bounded compiler-admission and host ownership checkpoint, based on
`e78747a3f1042ce4e7cf2d065cf5e4e22df04d7f`. It does not by itself add a driver
command, installed compiler, LLVM body replacement, or a new generated kernel.
Those consumers must authenticate the sealed entry and prove their own ABI and
execution connection. No public source syntax or ABI stability is promised.

## Chosen source boundary

```cpp
#include <matcore/region.h>
using namespace matcore::mdsl;

MATCORE_REGION
Result multiply(Storage A, Storage B, Storage C,
                Shape m, Shape k, Shape n) noexcept {
  auto a = read(A, m, k);
  auto b = read(B, k, n);
  auto c = gemm(a, b, Numerics::strict_f32);
  publish(c, C);
  observe(C);
  return complete();
}
```

The function has an ordinary C++ signature and owning checked result. Callers
need no `Session`, numeric frontier, `invoke<T>` protocol, global status, TLS,
or lambda/template interpreter. Ordinary Clang accepts the source, but linking
it without authenticated compiler handling fails at the undefined intrinsics,
even when the runtime library is linked. There is no executable proxy fallback.

`Value` is a trivial source-only type, not an allocation, tensor container, or
alias of the runtime's owning value. `Storage` is an external dense row-major
rank-2 `f32` descriptor with dimensions, capacity and access. Distinct descriptors
do not prove physical disjointness. Existing `gemm(out(C), A, B, policy)` remains
a separate mutating overload with its existing semantics. Both public headers
co-include in either order.

Admission preserves the existing closed mathematical grammar: named immutable
bindings, explicit read/GEMM/publication/observation, dynamic dimensions and
shape obligations, bounded shape-if with an explicit else, and pure reusable
Value/Shape helpers (including bounded concrete template instantiations).
Nested helper calls in intrinsic arguments remain unsupported: name the result
first. Helpers are statically expanded by existing admission, not interpreted.

The only Result cleanup admitted inside the closed body is the exact final
`return complete();`: canonical direct callee, zero arguments, canonical Result
type and its canonical temporary destructor, with no additional cleanup objects.
Early returns, comma expressions, other conversions/construction, arbitrary
destructors, indirect callees, callbacks, volatile/pointer effects and exceptions
remain rejected. Ordinary caller argument evaluation and caller-side Result
lifetime remain ordinary C++; they are not swept into the mathematical region.

## Authority and host linkage

The compiler supplies both expected physical header paths. Each must resolve to
the parsed FileID and exact bytes embedded in the compiler build. A source-side
copy, shadow header, altered declaration or configured replacement bytes cannot
issue authority. Owned declaration macro expansions are rejected. The one
ergonomic `MATCORE_REGION` expansion must directly originate in the canonical
header; a copied/redefined/nested macro is not sufficient. A direct literal
canonical annotation is also accepted experimentally.

Ambient attributes on owned records, including packing and ownership/ABI
attributes, fail closed. Existing expression/compound floating-point policy
checks reject inherited FP permission changes. Clang's exact resolved primitive
declarations, annotation payloads and redeclarations remain authoritative.

The existing immutable physical-host capture, bounded flags/toolchain contract,
dependency closure, preprocessing transcript and frozen replay are reused. Its
unused private fixture preinclude remains an implementation compatibility input;
public admission cannot discover or use those private intrinsics as authority.

The in-process entry seal additionally binds:

- original qualified and mangled function names and canonical signature digest;
- named/inline namespace chain and ordered named Storage/Shape parameters;
- exact body and completion source spans;
- each statically expanded source-Value helper's concrete mangled symbol and
  source body span (Shape-only helpers remain ordinary host code).

These C++-specific witnesses are absent from the frontend-neutral semantic
Program. Pairing replays frozen inputs and compares both the complete semantic
graph and the entry witness. A private compiler accessor gives read-only access
to that same snapshot; it does not authorize arbitrary rewritten bytes or LLVM.
Header declaration-only prototypes are admitted, but unowned helper definitions,
unknown linkage contexts and overloaded selected entries remain unsupported.

A later LLVM thunk consumer must prove exact host/helper ABI compatibility and
that erasing source-Value helper functions cannot affect remaining host uses.
The seal does not itself establish either condition. No textual whole-body
rewrite, dead-code-elimination assumption or weak intrinsic stub is authorized.

## Owning results without host STL coupling

The private synchronous adapter gains one-way `takeResult`. It transfers an
already-owned opaque observation block and status without allocating. It rejects
unfinished success, repeated retirement and same-Session retirement during an
active candidate. Active reentry makes the outer failure sticky and cannot steal
earlier observations or cause a later Value/publication to be issued.

The block is lazily allocated before the first observation effect. Snapshot,
block and vector growth failures all remain checked failures before that
observation retires. Earlier publication/observation prefixes remain intact.
The block is mutable only inside the thread-confined producing Session and is
immutable after transfer. Public Observation handles retain it using atomic
reference counts; independent handles may be copied/read/released across host
threads. Sharing and mutating the same handle object without synchronization is
not permitted. Session itself is still not concurrently callable.

Result and Observation expose only scalar/opaque-pointer layout. Ownership,
copy/move and destruction methods are compiled into the runtime; standard-library
containers and shared ownership implementation remain private. Observation data
is read-only and lives while an owning handle remains. This is a caller contract,
not protection against invalid `const_cast`, memory corruption or interposition.
The existing trusted allocation/deallocation, valid exclusive host memory,
normal-return publication and full per-candidate FP restoration conditions still
apply. No new concurrency-atomic publication or crash/whole-region rollback claim
is made. Static diagnostic source locations attach without allocation.

## Rejected alternatives and discovered counterexamples

1. **Void function plus `invoke<T>`** creates a second call/identity protocol and
   makes direct ordinary calls lose checked status. A named Result function does
   not require that extra mechanism.
2. **Immediate lambda-only surface** adds capture construction/destruction and
   template-body requirements before proving useful region behavior. A minimal
   lambda body is not proof that capture effects are closed. It may be future
   sugar, not the foundational ABI.
3. **Owning STL members directly in public Result** initially passed 37 owning
   tests and public admission. Nevertheless `_GLIBCXX_DEBUG` can change vector
   layout in one host TU without changing canonical header bytes or an opaque
   LLVM sret pointer signature. The implementation was replaced with opaque
   ownership instead of growing a standard-library semantic authenticator or
   a macro blacklist. The surviving mixed-configuration test actually compiles
   producer/runtime normally and the consumer with debug/old libstdc++ ABI.
4. **Canonical header bytes alone** incorrectly admitted ambient `#pragma pack`.
   Exact parsed record attribute checks now reject it. The hostile fixture stays.
5. **Checking record attributes but not members** still admitted a well-formed
   `#pragma clang attribute push([[noreturn]], apply_to=function(is_member))`
   around the header. An independent reviewer reproduced successful admission
   and immutable replay while Result members carried the foreign contract.
   Canonical record members now reject inherited attributes too. The GNU
   `__attribute__((noreturn))` pragma spelling is *not* supported by Clang 21;
   that syntax failure was not counted as a successful closure defense.
6. **Checking only original member declarations** admitted out-of-line source
   definitions of Observation destruction and Result move construction. Source
   could also complete the forward-declared friend Session or opaque block.
   Independent fixtures reproduced all four. Canonical function/member and
   record/enum redeclaration chains now remain wholly compiler-header-owned;
   source definitions cannot obtain authority through an original declaration.

Constant evaluation is not an alternate admission route. `constexpr`/`consteval`
entry functions are forbidden; the actual owning Result also makes them invalid
under the pinned C++20 source contract. Valid C++ helpers that attempt to derive
mathematical dimensions from `sizeof(Value)` or consteval placeholder evaluation
are rejected as outside the mathematical vocabulary.

## Validation and limits

Focused Release and whole-component Debug ASan+UBSan use Clang/LLVM/MLIR 21.1.8
on Linux x64. No sanitizer exemption was added. Exact commands and independent
evidence are in `agent-reports/experimental-region-frontend-review-v1.md`.

The CPU adapter's numerical algorithms, strong per-publication contract, failure
order and old native/provider route are unchanged. One extra lazy observation
block allocation is an explicit initial realization, not mathematical semantics.
No performance, zero-copy, accelerator, installed-driver, public tensor/view,
API/ABI stability or broad generated-execution claim follows from this change.

The next integration boundary is the same-signature authenticated host-codegen
consumer: original frozen C++ compilation, exact selected function/ABI binding,
checked compiler-issued orchestration, ordinary linked callers and refusal when
unselected source-only helper uses remain. It must not recover authority from
serialized MLIR or leave the original C++ intrinsics executable.
