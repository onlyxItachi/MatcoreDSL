# Experimental Result frontend: implementation evidence

Base: `e78747a3f1042ce4e7cf2d065cf5e4e22df04d7f`.
Branch: `mdslc/experimental-region-frontend-v1`.
Architecture: [owning-result frontend](../EXPERIMENTAL_REGION_FRONTEND_V1.md).

## Scope and review provenance

The value/transaction advocate authored the public facade, frontend admission
adaptation, owning adapter transfer and primary fixtures. The independently
assigned CPU execution reviewer did not author those changes. It authored the
three `independent_*` Result fixtures and challenged retirement, ownership,
actual allocation failure, mixed host standard-library modes and source-member
attributes. This is agent engineering review, not a forged GitHub approval.

Opaque ownership received independent **ACCEPT within the bounded host
contract** after the initial direct-STL design was replaced. Source admission
also underwent independent adversarial testing; the discovered member-attribute
counterexample is retained below. Broader driver/ABI-thunk/package integration
and its hosted checks belong to the separate consuming boundary.

## Mechanical checks

The focused surface contains 16 CTest tests covering:

- existing private semantic graph and hermetic/physical-host admission;
- existing host adapter contract, production behavior, independent adversarial
  tests, actual global-new failure sweep and production injection refusal;
- new public admission, canonical headers, helper/name/source witnesses and
  frozen replay;
- ordinary C++ compile success followed by intrinsic link failure without
  authenticated code generation, even with the runtime linked;
- owning Result transfer, both legacy/new header orders, mixed-standard-library
  producer/consumer ownership, independent retirement and independent OOM.

Direct test counters on the opaque implementation:

| Surface | Exact outcome |
|---|---|
| Public frontend fixture suite | 97 checks, zero failures |
| Original adapter contract | 137 checks, zero failures |
| Original independent adapter | 217 checks, zero failures |
| Actual allocator sweep | 132 checks, zero failures |
| Owning Result primary fixture | 37 checks, zero failures |
| Independent retirement | 16 checks, zero failures |
| Independent mixed-STL/OOM/concurrency | 16,068 checks, zero failures |
| Additional mixed-STL consumer | 8,000 concurrent handle cycles, zero failures |

The 19 ordinary hostile-program cases explicitly require `syntax_valid=true`
before checking that semantic admission rejects them. Ambient member/enum
attributes and out-of-line definitions have equivalent well-formedness checks.
The two constant-evaluation entry tests separately assert C++20 type rejection.
The tests do not relabel
syntax/type failures as semantic closure proofs.

Both Release and whole-component Debug ASan+UBSan run the complete 16-test
surface with zero failures. Build dependencies include in-process Clang,
frontend admission, semantic verifier and runtime; no local sanitizer exclusion
was introduced. The deliberate self-move test suppresses that warning only.

Reproduction from repository root:

```sh
cmake -S compiler -B build-region -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-21 \
  -DCMAKE_CXX_COMPILER=clang++-21 -DMDSLC_ENABLE_MATCORE_MLIR=ON \
  -DMLIR_DIR=/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang -DMDSLC_ENABLE_OPENBLAS=OFF
```

Build the `matcore_experimental_region_*`, `matcore_closed_host_*`,
`matcore_closed_region_{admission,host,semantic}_tests` targets with `-j2`.
The ASan configuration additionally uses Debug and these flags:

```text
CMAKE_CXX_FLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
CMAKE_C_FLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
CMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined
```

The exact test command is:

```sh
ctest --test-dir build-region \
  -R 'experimental_region|runtime.closed_host|frontend.closed_region|mlir.closed_region_semantics' \
  --output-on-failure -j1
env DEBUGINFOD_URLS= ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-region-asan \
  -R 'experimental_region|runtime.closed_host|frontend.closed_region|mlir.closed_region_semantics' \
  --output-on-failure -j1
```

Compiler/toolchain: Clang and LLVM 21.1.8, exact MLIR 21.1.8, Linux x64.
The distro MLIR 22.1.2 was not silently substituted; the exact package is at the
local toolchain path above. This record does not claim Windows execution or a
toolchain migration. Existing unrelated generated-MLIR unused-parameter build
warnings remain visible.

## Preserved counterexample provenance

The initial public Result directly owned a standard-library vector. Passing
tests did not defend cross-TU `_GLIBCXX_DEBUG` layout differences, so that
representation was rejected before integration. The independent mixed-STL
fixture now compiles only the consumer with `_GLIBCXX_DEBUG=1` and
`_GLIBCXX_USE_CXX11_ABI=0`; producer/runtime are normal. No recursive STL semantic
authenticator or configuration blacklist was added.

The independent member-only attribute attack compiled ordinary C++21 and
passed pre-fix admission plus both immutable replays:

```text
syntax_valid=1 admitted=1 error=
```

The attacking source uses C++11 `[[noreturn]]` in `#pragma clang attribute`, not
the unsupported GNU spelling. Preserved local pre-fix evidence:

```text
/tmp/mdslc-result-owning-review.d9I9Zs/member_attributes.cpp
/tmp/mdslc-result-owning-review.d9I9Zs/admission_attack_before
/tmp/mdslc-result-owning-review.d9I9Zs/admission-before-member-auth-asan.a
archive SHA256 a86f020316b0a0f9fab211c003a67b9f5abcf4e48f379343eb692ab79a38342b
```

Those temporary paths preserve this run's negative evidence, not a dependency
required by the durable tests. The equivalent hostile source is permanently in
`compiler/tests/experimental_region/admission_test.cpp`.

The same independent reviewer subsequently reproduced valid-source admission
of `Observation::~Observation` and the Result move constructor defined outside
the canonical header, then host completions of the forward-declared Session
and ObservationBlock. These four counterexamples are now durable negatives too.
All owned function/member/tag redeclaration chains, bodies and attributes are
checked; opaque forward declarations cannot gain a foreign host definition.

Reviewed opaque runtime identities:

```text
region.h 2717d27b8787f976e2b5b4af2645b38d1409ce0c91181f4e7413e2a00265b45e
region_storage.h acfcb2427fe9efc564fdda78723a9454d313a3ad4d2bb854a25de1f27355ab74
closed_host_v1.cpp b51ac9b89be9255709b868287e5c4d71e97bc561945659760d6bf9791356bed0
closed_host_v1.h 2d888bb3ccc62180da9524c9063a79be5183ddac5c493912a0339725077b0a6d
```

Remaining conditions: valid caller memory/capacity/lifetime/access, no conflicting
resource access, thread-confined Session, independent-handle ownership sharing
only, trusted conforming allocator/deallocator and runtime, exact admitted
numerical contract. No sandbox, publication atomicity, whole-region rollback,
device execution, new planner threshold or performance claim is implied.
