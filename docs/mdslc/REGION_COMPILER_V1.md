# Experimental C++ region compiler

`mdslc-region` compiles an authenticated mathematical region inside ordinary
C++ into a native CPU executable. The current implementation supports dense
row-major rank-2 `float` GEMM, immutable logical values, and ordered host-storage
effects. It is opt-in and experimental: **the source API, ABI and private
compiler contracts are not frozen**. See [CURRENT_STATE](CURRENT_STATE.md) for
the latest canonical engineering checkpoint, rather than treating this guide
as a release tag.

## Build and install

Use native Linux x86-64, CMake 3.24 or newer, Ninja, a C++20 standard library,
RapidJSON headers, GNU binutils, and coherent **Clang/LLVM/MLIR 21.1.8** development
packages including shared `clang-cpp`. The installed compiler still needs its
configured toolchain and shared libraries; this is not a self-contained binary
distribution. Matching compiler-rt is needed for sanitizer builds/tests.
Cross compilation and the separate 22.1.8 compatibility experiment are not
supported by this execution path.

From the repository root, substitute the actual package/toolchain paths:

```sh
cmake -S compiler -B build-regions -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-21 -DCMAKE_CXX_COMPILER=clang++-21 \
  -DLLVM_DIR=/path/to/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/path/to/llvm-21/lib/cmake/clang \
  -DMLIR_DIR=/path/to/mlir-21.1.8/lib/cmake/mlir \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_MATCORE_MLIR=ON \
  -DMDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON \
  -DMDSLC_ENABLE_OPENBLAS=OFF -DBUILD_TESTING=OFF
cmake --build build-regions -j2
cmake --install build-regions --prefix /path/to/package
```

The feature defaults to `OFF`; production support does not depend on enabling
tests. OpenBLAS is optional. Use an authenticated supported OpenBLAS build and
the repository's provider configuration if that candidate is required; merely
finding a library does not prove its numerical contract.

## Compile and run a real program

The complete [two-GEMM example](../../compiler/examples/experimental/two_gemm.mdsl)
includes ordinary C++ callers, a function-pointer call and checked failure:

```sh
/path/to/package/bin/mdslc-region --version
/path/to/package/bin/mdslc-region \
  compiler/examples/experimental/two_gemm.mdsl \
  --region pipeline --candidate generated-strict -o ./two-gemm
./two-gemm
./two-gemm fail
```

The successful output is `22 28 49 64 -> 120 156 49 64`. The second invocation
deliberately supplies insufficient capacity to a later read and prints
`checked failure; first publication and observation retained`. The output path
must not already exist; failed compilation does not publish an output file.

Include `<matcore/region.h>`. A region is a named free function annotated with
`MATCORE_REGION`, taking named `Storage`/`Shape` parameters by value and returning
`Result` with `noexcept`. End it with `return mdsl::complete();`. Ordinary Clang
can parse this valid C++, but the mathematical intrinsics have no runtime
fallback: ordinary linking is not a replacement for authenticated compilation.
Select the function with `--region qualified::name`.

For real programs spanning files, the [multi-source driver contract](MULTI_SOURCE_PROGRAMS_V1.md)
documents `--program`, `--host` and per-source `--region` selection. One invocation
authenticates and links 2–8 original translation units, preserving each region's
effects and numerical policy. This is ordinary host-program composition, not
opaque mathematical imports or cross-region optimization; mathematical helper
bodies must still be source-visible, typically in included headers.

## Values, storage and ordered effects

`Storage` describes external host memory: pointer, rows, columns, capacity in
**float elements**, and access (`read_only` by default; `read_write` for an output).
It neither allocates memory nor proves disjointness. The caller supplies valid
initialized float objects, truthful capacity/lifetime/access and freedom from
conflicting concurrent accesses. Different descriptors may overlap.

`Value` is a source-only immutable mathematical value, not a C++ tensor
container, allocation, expression-template proxy or runtime handle. Use it in
the admitted region or pure mathematical helpers, not host pointer arithmetic,
`sizeof`-based shape calculations or a hand-written execution protocol.

| Source operation | Meaning |
| --- | --- |
| `read(S, rows, columns)` | Read immutable contents at this point; requested shape must match the descriptor. An earlier value does not change when storage is later overwritten. |
| `gemm(a, b, numerics)` | Produce a new logical matrix product; it does not mutate a destination. Inner dimensions must agree. |
| `rows(v)`, `cols(v)` | Obtain logical dimensions for admitted shape logic. |
| `publish(v, S)` | Ordered replacement of a matching writable destination. A failed publication leaves its own destination unchanged on a defined normal return. |
| `observe(S)` | Capture an owning immutable snapshot of storage at this point, retrievable from the returned `Result`. It is not a host callback or file export. |
| `complete()` | Finish the invocation successfully only if all required earlier operations succeeded. |

The first checked failure stops later region effects; successful earlier
publications and observations remain. Checks are not hoisted to entry merely
because a later operation will fail, nor deleted merely because a result is
unused. A late read through an alias sees earlier publication; an older logical
value keeps its previous contents. Untaken shape branches do not access their
resources. Empty dimensions are supported with checked representability; a
nonempty zero-reduction product contains positive zero.

Check `result.ok()`/`operator bool`, then `error()`, `message()` and
`failure_location()` as needed. Publication/observation counts describe the
completed effect prefix. `result.observation(i)` returns an owning read-only
handle that survives Result destruction and later storage mutation. Its data
pointer remains valid while an owning handle lives. Moving Result does not
allocate; numeric frontier IDs are diagnostics, not source scheduling controls.

Currently reads and observations take snapshots and candidates compute into
private storage. These are conservative implementations, **not** a language
requirement to materialize every intermediate forever. There is no whole-region
rollback, concurrent atomic publication, crash recovery or fallible device/export
transaction guarantee. The existing mutating `gemm(out(C), ...)` overload and
`mdslc++` compatibility route retain their separate semantics.

## Numerical permissions and candidate choice

Each GEMM explicitly chooses `Numerics::strict_f32` or `reassociate_f32`.
Strict means increasing-reduction-order separate f32 multiply/add, without FMA.
Reassociation permits per-operation reduction reordering/FMA, not reassociation
across GEMMs or removal of their f32 result boundaries. NaN payload bits are not
promised. Computation uses nearest-even rounding, gradual underflow and masked
machine exceptions, restoring caller FP controls and status on normal return.

| `--candidate` | Current implementation and legality |
| --- | --- |
| `automatic` (default) | Selects the linked generated-strict candidate in this package; no benchmark-derived cost model or crossover threshold. |
| `generated-strict` | Compiler-owned strict GEMM lowered through MLIR/LLVM; legal for both profiles. |
| `generated-reassociate` | Separately issued MLIR/LLVM GEMM with full-tile FMA and strict tails; requires each executed GEMM's `reassociate_f32` permission and directly discovered AVX2/FMA hardware plus OS-enabled vector state. |
| `native-strict` | Matcore strict scalar implementation; legal for both profiles. |
| `existing-native` | Existing Runtime's forced reference implementation; requires `reassociate_f32`. It is not the legacy planner's fastest-choice mode. |
| `openblas` | Authenticated linked provider; requires `reassociate_f32`, availability and provider conformance checks. |

A forced unavailable/incompatible candidate returns checked failure; it does
not silently fall back or change numerical permissions. Provider probes are
distinct from the requested GEMM, and their passing examples are not a universal
provider theorem. See the [candidate contract](agent-reports/closed-candidate-coexistence-v1.md).

Forced numerical/capability checks occur at each executed GEMM, including empty
output and zero reduction, before its allocation or computation. A later strict
GEMM under `generated-reassociate` fails at that GEMM without erasing earlier
publication/observation effects; the compiler does not reject the entire region
in advance. Unchosen generated-reassociate code performs no feature discovery.
Its leaf receives fresh private C, not public storage; the existing complete FP
restoration and normal-return publication guarantee are unchanged. This explicit
choice does not change `automatic`, promise a fastest choice, or establish BLAS
parity. CPU instructions are enabled only for the isolated generated object.

## Compiler and deployment boundaries

The whole-region Matcore MLIR module is presently an **exact untransformed paired
witness** of admitted semantics. The compiler emits ordered orchestration from
the sealed semantic graph; it does not execute an AST/IR interpreter. The strict
GEMM primitive separately passes through verified Linalg, bufferization and LLVM
lowering to actual generated machine code. This is not whole-region MLIR
transformation, fusion, tiling, vectorization or automatic buffer-reuse support.

The original C++ host is compiled without source-text replacement. Its sealed
region body becomes an ABI-checked call to compiler-owned orchestration. Private
helper definitions are isolated before LLVM linking; the candidate DSO binds its
implementation locally. The driver binds its configured compiler, linker and
runtime/provider artifacts, and rejects conflicting host definitions. Ordinary
host C++ remains subject to these bounded driver/ownership restrictions.

Only bounded include/macro options belong after `--`, for example
`-- -I/path/to/includes -DAPP_SIZE=16`. This is not a general Clang flag-forwarding
interface; arbitrary linker flags, imported LLVM/MLIR and caller-selected
sanitizer profiles are rejected. `-c` emits a checked relocatable object, **not**
authorization for an arbitrary later manual link. Use the driver's executable
mode for its complete link contract; do not substitute the private archive.

Keep the installed toolchain and Runtime/candidate/provider libraries coherent.
Rebuild/reinstall together when they change. Compile-time identity checks do not
authenticate future dynamic loading or make an executable self-contained.
Trusted library loading and conforming allocation/deallocation that preserve FP
state remain preconditions; arbitrary interposed hooks are not sandboxed.
Each synchronous invocation owns a thread-confined internal session; separate
invocations and owning observations may be used across threads with ordinary
C++ synchronization and no conflicting storage access.

Arbitrary host effects, exceptions, indirect calls and cleanup are rejected
inside regions. Pure Value/Shape helpers and bounded shape-if/else are supported
under the [admission grammar](EXPERIMENTAL_REGION_FRONTEND_V1.md). Pure helpers
may be defined in ordinary included headers, with transitive source dependencies
frozen and rechecked by the compiler. All helper bodies must remain visible to
Clang and satisfy the same closed grammar; an external ABI declaration or object
is not a mathematical definition. Physical file identities and the full helper
call chain are retained without changing public caller-side failure attribution.
General loops,
strided views, public tensor containers, accelerator/device residency, and
generated Windows execution are not provided here. Existing supported platforms
retain their separate compatibility routes. No performance, zero-copy or Native
BLAS Parity claim follows from successful compilation or execution.

Detailed rationale: [value/resource/failure contract](FOUNDATION_RESOURCE_DECISION_V1.md),
[host linkage decision](decisions/REGION_HOST_LINKAGE_V1.md),
[independent ownership review](agent-reports/private-driver-ownership-independent-v1.md),
and [bounded architecture audit](agent-reports/cpu-foundation-architecture-audit-v1.md).
