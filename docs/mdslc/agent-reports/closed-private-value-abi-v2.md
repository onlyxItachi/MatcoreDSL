# Opaque private value ownership and helper/runtime ABI

Branch: `mdslc/closed-private-value-abi-v1`.
Integration base `de0911eebc788b2d05266efadb9d3902ebc2b7c7` normally composes
source/generated checkpoint `f8bce9f` and experimental owning-result frontend
`6dbd4b4e0853748b44312aa3f332af2b6303ac3d`. Its pre-change ten-test runtime/result
and native-registry control passed. The union retains candidate options/reports,
forced profile/availability checks, public owning observations and every existing
publication/failure rule. Candidate error codes move with the shared status
declarations into the experimental detail header. Tests gain the public include
path now required by that header; frontend implementation is not changed here.

## Decision and implementation

**Observed:** the private helper header exposed a `std::shared_ptr` member and
its inline ownership implementation. Fixing helper compiler flags does not by
itself establish compatibility with an installed runtime's standard-library
configuration. GCC's [ABI policy](https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html)
explicitly includes compiler ABI, library API and configuration; its
[debug-mode guidance](https://gcc.gnu.org/onlinedocs/libstdc++/manual/debug_mode_using.html)
limits interchange of affected container instantiations across configurations.
This is not a claim that every such macro changes `shared_ptr` specifically.

**Chosen:** private `Value` is one opaque pointer, with copy/move/destruction and
intrusive atomic retention wholly implemented in the runtime. `Session` remains
allocation-free stack state; no pimpl or new early allocation/failure frontier
is introduced. The private header no longer includes `memory` or `vector`.
Observation records move entirely into the runtime. Mathematical source `Value`
is unchanged and remains distinct from this private execution handle.

The owner allocation and optional element-vector allocation remain the same two
fallible preparation points. A temporary owned value protects private candidate
output until all guards, candidate execution and FP restoration succeed. Old
output handles survive failure; an output handle may also be both input handles.
Publication performs no new allocation. Retired public observations continue to
own immutable snapshots independently of the producing Session.

## Falsified marker-only design

An inline Session constructor referencing a new ABI marker initially passed
ordinary compatible/incompatible archive tests. A real separately compiled old
inline constructor COMDAT placed first then made an incompatible-runtime link
**succeed** by discarding the constructor containing the new marker reference.
The resulting `weak-bypass` binary was deliberately **not executed**. Its retained
SHA-256 is `8732f95d2de1a92a9e4239d907eae2dd99ba8ad591c4c91af4fd860680ae59eb`.

The correction versions actual private C++ linkage:
`SessionAbiV2`, `ValueAbiV2`, `ValueStorageAbiV2`, `ObservationAbiV2`.
`Session`/`Value` aliases preserve existing private source spellings. A mandatory
revision symbol remains checked at ordinary final link. Old constructors cannot
satisfy new class symbols. The test retains old-object-first/last order at O0/O2,
with real compatible-runtime executions before expecting incompatible links to
fail. Symbol inspection rejects unversioned private owning-record definitions.

Independent review also identified a plausible old `vector<Observation>` COMDAT
collision after its element layout changed. No runtime failure was reproduced
for that separate hypothesis. Removing the unused header-level record and
versioning runtime-only record names prevents that overlap structurally.

This is an accidental build/revision mismatch gate, **not source authentication**,
a cryptographic certificate, or protection from deliberately forged runtime
symbols, arbitrary interposition, or duplicate incompatible runtime instances.
Incompatible future private layout changes must advance class and symbol names.
Public Result/Storage layouts do not change. The canonical Result friend now
names the versioned Session through its alias; its hostile source-authentication
fixture is retargeted to the real class and still requires valid C++ before
checking refusal. The initial obsolete fixture failed precisely because it was
no longer well-formed; that failure is not counted as an authority rejection.

## Validation

Exact final runtime sources:

- Header SHA-256: `d8f3d0312c4cd71e869f32e350d0aa3baa5bbc139b7bdef948099ad6b7a3db7f`.
- Implementation SHA-256: `102d88f0d0430dfca16394b549df8824c2fe54d5beed44900f9c1d51ab6d3cf6`.

Linux x86-64, exact Clang/LLVM/MLIR 21.1.8. Builds use at most two jobs.
`build-value-release` and `build-value-asan` configure the ordinary compiler
project with native frontend/MLIR enabled and OpenBLAS disabled. The sanitizer
build uses `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` and matching
executable/shared linker flags. The existing source test instruments the actual
child orchestration/runtime, and the generated candidate retains its actual
kernel-load ASan positive/negative controls.

The bounded scope comprises all eight `runtime.closed_host.*` tests, four owning
result/mixed-configuration/retirement tests, all ten `closed_candidates.*` tests,
and both `frontend.closed_source*_execution_v1` tests: **24/24 ASan+UBSan passed,
17.96 seconds**, with strict leak/halt/string/initialization-order options. The
matching Release scope passed; adding closed admission, real-host context and
experimental admission exposed the obsolete friend fixture (26/27). After
retargeting that syntax-valid adversary, the complete expanded Release scope
passed **27/27, 36.88 seconds**, including all **97 experimental admission checks**.

The three new private-value tests cover 14 direct ownership/actual-allocation
assertions, 16,000 cross-thread private handle copy/move cycles, a separately
compiled normal producer plus `_GLIBCXX_DEBUG=1`/`_GLIBCXX_USE_CXX11_ABI=0` consumer,
and positive/negative real-artifact ABI links including stale COMDATs. Existing
real OOM sweeps, adversarial partial candidate writes, retained publication
prefixes, old snapshots, observations, source authority and forced candidate
profiles remain in the regression scope.

Independent review accepted the exact sources above. Its separately authored
test passed **85 assertions and 32,000 cross-thread owning-handle cycles** in
strict-warning Release and ASan+UBSan, including actual global-allocation failures
1 through 12 across publication, observation, same-handle GEMM and retirement.
The reviewer also independently repeated six focused tests: Release 6/6 in
0.66 seconds, ASan+UBSan 6/6 in 1.18 seconds. That independent harness is retained
as the additional `runtime.closed_host.private_value_independent_v2` test.
Its retained CMake target then passed 1/1 in each build (0.01 seconds each).
See the [independent review](private-value-abi-independent-v2.md) for exact scope.

**Actual provider execution:** separately compile the final new adapter and
`closed_candidates/independent_arithmetic_test.cpp` with generated/legacy/OpenBLAS
registry defines, its own issued generated object and the already authenticated
runtime DSO from `MatcoreDSL-wt-closed-candidate-coexistence-v1/build-candidates`
(or `build-candidates-asan`). Both Release and ASan+UBSan executed **47,638
independent arithmetic/concurrency checks, zero failures**. This reuses the
validated owning runtime/provider artifact; it is not a fresh provider rebuild.

| Artifact | SHA-256 |
| --- | --- |
| Reused Release runtime DSO | `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014` |
| Reused ASan runtime DSO | `98da5261a24ae7ca3dfbc5d2a2c70b51528da37ad81c9527565bded9474bed75` |
| Issued Release leaf | `8a5334297a2b680f2df2cea28945ce773b4b250abfa567b19f313c446b50b847` |
| Issued ASan leaf | `7c7a6986a264cde615c804badc296cac1c681c6756334148ee3dc6ee84137892` |

## Packaging handoff, not implemented here

The audit found production adapters/leaf production still registered beneath
test directories, while public headers install recursively. Product promotion
must define production runtime and leaf generation independently of
`BUILD_TESTING`, preserve one owning legacy runtime/provider instance, and make
the driver link the matching installed region runtime. Private helper compilation
must use compiler-owned flags/headers, not arbitrary host flags or macros.
No LLVM/MLIR targets should leak into ordinary package consumers. A no-tests
install/relocation/consumer execution is required before claiming this public
source route is packaged. Runtime promotion is the integration owner's separate
lane. This change adds no public tensor API, new target, fusion, scheduling policy,
performance result or BLAS-parity completion claim.
