# Closed reassociate GEMM issuer: bounded implementation

Starting canonical main: `c04215e787832b35bbd6a85a51d9285b053181bc` (PR #62).
Branch: `mdslc/generated-reassociate-gemm-v1`. Date: 2026-09-09.

**PROVEN WITHIN A BOUNDED CONTRACT:** a separately identified compiler-owned
`reassociate_f32` GEMM traverses the real semantic / Linalg / One-Shot /
Transform / Vector / LLVM path and executes on the available Linux x64 CPU.
This branch alone does **not** connect that leaf to authenticated source/runtime
selection; that is an independently reviewed integration boundary. No default
strict schedule, public numerical syntax, provider policy or source authority
was changed. No timing experiment was performed on these new objects.

## Decision and ownership

The source already permits per-GEMM reassociation and FMA, while retaining the
f32 operation boundary, IEEE special values, signed zero and caller FP contract.
The new primitive sets that actual profile in the frontend-neutral Program and
has its own semantic identity/digest. It is not a relaxed flag added to the
strict primitive or a member of `StrictGemmScheduleV1`.

The common exact topology, overwrite/DPS and One-Shot no-allocation checks are
reused through private profile-specific builders. The old strict entry points
and both strict schedules keep their existing checks. Shared result containers
have profile-neutral names with the old internal names retained as aliases.

The fixed upstream schedule tiles `[4,8,1]`, peels N then M, vectorizes static
full tiles, lowers contractions to outer products, lowers remaining scalar
Linalg and hoists the serial C transfers. Full-tile cells carry a 4x8 f32 value
through increasing K; three scalar remainder regions retain separate f32
multiply/add. There is no K padding, packing, distribution, storage allocation,
copy, cross-operation transformation, publication or provider selection here.
Tile/vector sizes belong only to this realization, not source semantics.

Exact coherent LLVM/Clang/MLIR 21.1.8 remains pinned. The isolated object target
is **baseline x86-64 plus AVX2/FMA**, not the research object's full x86-64-v3.
No blanket fast-math flag is permitted. Normal object compilation uses O3,
`-march=x86-64 -mavx2 -mfma -ffp-contract=off`; actual ASan objects use O1 with
`sanitize_address` carried on emitted functions before Clang instrumentation.
LLVM target attributes independently record the same CPU/features.

Only aligned-C leaf argument 15 receives the already-established private-output
NoAlias fact. A/B may be identical or partially overlap; wrapper descriptors and
allocated-pointer fields do not acquire noalias or alignment assumptions.
The existing descriptor checker is shared only after each issuer's own exact
symbol/body/intrinsic checks, and grants no arbitrary-module authority.

The runtime integrator must check the source profile and physical AVX2/FMA plus
OS XMM/YMM before invocation, keep empty/zero-K handling local, and retain all
shape/capacity, allocation, FP restoration and ordered-retirement obligations.
The legacy public capability query runs unrelated packed-kernel selftests and
must not be mistaken for a pure hardware probe. Direct existing platform-v2
discovery is the intended dependency; it must preserve errno. Candidate archive
and DSO definitions include the unchanged owning capability/platform sources so
installed archive consumers do not require an uninstalled dependency archive.
No provider adapter or shared provider-policy state is duplicated.

## What the mechanical checks prove

The scheduled verifier independently checks the private dynamic rank-2 identity
MemRef envelope, positive-zero full overwrite before contraction, complete
increasing scalar K, bounded full-M/N loops, register carry, exact A/B vector
indices/shapes/permutations, C read-before-K/write-after-K, and scalar-tail
multiply/add/C update dataflow. Its operation inventory excludes allocation,
copy, unknown calls and distributed/control effects.

Exact whole-module MLIR equivalence against a fresh fixed upstream derivation
then binds every remainder bound/index, SSA connection, property and attribute.
Location is deliberately ignored. An exact LLVM replay additionally binds the
lowered body, wrapper, allowed fmuladd intrinsic, attributes, target and private
data-pointer fact; only the parser's diagnostic module identifier is ignored.
The intrinsic has four vector calls, with three scalar tail multiply/add pairs.

**Replay is pipeline identity/drift checking, not an independent proof of LLVM
or the upstream transforms.** The trusted pinned implementation and the checked
serial/fresh-C preconditions remain explicit. In particular, the upstream
hoister warns against distributed MemRef loops. Here its C transfers remain
inside nonempty full-M/N loops after initialization. K=0 can access only already
initialized C, not A/B; zero M/N executes no vector transfer. No arbitrary
possibly-empty loop hoisting is admitted.

Official pinned interfaces:

- [Linalg transfer hoisting, including the distribution warning](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td#L2235-L2272).
- [Vector FMA lowering to LLVM FMulAdd](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp#L1105-L1131).
- [x86-64-v3 requires more than AVX2/FMA](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/TargetParser/X86TargetParser.cpp#L61-L68).

FMulAdd permits fusion; it is not a universal hardware-fusion promise. The exact
local objects demonstrate four YMM fused accumulators and scalar tails. Their
full/tail rounding difference is source-authorized under reassociate, but fails
the strict-result counter-oracle. A different permitted realization is not a
source-profile violation.

## Validation and retained failures

| Executed scope | Outcome |
| --- | --- |
| Release generated-primitive regression, excluding source-driver tests | 23/23 passed, 0.76 s |
| Debug/O1 ASan+UBSan compiler and affected strict/reassociate scope | 13/13 passed, 1.08 s |
| Existing strict issuer mutation suite | 92 checks, zero failures |
| New issuer semantic/schedule/LLVM mutation suite | 53 checks, zero failures |
| Ordinary and actual generated-ASan numerical oracles | Each 1,417 cases / 109,632 checks, zero failures |
| Corruption and strict-result negative controls | Required exit 1 and chosen-realization diagnostic, both profiles |
| Undersized A and independently undersized full-tile B | Exact ASan exit 1; generated frame #0; READ 4 and READ 32 respectively |
| Closed CLI | Positive control plus six unsupported invocations refused with exit 2 and no artifact |
| Normal/ASan object contracts | Exact entries, bounded imports, full-tile YMM FMA and separate scalar tails |

Both local builds disable OpenBLAS. No source-driver, full standalone, provider,
installed-package or hosted CI pass is claimed by these focused runs. Hardware
absence returns CTest skip 77 for actual leaf execution, never fabricated pass;
no such skips occurred here. Suite durations are not kernel benchmarks.

Mutations cover wrong profile/symbol, swapped operands, negative-zero seed,
finite-only/blanket flags, changed K bounds/step, well-typed per-K zero reset,
late initialization, wrong tile/tail indices, input destination, extra copies /
allocations / effects, target widening, calling convention and descriptor/data
alias confusion. Sanitization of the C++ caller alone is not accepted.

Retained integration failures: the first actual Transform invocation refused
the unregistered Linalg TilingInterface; explicit upstream external-model
registration corrected the dependency. Initial compile attempts exposed pinned
operation-accessor/pass names, and linking required `MLIRVectorToLLVMPass`
separately from `MLIRVectorToLLVM`. The new object inspector initially rejected
the pinned ASan pass's common `___asan_globals_registered` bookkeeping symbol;
only that exact symbol is now admitted in the sanitizer lane. None was recorded
as a successful gate or solved by loosening mathematical verification.

Earlier masked-K4 research failed to establish register carry; strict cache
tiling lost the measured comparisons. Neither alternative is adopted. Their
negative records remain on the research branches. Full-v3 behind only AVX2/FMA
gating and replacing the strict default are also rejected.

## Exact artifacts and remaining gate

Implementation source SHA256:

- register issuer: `26166ceb5e7a7169568284cf2c4ff6e7de90656d516830cde2a2a5358e251fee`;
- shared strict/stage source: `50f381cd4d3c7f1411dec0256e13407c1f743b39485352a741613424d6abb073`.

Release build artifacts under `build-reassociate-release/lib/regions/`:

- `reassociate-normal.o`: `22285cebb5ea339caf3b65170a41b316de1110091b7dac7f5ca8f0580d71cfd3`;
- `reassociate-asan.o`: `d030880dfd31d7c0a43bcb0143535b513e7664936f6a51058a745c58bc33dbd2`;
- `reassociate-normal.ll`: `9e2d43480535d1fe53d07d678b1f44278bc51657b594cbd087d9da87eb71e6cb`;
- semantic MLIR: `b1e8863cdb0df4847d1ebd482ca0935aca129ee7308f471ae9bf4bcb1410f687`;
- scheduled MLIR: `1be828c4bd1989a823b6ecbab478df6eeeda4bfc14205f1c16bb49c618091a3e`.

Ignored logs `build-reassociate-release/primitive-regression-ctest.log` and
`build-reassociate-asan/focused-ctest.log` have hashes respectively
`5a6e25a69b7b6217e425b585b744e6d4f0bd09e65da8c6689aef3361acff3c13` and
`e9e6cbfabc0579730ecd1754df0b1c925821db25b541bc01a764585c50459ae0`.

Next gate is the separately implemented closed registry/source-policy connection
and its independent mixed-profile failure-prefix, capability, owning-storage,
FP, package and hosted regressions. This record does not pre-accept that work.
