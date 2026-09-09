# Preserving the private generated GEMM output-storage fact

2026-09-09. This is a lower-boundary fact-preservation correction, not a new
source alias restriction, schedule, allocation policy or execution route.

## Evidence and decision

Starting engineering checkpoint: `0b2c21f126288501fe27afea9320ec9fce392dcc`
(PR #60). Research instrument:
[`9a0b47644cefde004f15f111fe02244bcaa3088d`](https://github.com/onlyxItachi/MatcoreDSL/tree/9a0b47644cefde004f15f111fe02244bcaa3088d/compiler/experiments/gemm_alias_contract_v1).

**OBSERVED:** in the exact coherent MLIR/LLVM 21.1.8 dynamic rank-2 MemRef
path, annotating the MemRef argument with `llvm.noalias` marks the C wrapper's
descriptor pointer, not the flattened leaf's aligned matrix-data pointer.
These are different objects. The current runtime already creates fresh private
output storage while both immutable input values remain alive
(`SessionAbiV2::gemm` / `allocate` in `closed_host_v1.cpp`). External Storage
descriptors still all MAY alias; input values may be identical.

**ARCHITECTURAL IMPLICATION:** preserve that established private output-data
fact at the actual LLVM data-pointer boundary. Do not invent a higher-level
noalias policy, infer disjoint input values, or rewrite upstream lowering.
The adapter invokes LLVM's ordinary parameter-attribute API only after the
closed issuer's existing semantic, structured, buffer, schedule and LLVM-body
checks. This is not an arbitrary LLVM importer or independent body verifier.

`preserveStrictGemmOutputStorageV1` requires the exact two-function private
module and pinned C ABI: three rank-2 descriptor expansions (21 leaf arguments),
three wrapper pointers, address space zero, i64 offsets/extents/strides, void
returns, nonvariadic definitions and the expected calling convention. Unexpected
parameter attributes fail closed. Only aligned-C argument 15 receives NoAlias;
all guards precede mutation. No allocated pointer, wrapper descriptor or input
argument is annotated. No nonnull, alignment, dereferenceability, numerical,
publication or lifetime fact is added.

The manifest explicitly records caller-private output isolation, the leaf data
binding and permitted input/input aliasing. Existing semantic contract identity
is unchanged; derived LLVM and artifact hashes naturally change. Scalar and
row-contiguous issuers both preserve the same fact; scalar remains the default.

Upstream references: pinned
[FuncToLLVM](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp)
and [LLVM parameter attributes](https://releases.llvm.org/21.1.0/docs/LangRef.html#parameter-attributes).
Argument NoAlias constrains accesses to modified locations: read-only A/B
overlap alone is not a negative test for the absence of input attributes. Exact
structural inspection therefore accompanies execution oracles. The separate
22 tuple's `memref.distinct_objects` does not establish availability in pinned
21 or justify asserting that all three data buffers are pairwise distinct.

## Mechanical defenses

- Issued scalar, row and sanitized LLVM must contain exactly one NoAlias,
  attached to leaf aligned-C, and no added nonnull precondition.
- Wrong signatures, arities, address spaces, calling conventions, declarations,
  missing/extra functions and preexisting attributes must reject without
  changing LLVM attributes. Signature checks are not a theorem about arbitrary
  bodies; only the existing compiler-owned derivation calls this adapter.
- Independent input-overlap oracle source is
  `a94541d4a7fee14a17152218f327b0c54e00d81a` (cherry-picked separately).
  Exact new ordinary and generated-ASan objects each passed 72 cases / 86,898
  checks. Identical and forward/backward partially overlapping inputs, rectangles,
  zero extents/reduction, strict FP, descriptor/input immutability and fresh-C
  canaries are covered. Deliberate result corruption produces exactly 54 failures;
  CTest checks that diagnostic and status, not merely a nonzero exit.
- The 225-case / 563,296-check source oracle from rejected cache research
  `902a4ef3bba940460d75cc474d879d56ffa42355` survives as a schedule-independent
  regression. It tests reduction boundaries, FMA-sensitive values, IEEE tails,
  dependent GEMMs, old values, alias-sensitive late reads and publication/failure
  prefixes. Retaining this test does not adopt the losing cache schedule.
- Existing source, object, actual wide SIMD ASan negative, private-runtime,
  provider, ownership and package tests remain independent gates.

## Validation record

Implementation: `94eab4c3d9c53a6b075727d505f09f8f4b7b5740`.
Clean full validation checkpoint:
`c932bcfbc754fb2479c376581b61ddab075e7ad1`, incorporating canonical documentation
merge `3eaf5e4e58d92e3898a0ccdc802c5e89758c79f8` (PR #61). No source/report edits
occurred during either complete validation run.

| Scope | Actual outcome |
| --- | --- |
| Focused connected Release | 16/16 passed, 9.70 s |
| Candidate signature/body/mutation unit | 92 checks, zero failures |
| Full Release, OpenBLAS ON and required | 143/143 passed, 388.52 s |
| Debug/O1 ASan+UBSan, OpenBLAS OFF | Full build; affected 88/88 passed, 287.01 s |

Both builds use `matcore-mlir`, row-contiguous and coherent 21.1.8. Release also
enables inspection-only vector readiness. Sanitizer execution uses the hosted
ASan/UBSan environment and the exact union of both committed selection regexes;
88 registrations were checked before execution. Dynamic linkage confirms no
OpenBLAS in the disabled build. Elapsed suite times are not benchmarks.

Release driver SHA256:
`2c79847729d5d58e5766a8c6a78137e5a57dc89abc17900684cd71938ba61f23`;
ASan driver: `200b290b1f9e607b709a40755cd515e28218b0d58460d912febef6379c65dd5e`.
Ignored logs `build-alias-release/full-ctest.log` and
`build-alias-asan/focused-ctest.log` have SHA256 respectively
`870e5c690232cf8a73d4db8d264f66a9bb0ea65e04cecf7f8dd98b4fe4ef3eff` and
`9f269c14bb72b05787b59c5a8228decc09e998476c43ed0ec89fb3cc307aec87`.
Hosted CI and normal integration are separate, still-pending gates in PR #62.

Frozen generated object SHA256 identities for the independent overlap runs:

- ordinary: `5951cec81bd48b3a72b25583e41e3422ee58e309c127cd0faaf27398c4321a50`;
- actual generated ASan: `6b921c68482055476d5cd055aa61e900ab1b45650eb24c9f82db05e1fec4d3f8`.

Independent production-source review accepted the unchanged caller contract and
the adapter's deliberately narrow proof scope. It also found the initial test
link mixed static LLVMAsmParser/Core/Support with shared LLVM; actual launch
aborted with duplicate `allow-incomplete-ir` registration (exit 134). The test
now links the coherent shared LLVM target, following existing integrated tests.
An earlier test-only C++ conditional pointer-type compile error was also fixed.
Neither failure was a successful validation gate or a production LLVM workaround.

## Limits and next evidence

There is no measured performance claim for this integration yet. Smaller
research object text and fewer possible alias checks are not timing evidence.
Compare authenticated old/new actual source and primitive paths before choosing
any performance policy. No provider crossover, public ABI, new target, input
noalias, zero-copy, buffer reuse or general transformed-region execution follows.
The private leaf's caller-storage contract and exact pinned descriptor lowering
remain preconditions; a future ABI or bufferization change must re-establish the
binding rather than bypass these guards.
