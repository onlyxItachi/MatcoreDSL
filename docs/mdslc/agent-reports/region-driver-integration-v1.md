# Installed region compiler: integration evidence v1

2026-09-08. Integration branch: `mdslc/private-candidate-dso-v1`,
[PR #58](https://github.com/onlyxItachi/MatcoreDSL/pull/58).
Canonical reconciliation input: `199a841e5049130fffe5a3067c75ba1016613903`
([operator PR #57](https://github.com/onlyxItachi/MatcoreDSL/pull/57)); latest
preceding engineering merge: `3a0f995d522bb3809c907ee68bc50e2a820d7ea6`
([PR #55](https://github.com/onlyxItachi/MatcoreDSL/pull/55)).
No main reset, research-branch adoption, history rewrite or unrelated cleanup.

This record distinguishes implementation evidence from final integration.
The final exact-head hosted gate and normal merge identity belong to the PR
gate comment and subsequent [operator checkpoint](../CURRENT_STATE.md).
At this record's creation the PR was draft; no pending check is called passed.

## What survived

The [experimental source compiler](../REGION_COMPILER_V1.md) now connects real
named closed-region C++ to an executable and a relocatable object, preserving
the original Clang host module and substituting only a sealed, ABI-checked
region entry. Compiler-issued orchestration is not a shadow interpreter.

Authoritative admission produces frontend-neutral values, resources, shapes,
numerical permissions and ordered check/effect frontiers. Its whole-region MLIR
module is an **exact untransformed paired witness**, not a transformed program.
The strict GEMM primitive independently traverses verified Linalg, One-Shot
Bufferize and LLVM into actual generated x86-64 computation. Source orchestration
uses checked immutable-value issuance and ordered runtime effects. Future
transformed region lowering must prove a new derivation; it cannot reuse exact
source pairing as blanket execution authority.

The implementation owner is separate at each composition boundary:

- Original host exports are checked against the exact authenticated Runtime,
  private candidate and optional provider ELF export sets, not name heuristics.
- Compiler-issued helper definitions are internalized using upstream LLVM before
  linking with the host. Residual nonlocal bodies fail closed; imported runtime
  declarations remain external. Original host calls and function pointers retain
  their identities.
- An isolated candidate DSO binds its implementation locally and exports exactly
  47 reviewed entrypoints. It uses the same unchanged checked runtime source;
  canonical Runtime remains the sole Matcore OpenBLAS policy-state owner.
- The installed driver pins its compiler, linker, headers and exact build or
  installed Runtime/candidate/provider images, rechecks before no-clobber
  publication, and verifies the resulting ELF kind. Its child compiler uses a
  staging-private temporary directory, avoiding self-invalidating input metadata.

The old private archive remains an internal compatibility/test artifact, not an
equivalent hostile-host link contract. Trusted deployment/library loading,
valid caller memory/lifetimes, no conflicting concurrent storage access and a
conforming allocator remain preconditions. This is not a sandbox or a guarantee
against later library replacement or arbitrary manual linking of object output.

## Focused checkpoints and losing alternatives

| Checkpoint / finding | Disposition |
| --- | --- |
| `0d880002402f3d04de0f96a77749dbcd1419a90a` | Frozen original host, artifact/export ownership, pinned linker, actual ELF output and no-clobber publication. A PATH fake linker previously published ASCII after exit zero; now refused. |
| `2e4a59eb7fb71ab342fb863458cdbaa948c3caf6` | Isolate compiler helper LLVM definitions; ordinary weak ODR selection had substituted host return 91 for issued return 7. |
| `37301498be8de1d082769e16c807c3044fa50c11`, `103dfd2168785a09ca8c039e21e74c672c52f251` | Private DSO plus exact exports/local binding. Whole-archive cleanup was replaced by host exits 71/72 only on failure. Syntax/mangling blacklists and exports-only checks do not solve both links. |
| `e9a1d40197d5efe08fe42bd3081fdea904ed6217` | Actual driver connection to both DSOs with exact installed-image identities and explicit search paths. |
| `9f692c2527cf9ce9e6157b0103070d7b4503bd2d` | Independent `available_externally` LLVM counterexample: upstream internalization preserves the body. Reject residual bodies using `isDeclaration()`, not `isDeclarationForLinker()`. Root regression failed before correction; independent 63-check ABI suite passed afterward. This was a utility counterexample, not an observed fixed-O0 admitted-source failure. |
| `24595d2cd027d21adcff11673b6fa837562c6413` | Real source ownership tests: four generated/OOM-prefix positives, twelve object/executable export or record refusals. |
| `849082e6c9e8f7e40ffe90e75babc2705ee8acc1` | Production installation and actual installed-driver oracle, separate from archive consumer tests. |
| `75602759cae68074d3406d34f59e14620415fb41` | Independent storage fixtures integrated into CTest, installed/feature-OFF checks and hosted test-count gates. |
| `470d7096e9be461e99fcf32c0f8863e0f95c8d46` | Inherited TMPDIR in captured working directory let Clang's temporary link object invalidate input metadata. Pre-create a private staging temp directory before admission; no identity exemptions. Previously failing storage scope passed 2/2 afterward. |

Earlier preserved failures include textual source-body replacement changing
later source columns, mixed static/shared LLVM duplicating APFloat identities,
and treating identified LLVM struct pointer equality as aggregate C++ ABI
equivalence. The selected solution freezes the original module, uses one shared
LLVM, and checks aggregate ABI structurally. See the
[linkage decision](../decisions/REGION_HOST_LINKAGE_V1.md) and
[historical integration hold](../REGION_DRIVER_INTEGRATION_HOLD_V1.md).

Release weak-archive cleanup controls must reproduce exits 71/72. ASan lowering
does not emit those particular weak bodies; its archive result is only an
observation, not ownership proof. Even the optimized DSO lacks one exact old
destructor, so a passing OOM case alone is insufficient causal proof. The exact
export/local-binding checks and actual source/ABI adversaries are complementary.

## Actual local and independent validation

Coherent Clang/LLVM/MLIR 21.1.8; native Linux x86-64, Ryzen AI 9 HX 370.
Release uses authenticated OpenBLAS 0.3.32; Debug ASan+UBSan explicitly disables
OpenBLAS. Ninja builds use `-j2`, CTest `-j1`. No toolchain package was changed.

| Scope | Actual outcome |
| --- | --- |
| Full Release at clean `849082e` | **131/131 pass**, 340.68 s. |
| Affected ASan+UBSan at clean `849082e` | **78/78 pass**, 141.89 s. |
| Expanded Release at clean `470d709` | **132/133 pass**, 319.17 s: the source-hidden package test correctly refused stale CMake-captured HEAD `7560275`. Explicit reconfiguration at `470d709`, unchanged source, then that test **1/1 pass**, 12.68 s. This is not claimed as a single 133/133 invocation. |
| Expanded affected ASan+UBSan at clean `470d709` | **80/80 pass**, 188.55 s, including actual generated load/store instrumentation negative controls and source storage/ownership tests. |
| Independent runtime ownership | Release **8/8**, ASan **8/8**; 151358/121215 candidate checks, plus owning Value/Result cycles. |
| Independent corrected source/artifact/ABI | **3/3 pass**, 5.15 s; 11 source, 82 ownership, 63 ABI checks. |
| Independent final Release source storage matrix | **62891 assertion evaluations**, 286 case executions, 14 ELF checks, zero failures. |
| Independent final ASan source storage matrix | **53103 assertion evaluations**, 12 ELF checks, zero conformance or sanitizer failures. |

The independent matrix covers noncommuting rectangular lhs/rhs carried products,
seven disjoint/overlapping layouts, late reads that differ from illegally hoisted
reads, alias-plus-failure cases, exact check/publication/observation/completion
prefixes and source locations, dynamic and zero dimensions, strict versus legal
reassociation policies, owning observations after Result destruction and source
mutation, and ordinary object linking. Requested policy is not proof of chosen
implementation identity; the registry/provider suites provide that separate
evidence. Counted assertions are not performance samples or distinct operations.

Independent reports:
[ownership](private-driver-ownership-independent-v1.md),
[initial storage matrix](driver-storage-conformance-v1.md),
[expanded numerical/provider matrix](driver-storage-conformance-modes-v1.md),
[actual sanitizer matrix](driver-storage-conformance-asan-v1.md),
[conditional architecture audit](cpu-foundation-architecture-audit-v1.md).

At `470d709` the local build driver SHA-256 values were Release
`2b6692d310e336dfd6a858be614b020c3f952f26ca02b880fb88915285d3268e`
and ASan `1dbcfe5d60aa28e1d8bfdbd1e8e720bda2c45220528deaa75a8b9859cf66c35a`.
The checked adapter implementation was unchanged: `closed_host_v1.cpp`
`102d88f0d0430dfca16394b549df8824c2fe54d5beed44900f9c1d51ab6d3cf6`;
header `d8f3d0312c4cd71e869f32e350d0aa3baa5bbc139b7bdef948099ad6b7a3db7f`.

## Tests-disabled installed execution

A fresh Release/OpenBLAS ON production build at clean `849082e`, with
`BUILD_TESTING=OFF`, completed **111/111 Ninja steps**. Its prefix deliberately
contained spaces and a comma. With both the complete source worktree and build
directory hidden by bwrap, the installed driver compiled the real example;
the resulting CPU executable returned zero for both rectangular math
(`22 28 49 64 -> 120 156 49 64`) and later checked failure retaining the first
publication and observation. ELF inspection recorded both candidate and canonical
Runtime dependencies and the two correct installed search paths.

Installed image hashes: driver
`44a8202fbdb3941fb47be3b603ceb7c0db2f4b77e31a634ad5bce635ae80cf7e`,
candidate DSO `87700e2ebb20a34881e73fc1545b4d753acc9d8e0c7d439ab4eca62116e548ad`,
Runtime `bbfbbd75b6c73aa7f1cd22227cdc2ab2fa6bdc21b3dee045f9b4346e6dc1851e`.
The initial sandbox setup made Clang's ordinary temporary directory read-only;
it failed without an output. The successful pre-`470d709` run allowed `/tmp`
writes. Neither setup result is a claim of unrestricted sandbox compatibility.

## Reconciliation and remaining claims

**PROVEN WITHIN A BOUNDED CONTRACT:** the inspected architecture supports a real
source-to-generated CPU executable with frontend-neutral semantics, explicit
observable/failure order, and separately validated native/generated/provider
choices. Independent architecture review found no new foundational contradiction,
conditional on final CI, integration and truthful operator documentation.

**UNSUPPORTED:** transformed whole-region execution, fusion, automatic buffer
reuse, asynchronous/device execution, generated Windows support, public API/ABI
stability, general tensor/view types, arbitrary interposed host effects, crash
atomicity, zero-copy or a performance advantage. Existing compatibility paths
and supported Windows regression remain separate gates. Issue #15's native BLAS
parity remains partial: corpus observations and new correctness tests do not
supply its missing same-checkpoint envelope/scaling/planner-regret evidence.
Issue #20 remains a design proposal; this minimal dense host Storage descriptor
does not implement its general public tensor/view model.

**One next non-foundational boundary after integration:** a separately verified
legality-preserving transformation of closed-region orchestration. Start from the
sealed value/resource/effect semantics, retain all required checks and permitted
observable prefixes, and derive an execution-authorized transformed candidate.
This would add optimization, not decide again what the language means. It is
not implemented by this integration and must not weaken exact source pairing.
