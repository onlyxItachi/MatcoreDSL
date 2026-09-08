# Bounded CPU compiler foundation checkpoint

2026-09-08. **PROVEN WITHIN A BOUNDED CONTRACT:** MDSLC has a real, installed,
synchronous Linux x86-64 source-to-generated-executable compiler with explicit
mathematical values, host resources, numerical permissions and observable/failure
semantics. The foundational model is established; all optimizations and targets
are not. No surviving adversarial counterexample requires another language or
resource-model redesign before the next optimization can be implemented.

## Canonical identity and acceptance

| Identity | Exact checkpoint |
| --- | --- |
| Canonical input | `199a841e5049130fffe5a3067c75ba1016613903` (operator PR #57) |
| Reviewed implementation | `b0e7ac9ac67c3c314a44e5db108be390485404eb` |
| Normal engineering merge | `8e0cf0f39833966227dee44d4367b99927a7ae03`, [PR #58](https://github.com/onlyxItachi/MatcoreDSL/pull/58) |
| Merge parents | Canonical input, then reviewed implementation above |
| Candidate and merge tree | `2a226770a17e7bba9b446a7f39adc5bbc428f2e0` |
| Merge time | `2026-09-08T16:24:05Z` |

All **19/19 exact-head hosted checks succeeded before normal merge**, including
duplicate push/PR checks, not 19 independent configurations. The
[acceptance gate](https://github.com/onlyxItachi/MatcoreDSL/pull/58#issuecomment-5588403701)
records local evidence and independent agent reviews; no separate human GitHub
approval is claimed. The merge tree was checked against the reviewed head and
the clean canonical checkout fast-forwarded normally. No force update, research
branch wholesale adoption, unrelated cleanup or release tag was performed.
[#56](https://github.com/onlyxItachi/MatcoreDSL/issues/56) is closed by this bounded
ownership proof. Documentation-only commits may follow this engineering merge.

## Why this architecture won

C++ remains the interoperability host; explicitly admitted regions own a closed
mathematical/effect vocabulary. Clang handles C++ parsing, Sema and original host
codegen. Source-only `Value` is not an executable proxy; pure helpers are expanded
at compilation. A frontend-neutral semantic Program contains values/resources,
dynamic dimensions, operation numerics and ordered check/effect frontiers. There
is no runtime AST/graph interpreter and no second general C++ interpreter.

The semantic model separates immutable tensor contents from allocation identity.
Descriptors remain all MAY-alias. Current reads snapshot at their actual frontier,
not region entry; private candidate output is issued only after completion and
FP checks. Host publication and owning observation retire in semantic order;
first failure stops later effects without erasing earlier ones. Allocation
opportunities may change with realization; required source checks may not vanish.

| Alternative | Evidence-backed disposition |
| --- | --- |
| Arbitrary C++ as optimization-region semantics | Rejected for closed regions; ordinary C++ around authenticated mutating calls remains the compatibility host. Hidden calls, cleanup, conversions and observations do not belong in the closed mathematical region. |
| Standalone language now | Not selected: closed C++ admission demonstrated useful math, pure helpers and bounded shape control without executable proxies or an interpreter. A future frontend can target the same semantic model. This does not prove standalone MDSL can never become useful. |
| Permanently pointer-backed values / entry snapshots | Rejected: alias publication changes an old value / a late read misses earlier publication. |
| Direct fallible candidate output / whole-region rollback | Rejected: partial candidate writes escape / successful earlier effects are incorrectly erased. |
| Source-text body replacement | Rejected: later host source columns change. Preserve original Clang host LLVM and replace only the authenticated entry with a checked ABI thunk. |
| Weak helper/cleanup selection; whole-archive alone | Rejected by actual host substitution and allocation-failure probes. Isolate issued LLVM definitions and bind the candidate DSO implementation locally. |
| Ambient linker/provider discovery | Rejected by a zero-exit ASCII-emitting fake linker and a nonstandard provider link failure. Bind configured artifacts and recheck before output publication. |
| Imported or transformed inspection IR as authority | Rejected: exact source pairing is not a derivation for transformed code. |

The [resource decision](FOUNDATION_RESOURCE_DECISION_V1.md),
[host-linkage decision](decisions/REGION_HOST_LINKAGE_V1.md),
[integration history](agent-reports/region-driver-integration-v1.md) and
[independent architecture audit](agent-reports/cpu-foundation-architecture-audit-v1.md)
retain the original hypotheses, probes, failed checkpoints and scope qualifications.

## Actual ownership and optimizer boundary

| Owner | Responsibility at this checkpoint |
| --- | --- |
| Source / Matcore | Closed admission, immutable mathematical values, resource epochs, numerical permissions, required checks, allowed effect/failure traces and authenticated derivations. |
| Clang | C++ semantics outside regions, resolved declarations, source context and frozen original host LLVM. |
| MLIR | Registered semantic witness and upstream structured/buffer lowering of the issued strict GEMM primitive. |
| LLVM/backend | Helper isolation, machine lowering, instruction selection, register allocation and target scheduling. |
| Runtime | Dynamic legality/capability checks, candidate completion, concrete storage, ordered publication/observation and shared provider policy. |
| Provider | Its authenticated admitted mathematical implementation under explicit adapter/numerical contracts; not source authority or a universal semantics theorem. |

**Whole-region MLIR remains an exact, untransformed paired witness.** Static
orchestration is emitted from the sealed Program after pairing verification.
Only the strict GEMM primitive currently passes through verified Linalg,
One-Shot Bufferize and LLVM into generated computation. A future transformed
region needs a separate legality-preserving derivation; relaxing the exact
pair verifier is not authorization. No target tile/width/packing threshold was
encoded in mathematical semantics.

Compile-time authority fixes the admitted program, numerical candidate set,
toolchain and implementation artifacts. Runtime resolves concrete shapes,
capacity/access, implementation availability, provider conformance and failure.
Automatic currently selects the linked strict generated candidate, not a measured
cost optimum. Native strict and numerically legal existing-reference/OpenBLAS
choices coexist; forced illegality fails without silent fallback.

## Validation at the reviewed implementation

| Scope | Actual outcome |
| --- | --- |
| Local Release, OpenBLAS ON | **133/133 passed**, 341.66 s, explicitly reconfigured at clean `b0e7ac9`. |
| Local affected Debug ASan/UBSan, provider OFF | **80/80 passed**, 195.97 s, same clean-head discipline. Generated-load/store negative controls actually exercise instrumentation. |
| Hosted MLIR Release, provider ON / OFF | 133 / 132 registered, zero failures; selected PR jobs each explicitly skipped one hardware-dependent AVX-512 test. |
| Hosted MLIR Debug, provider OFF | 131 registered, zero failures; PR job explicitly skipped one AVX-512 test. |
| Hosted affected ASan/UBSan / focused TSan | **80/80 / 4/4 passed**. TSan scope is existing runtime concurrency, not the entire compiler. |
| Hosted legacy Windows | Release 49 / Debug 32 registered with one AVX-512 skip each; focused ASan **1/1 passed**. Not generated-region Windows evidence. |
| Tests-disabled production / fresh install | Final-head full incremental build passed; nonstandard-provider install ran Result 37, candidates 151358, Value 85 plus 32000 ownership cycles, mixed handles, ABI refusals and installed source execution. |

Hosted [native](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34248533071),
[Windows](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34248533088),
[legacy CI](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34248533042)
and hygiene are separately recorded by the gate. Test durations are not benchmarks.
The earlier `c3ffafb` local 133/80 successes did **not** justify integration: hosted
CI failed. Corrected provider dependency, CMake oracle identity and profile-aware
archive falsifiers passed the fresh gates; no failed run was relabeled successful.

Independent source matrices recorded 62,891 Release and 53,103 sanitizer assertion
evaluations with noncommuting rectangular lhs/rhs carries, seven alias layouts,
late-read counteroracles, all-arena sentinels, dynamic/zero shapes, exact failure
prefixes, persistent owning observations and ordinary object links. Their earlier
driver checkpoint provenance is retained in the [reports](agent-reports/driver-storage-conformance-v1.md),
not silently relabeled as an independent final-head rerun. The committed harness
also ran in the final full CTest gates.

Forged source/declarations/context, hidden host effects, helper bodies, export
collisions, ABI drift, OOM cleanup, partial provider mutation, FP corruption,
incorrect variant/thread reports and missing/wrong artifacts have permanent
refusal tests. Source/build/system-provider-hidden installed executions are
[separately recorded](agent-reports/region-driver-integration-v1.md#hosted-falsifiers-at-c3ffafb-not-an-accepted-merge-head)
with their actual artifact chronology. They do not authenticate arbitrary future
dynamic loading. A comma in a provider build-dependency prefix remains an observed
upstream-generated linker limitation; installed product paths with spaces/comma
were exercised, not generalized into arbitrary-path support.

## Deliberate limits and remaining work

Experimental execution is **Linux x86-64, coherent Clang/LLVM/MLIR 21.1.8**.
22.1.8 remains separate internal compatibility evidence. Existing mutating GEMM,
IR v1 capture, CPU runtime/provider and standalone Windows compatibility survive;
the separate Python/JIT compatibility lane is tested on Linux. The
older recovered/contraction/buffer/vector/two-GEMM inspection specimens did not
acquire execution authority. Public experimental syntax/ABI and private contracts
are unfrozen. The installed compiler retains configured toolchain dependencies;
ordinary installed consumers do not inherit LLVM/MLIR development interfaces.

No generated-region Windows, GPU/NPU, asynchronous residency/transfer/export contract,
general public tensor/view surface, transformed region execution, fusion,
automatic reuse or zero-copy claim. Host storage requires valid initialized
objects, truthful lifetime/capacity/access and race freedom. Allocation and
library loading remain trusted. Normal-return publication is not crash safety,
concurrent atomicity, a sandbox or protection against arbitrary interposed hooks.

**No performance comparison exists for the new generated region path.** The
[historical native audit](../performance/cpu/cpu-performance-deep-audit-v1.md)
at `509ef2b775e501783dfa7f2c4aa21e91f513bd6a` recorded 711 cases: 583 accepted and 128
expected legality refusals. It is not current generated-path performance.
[#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15) remains partial/open:
authenticated same-checkpoint forward/reverse envelope, scaling and full planner
regret remain missing; cooperative packed-B preparation remains production-dormant.
[#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

**Exactly one next boundary: one effect-preserving two-GEMM schedule with one
bounded private-storage reuse.** Prove a new authenticated derived lowering while
preserving live old values, alias-sensitive late reads, required checks,
per-GEMM f32 rounding and permitted effect/failure prefixes. In particular, retain
the selected host adapter's **stronger normal-return publication guarantee**,
not merely the generic dialect's weaker partial-write allowance. Successful
earlier publications remain; a failed publication itself leaves its destination
unchanged. This is optimization implementation, not a new language decision.
It has not been started by this checkpoint. Additional operations, targets,
schedules/cost models, performance validation and tooling are subsequent work.
