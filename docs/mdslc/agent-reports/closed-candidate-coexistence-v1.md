# Closed CPU candidate coexistence v1

## Scope and provenance

Canonical starting point: `fd64850a0c8cb7d2c0801a64dd94d515dd714130`.
Isolated branch `mdslc/closed-candidate-coexistence-v1` incorporates host adapter
`b33a1dd` and generated primitive commits `2964e2f`, `a3548ee` by normal cherry-pick.
This lane adds a private compile-trusted registry to the new immutable-value
adapter. It does not change the existing runtime/provider C ABI or its planner.
Root integration separately owns source admission, packaging, and hosted CI.

The exact contract and test layout are in
[closed_candidates/README.md](../../../compiler/tests/closed_candidates/README.md).
Source/inspection evidence is not executable input to this registry. No public
syntax, API/ABI stability, accelerator, performance, or parity claim follows.

## Surviving ownership decisions

- **PROVEN WITHIN A BOUNDED CONTRACT:** strict native and the fixed strict MLIR21
  generated leaf coexist under one private immutable-Value adapter. New outputs
  are isolated, numerical controls normalized privately, and full caller FP state
  restored before value issuance. Every GEMM ends at f32; cross-operation
  reassociation remains forbidden.
- **OBSERVED:** the legacy runtime's reference loop starts at positive zero and
  its precise-FP compile flags permit contraction. It is reused through the real
  forced-reference C ABI, not re-instantiated as a misleading strict candidate.
- **OBSERVED:** the exact local OpenBLAS 0.3.32 dynamic Cooperlake provider passes
  the existing identity/nonfinite/thread/control gate and new closed-specific
  gate, plus the extended arithmetic/trace tests. This is not evidence for all
  OpenBLAS versions/cores or any other provider.
- **INFERRED, supported by source and execution:** the inspected provider's
  f32 reduction/FMA algorithm belongs to the reassociation-permitted profile.
  Arbitrary provider success or finite-only correctness would not establish that.
- **PROVEN WITHIN A BOUNDED CONTRACT:** forced unknown/unlinked/incompatible
  requests cannot fall back. Default Sessions retain strict native; explicit
  automatic selects linked strict generated or strict native without thresholds.

The closed-provider gate performs five fixed stack-fixture calls once per
process/adapter instance, inside the full FP scope. K2 outputs must belong to a
legal f32 reassociation/FMA expression family; K1 checks signed zero, gradual
subnormal behavior and Inf-times-zero. Probe activity is separately reported.
The gate is a falsification guard, not a universal theorem or a performance probe.
The linked compiler/runtime/provider and valid allocation/lifetime/race-free host
storage remain trusted preconditions, as for the existing execution route.

## Upstream evidence actually inspected

At the exact OpenBLAS `v0.3.32` tag:

- [SkylakeX SGEMM kernel](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.32/kernel/x86_64/sgemm_kernel_16x4_skylakex_3.c):
  explicit zeroed accumulators, f32 FMA operations and bounded reduction tails.
  These operations explain why strict no-FMA authority cannot be inherited.
- [Generic beta kernel](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.32/kernel/generic/gemm_beta.c):
  the beta-zero branch overwrites C with zero rather than multiplying old C.
- [Level-3 driver](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.32/driver/level3/level3.c):
  beta processing and blocked kernel calls. The new adapter deliberately handles
  zero shapes itself; it does not derive its language contract from BLAS returns.

The existing hosted Release workflow builds OpenBLAS **0.3.32**, archive SHA256
`f8a1138e01fddca9e4c29f9684fd570ba39dedc9ca76055e1425d5d4b1a4a766`,
`DYNAMIC_ARCH=0 TARGET=GENERIC USE_THREAD=1 ONLY_CBLAS=1` with Clang21. Hosted
execution of that different core remains a separate integration requirement.

## Falsification surfaces

- Strict bit equality including signed zero, subnormals, NaN/Inf (NaN payload
  unspecified); a concrete nonzero-FMA-versus-positive-zero discriminator.
- Reassociation/FMA membership for all K1/K2 combinations of 12 selected special
  values, plus K=3/15/16/17/255/256/257 provider-tail/nonfinite fixtures.
- Noncommuting rectangular lhs/rhs-carried GEMMs, immutable read snapshots, late
  reads after aliasing publication, and earlier publications surviving failure.
- M/N=0 no candidate invocation, K=0 positive-zero output; a huge M=INT64_MAX,
  N=K=0 case proves the known generated empty-loop limitation is bypassed.
- Full MXCSR/x87 restoration under altered caller rounding/FTZ/status flags.
- Production archive has no test-injection setter; request/report snapshots do
  not carry function pointers or become mutable registry authority.
- Independent test-only replacement of the legacy C ABI lies about numeric
  results, selected variant, thread count and FP controls. Another substitute
  passes the gate then partially overwrites private output and returns failure.
  No malformed provider result is published; completed earlier effects survive.

## Validation checkpoint

All local builds use exact LLVM/Clang/MLIR 21.1.8, Linux x86-64. Release builds
cover OpenBLAS ON and OFF. The Debug lane instruments the issuer, generated leaf,
adapter and host tests with ASan/UBSan; external OpenBLAS itself is not instrumented.
Raw LLVM IR receives explicit sanitize-address attributes. UBSan claims are
limited to compiler/host code and explicit checked arithmetic, not reconstructed
source-level instrumentation inside generated LLVM arithmetic.

Final local refresh: Release ON **15/15** (10 registry tests plus 5 affected
existing runtime/workspace/provider/ABI/FP tests), Release OFF **10/10**, Debug
ASan+UBSan **13/13** (10 registry tests plus 3 generated sanitizer/independent
execution tests). The final production registry executable completed **151,358**
checks with zero failures. Existing generated OOB negative control demonstrated
real ASan detection rather than merely successful sanitizer linkage.

Independent `borrow_publication_advocate` review passed **39/39** separately
authored guard checks in Release and ASan+UBSan. Two proposed falsifiers were
retained permanently: zero-footprint operands whose output extent overflows, and
one Value handle supplied as both operands and result. No implementation change
was needed. Code reviewed: `closed_host_v1.cpp` SHA256
`08d51142dc47692ab75b41576d91e5ed556efe1745b2918829534cc0e467acf2`,
header `c008984b78bbdce06610dcd2c5c8f5ffe5aec979698dd6cf421d14863cc74626`.

Independent `value_transaction_advocate` review passed **47,638** ASan+UBSan
checks using an independently constructed K1..4 exhaustive tree/permutation/FMA
oracle, 180 random rectangular cases, special values and broad exponents. Eight
simultaneous first-use Sessions preserved independent thread FP state, produced
correct values/reports, and exactly one invoked the closed-provider probe. The
reviewer repeated all 10 registry CTests successfully and found no source blocker.
Its detailed report and durable oracle are separate review artifacts.

### Superseding concurrency counterexample

Retaining that oracle uncovered a fresh-process Release failure despite the
initial passing independent run: the provider's supposedly local thread policy
was actually process-global. The preliminary concurrency acceptance above was
withdrawn. See the [owning-layer correction and retained negative control](openblas-shared-policy-scope-v1.md).
Registry integration requires that correction and renewed independent validation;
passing arithmetic fixtures alone did not establish concurrent provider safety.

The fix is independently reviewable commit
`437bdcc1d15373634f6cacd5439566ea6cd3d689`; registry implementation is
`78135023dfdd99bd2dd8c3323c2137f6041d64ac`. With that fix, Release ON **17/17**,
Debug ASan+UBSan ON **20/20**, and Release OFF **10/10** passed. Independent review
repeated the formerly failing original oracle in **100 fresh Release processes**
without failure. The durable independent oracle is included in this follow-up
test commit with its failure diagnostics; no mathematical assertion was relaxed.

## Deliberate limits

No arbitrary generated-artifact/plugin registration, runtime autotuning, provider
crossover policy, packed/vector/parallel candidate extension, generated region
fusion, accelerator or physical-residency behavior. New source wrappers may use
the registry only through their independently authenticated authority seam.
Windows compatibility remains unchanged; this new registry/FP/generated candidate
proof is Linux x86-64. Global release/package evidence belongs to root integration.
Issue #15 remains unresolved: these correctness tests are not its missing native/
provider benchmark envelope, scaling, regret or cooperative-packing evidence.
