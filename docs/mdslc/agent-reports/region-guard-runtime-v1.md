# Region guard runtime/FP oracle lane

Base: `63d642d9e4eb4183acf58a4391e1c668e734b9f0`.
Branch: `agent/guard-ledger-oracles-v1`.
Scope: one new test source, `compiler/tests/runtime/region_guard_oracle_test.cpp`.
No runtime, public header, planner, provider, frontend or CMake changes.

## What the test establishes

The test calls the existing nonexecuting
`matcore_runtime_plan_gemm_f32_v1` API, whose private `validate_gemm_v0` is shared
with the established one-shot execution route. It does not implement another
descriptor validator or run provider conformance. Every plan case checks the
status and unchanged tensor canaries; failed queries also preserve the report.
Assertions use explicit checks rather than `assert`, so Release/NDEBUG cannot
remove them.

Test labels cross-reference the inspection ledger predicate vocabulary:

| Predicate | Existing-oracle cases |
|---|---|
| `data_nonnull` | Output, lhs and rhs data pointers independently null |
| `positive_dimensions` | Both axes of all three descriptors, zero and negative |
| `contraction_dimension_equal` | Incompatible reduction dimensions |
| `output_rows_equal`, `output_columns_equal` | Incompatible destination dimensions |
| `pointer_alignment_required` | Misaligned live-object byte addresses, never dereferenced as float |
| `byte_range_representable` | Element multiplication, byte multiplication and address-plus-extent overflow |
| `output_input_no_overlap` | Output/input equal or partially overlapping intervals rejected; equal and partially overlapping inputs accepted |
| `floating_environment_compatible` | Existing pure decoder and actual current-thread inspector, separately from plan success |

Invalid C-ABI rank/dtype/layout/space/mutability/policy records exercise existing
defensive checks. Their labels reference `source_tensor_contract`,
`source_layout_contract`, `source_host_designation`,
`source_access_designation` and `source_policy_intent`, without claiming such
invalid records are reachable through the authenticated source adapter.
Descriptor-header/reserved checks and policy-before-descriptor/output-before-lhs
first-failure ordering are also covered.

## Deliberate nonproofs

An explicit `backing_capacity_sufficient` counterexample supplies real live
one-float arrays with 64-byte-aligned starts and claims 4x4 descriptors. The
claimed intervals are representable and nonoverlapping, so the plan oracle
accepts despite insufficient actual typed array capacity. **That fixture is
never passed to an execution API.** No forged pointer is dereferenced; even the
overflow cases contain real live array addresses and only false descriptor
extents. The test generates no out-of-allocation C++ pointer arithmetic.

The other caller obligations (`descriptor_object_valid`,
`backing_host_accessible`, `backing_lifetime_valid`,
`backing_access_permitted`, `no_conflicting_concurrent_access`) are not tested
by invoking undefined behavior or represented as discharged.

The plan API does not check FP control state and does not reproduce the current
one-shot resource/provider planner v2. A Linux x86-64 scoped FTZ experiment
therefore expects the current FP inspector to reject while the nonexecuting
plan query succeeds. The original MXCSR is restored exactly. The sole call to
an execution entry point supplies a null output descriptor, requiring immediate
rejection before tensor reads, provider discovery or GEMM execution. Previous
milestone tests already cover actual two-call partial completion; this lane
does not duplicate those executors or fake provider failure.

The pure FP decoder tests reject FTZ, DAZ, each non-nearest SIMD/x87 rounding
mode and each unmasked exception. They also prove that sticky status flags and
x87 precision modes are not invented additional guards. Actual FTZ manipulation
is Linux x86-64 only; other builds explicitly report that subtest skipped while
retaining the portable synthetic decoder and descriptor cases.

Provider eligibility, resources, numerical conformance, state restoration and
synchronous completion remain dispatch/provider responsibilities. In
particular, `cpu_openblas.cpp` calls SGEMM before its final state checks, so an
external-provider failure can occur after output mutation. These tests do not
claim universal failure atomicity or rollback.

## Validation

The new source passed:

```sh
/usr/bin/clang++-21 -std=c++20 -Wall -Wextra -Wpedantic -Werror -DNDEBUG \
  -Icompiler/include -Icompiler/lib/platform -fsyntax-only \
  compiler/tests/runtime/region_guard_oracle_test.cpp
git diff --check
```

The integration owner also performed a narrow linked check with Clang 21,
`-O2 -DNDEBUG -Wall -Wextra -Wpedantic -Werror`, using the existing prior-region
Release/OpenBLAS-off runtime/platform artifacts at `187e`. Before running it,
the owner verified that those runtime/platform source trees have no diff to
canonical `63d642d`. The new oracle reported **219 checks, zero failures; no
GEMM executed**. This is a linked oracle result against unchanged existing
runtime/platform code, not a fresh integrated-CMake or full regression result.

Fresh integration and regression results remain the integration owner's next
validation step. The test requires C++20 and links existing `MatcoreDSL::Runtime` and
`MatcoreDSL::PlatformV1` targets. It requires no MLIR dependency and no new
production dependency.

Independent read-only review found no C++ validity or FP restoration blocker.
Its precision suggestions were incorporated: the canary assertion claims only
unchanged elements (the no-read property comes from the audited plan API), and
report snapshots use `memcpy` to avoid aggregate-padding copy assumptions.
