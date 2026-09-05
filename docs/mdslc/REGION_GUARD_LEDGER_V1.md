# Per-call region guard and obligation ledger v1

Canonical engineering starting point: `63d642d9e4eb4183acf58a4391e1c668e734b9f0`
([PR #29](https://github.com/onlyxItachi/MatcoreDSL/pull/29)). This implements the
single next boundary from [two-GEMM admission](TWO_GEMM_REGION_V1.md), not region
execution or a new general-purpose proof framework. The existing authenticated
CPU runtime/provider route remains unchanged.

The compact operator-checkpoint maintenance rule landed separately in
`c01e3851158fe87f3ace4fd5031d432ebcc483a4`
([PR #30](https://github.com/onlyxItachi/MatcoreDSL/pull/30)), from reviewed head
`a63729e665b41efcdf0a40c21552cec85962253d`, with 19/19 hosted checks passed and
identical reviewed/merged trees. It changes documentation only.

## Meaning and authority

The previous six guard-group strings retained obligations but did not
distinguish their evidence. Each admitted call now has a bounded typed ledger,
derived from its verified semantic contract and declaration bindings. The
ledger remains inside the existing ordered inspection region. It introduces
neither executable predicates nor a source of runtime truth.

| Evidence class | What it means | What it does not establish |
| --- | --- | --- |
| Source representation only | Authenticated declarations and semantic contracts designate rank-2 f32, contiguous row-major indexing, output overwrite, numerical permissions and CPU/error policy. | Concrete pointer validity, physical host accessibility, actual FP control state or provider availability. |
| Runtime validation required | Nonnull data, positive compatible extents, required natural alignment, representable byte intervals, output/input disjointness and current FP environment remain per-call predicates. | That any predicate has run, succeeded, or may move across the previous commit. |
| Caller precondition unproven | Live initialized descriptor objects and accessible, correctly typed backing storage of sufficient capacity; readable inputs, writable output, lifetime and no conflicting concurrent accesses are required. | These properties cannot be inferred from pointer tags, descriptor identity or numerical address-range checks. |
| Dispatch/execution obligation retained | An available legal candidate must satisfy numerical/build/capability/resource requirements and complete synchronously with required provider-state preservation. | Source CPU intent selects no provider or ISA; a successful plan is not successful execution. |

The call's validation frontier and its dispatch/execution/return obligations
are distinct. Canonical ledger row order is a serialization convention, not an
instruction to reorder the runtime's first-failure checks. The existing order
token denotes successful continuation: call 1 cannot borrow call 0's checks,
snapshots, or FP state, even when a descriptor binding is shared.
The frontier names identify obligation scopes, not an order among those scopes:
the real runtime discovers candidates/providers before its final FP check.

Provider failure is not universally pre-write. The retained commit contract
is `may_write_output_before_failure_no_rollback`. No rollback or retry is
authorized; synchronous completion and provider post-call checks are not
misrepresented as pre-read predicates. Exact host diagnostics and exception
control flow still belong to the unchanged C++/runtime route.

## Source evidence and ownership

These are **observed implementation facts**, not inferred compiler laws:

- [Native authentication](../../compiler/lib/frontend/native_frontend.cpp)
  verifies canonical declarations and captures per-call bindings. It does not
  inspect runtime descriptor contents or backing allocations.
- [Existing adapter generation](../../compiler/lib/codegen/codegen.cpp)
  supplies rank/dtype/stride/memory/mutability tags and snapshots the source
  descriptors. A host tag or mutable descriptor is not proof of physical
  host access or writable storage.
- [Runtime validation](../../compiler/lib/runtime/cpu_runtime.cpp), specifically
  `validate_gemm_v0` and `byte_count`, checks descriptor/policy fields, positive
  extents, shape equations, natural alignment and output/input byte overlap.
  Address arithmetic is checked against the current runtime address space;
  it does not establish allocation capacity. Input/input aliasing is allowed.
- The nonexecuting `matcore_runtime_plan_gemm_f32_v1` reuses descriptor
  validation but does not check the current FP environment. It uses planner v1,
  whereas one-shot execution uses resource/provider-aware planner v2. Its
  success cannot discharge the execution contract.
- [FP inspection/decoding](../../compiler/lib/platform/fp_environment_v1.h)
  checks nearest rounding, masked exceptions and gradual subnormals. Sticky
  exception status flags and recorded x87 precision are not additional legality
  requirements. Compile-time numerical conformance remains a build obligation.
- [OpenBLAS adapter](../../compiler/lib/runtime/cpu_openblas.cpp) can detect a
  provider-state violation after `cblas_sgemm` writes output. The public runtime
  promises unchanged output for pre-execution validation failures, not all
  failures. Provider discovery can also perform its own conformance work; it
  is not a pure semantic predicate evaluator.

**Architectural consequence:** Matcore must keep these evidence classes and
ordered failure frontiers distinct. The ledger does not duplicate runtime
validation, planner policy, upstream MLIR transformation machinery or backend
capability discovery. No target-specific alignment width, tile, cache, provider
crossover or scheduling decision enters semantic IR.

## Mechanical acceptance requirements

| ID | Required falsification control |
| --- | --- |
| L1 | A missing, duplicate, unknown or malformed ledger row/field fails closed. |
| L2 | Forged discharged/executed evidence or promoted caller/dispatch obligations reject. |
| L3 | Altered required alignment, dimension equation or either output/input overlap obligation rejects; input/input aliasing remains admitted. A=B still retains separate output/lhs and output/rhs role coverage, not duplicate rows for one role. |
| L4 | A ledger copied across call/site/binding/stage rejects; guard 1 cannot precede commit 0. |
| L5 | Source contract and ledger edited together cannot replace sealed native evidence. |
| L6 | Provider postconditions, synchronous completion and partial-write failure cannot become pre-write validation or rollback guarantees. |
| L7 | Named-to-generic Linalg transformation, canonicalization/CSE and fresh-context parsing still verify when source meaning is preserved. |
| L8 | Nonexecuting plan queries expose descriptor/shape/alignment/overlap predicates without reading tensor data or writing output. |
| L9 | A plan may accept numerically valid extents larger than the actual backing subobject; this fixture is never executed. Capacity stays unproven. |
| L10 | A plan may succeed under incompatible actual FP state; FP inspection rejects it. Descriptor validation retains priority over FP failure on the old route. |
| L11 | The FP decoder rejects control-state incompatibilities but does not invent sticky-status or precision-mode restrictions. |
| L12 | CLI inspection emits the bound ledger without source rewrite/executable artifacts; CPU lowering still rejects region IR despite forged authority labels. |

Standalone region verification checks internal consistency. Source-paired
verification reconstructs the expected ledger from the sealed native capture;
serialized attributes alone never authenticate runtime values or authorize
execution. Internal schema versioning is not a public interchange commitment.

## Validation and checkpoint

Implementation and validation are in progress. No tests or merge are claimed
by this working record yet. Final results and independent review will be
recorded here before integration.

## Remaining boundary

No runtime predicate is discharged in this milestone. No region execution,
bufferization, fusion, tiling, generated runtime replacement, broader region
admission, expanded tensor/view frontend, GPU/NPU support or API/ABI freeze is
introduced. Issue #15 remains partial; this work produces no benchmark or BLAS
parity evidence. Issue #20 remains a separate design proposal.

Exactly one next justified boundary is **mirrored RHS-only dependence in this
same inspection route**: `C = A*B; E = D*C`. The native seal already carries both
operand roles, but admission and value forwarding deliberately support only
the second lhs today. A separate bounded extension can test role symmetry,
noncommuting/non-square mathematics, late aliasing reads and unchanged failure
frontiers without new execution authority. Both-input forwarding and general
DAG admission are not part of that boundary. The ledger does not by itself
justify region bufferization, tiling or generated execution.
