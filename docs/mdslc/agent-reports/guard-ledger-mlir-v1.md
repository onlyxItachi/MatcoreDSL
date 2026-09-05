# Two-GEMM guard ledger: MLIR lane

Canonical starting checkpoint:
`63d642d9e4eb4183acf58a4391e1c668e734b9f0` (clean `main`, PR #29 merge).
Agent branch: `agent/guard-ledger-mlir-v1`.
Implementation/test commit: `b7171fafa8b385d0199abf4bb2fdf4409e68b8ed`.

## Decision and implementation

The six former guard-family strings conflated source representation, checks
that actual execution would need, caller-owned backing conditions, and later
provider obligations. The existing source-connected region, ordered effects,
committed values and native evidence remain unchanged. Only the guard's internal
inspection contract is refined.

`MatcoreRegionGuardLedger.{h,cpp}` defines a closed typed C++ vocabulary and an
ordinary MLIR dictionary codec. `RegionGuardOp.guard_ledger` replaces its former
`required_guards` string array; the unrelated runtime-lowering and recovered-C++
guard fields are untouched. No additional MLIR type/attribute registration,
runtime predicate engine, universal certificate framework, source rewrite or
public schema contract is introduced.

Each wrapper binds an exact source site, call stage, and output/lhs/rhs
descriptor identities. Each predicate records its operand roles, evidence
class, and obligation scope; only pointer alignment carries an additional
positive power-of-two minimum in bytes. Alignment is taken from the source
requirement, never detected from a concrete pointer. The original complete
semantic contract remains authoritative for type/layout/numerical/policy facts.

| Evidence class | Catalog entries per call | Meaning |
| --- | ---: | --- |
| `representation_only` | 7 | Authenticated source contract/designation only; not physical storage proof. |
| `runtime_validation_required` | 18 | Per-role nonnull, positive dimensions, alignment and representable ranges; exact K/M/N equations; both output/input overlaps; current FP environment. None has run. |
| `caller_precondition_unproven` | 6 | Valid descriptor objects; actual host accessibility, sufficient backing capacity, lifetime, required access permission and no conflicting concurrent access. |
| `dispatch_execution_obligation_retained` | 5 | Actual implementation eligibility/resources/numerical conformance plus provider state, synchronous completion and possible failure after writes. |

There is no executed/discharged enum. Neither `float *`, `out(C)`, a host tag,
nor numerically representable address ranges establish accessible allocation
capacity or physical writability. Distinct descriptor bindings may alias bytes;
equal A/B bindings retain both role-specific output-overlap obligations.

The frontier names are obligation scopes, not an execution schedule. In
particular `call_validation_before_compute` does not mean before provider
discovery: the existing runtime checks FP state later. Canonical predicate order
cannot reorder original first-failure precedence. Execution/return obligations
are explicitly distinct from pre-compute requirements, and commit still permits
failure after partial output mutation without rollback or retry authority.

## Mechanical rejection and review

The codec rejects unknown fields, predicate/evidence/frontier/role vocabulary,
and malformed integer types. The region verifier then requires the exact
regenerated source-scoped catalog: omissions, duplicate rows, wrong roles,
changed alignment and cross-call ledger borrowing fail closed. Native source
pairing is still required. A coordinated changed source alignment plus a newly
regenerated ledger can pass standalone self-consistency but cannot pass pairing
with the immutable original native evidence.

Independent compiler review found an adjacent pre-existing malformed-input
hazard in the descriptor `argument` index: `IntegerAttr::getInt()` preceded an
exact width check. The region now requires signless i64 before reading it.
Adversaries include i128 high-bit, negative, wrong-width and noninteger binding
indices/snapshot stages and malformed source/ledger alignment. This is a narrow
fail-closed parser correction, not an expansion of region semantics.

Other tests independently alter source site, stage and descriptor identity;
omit each output-versus-input overlap role; invent executed/proven states;
promote caller/provider obligations into representation facts; and move
post-execution obligations into a pre-compute scope. Existing real
named-to-generic/CSE/canonicalization, fresh-context roundtrip, A=B and CPU
execution-authority rejection controls remain active.

## Exact validation

A fresh exact Clang/LLVM/MLIR 21.1.8 Release build, native plus bootstrap,
Matcore MLIR enabled and OpenBLAS OFF, built only the focused
`matcore_mlir_two_gemm_region_tests` target with Ninja `--parallel 2`.
The existing prebuilt-MLIR allocator registration boundary was preserved.

Final executed command:

```sh
ctest --test-dir build-guard-ledger-release \
  -R '^mlir.semantic.two-gemm-region$' --output-on-failure -V
```

Result: **292/292 checks passed**, one CTest, 0.05 seconds. The earlier
intermediate 218-check run also passed. Exact final local logs are
`/tmp/mdslc-guard-ledger-final-build.log` and
`/tmp/mdslc-guard-ledger-final-tests.log`. `git diff --check` passed.
No full Release/Debug/OpenBLAS/sanitizer/hosted outcome is inferred here; the
integration owner records those separately. No runtime/frontend production
files were changed in this lane and no generated region execution was added.
