# Two-GEMM RHS mirror: independent architecture review

Reviewed immutable head `33423177da947bc40d43680a33a6501a31d14736`
against engineering base `8a825350ae8e3f24e55bc8d050375a0b2c0da3dd` (PR #31).

Verdict: ACCEPT the bounded inspection-only extension, subject to the
integration owner's complete regression and exact-head hosted CI gates.
No blocking architecture/code finding remains.

This independent lane read AGENTS.md, the complete implementation/test diff,
surrounding native admission, builder and paired verifier, boundary operation
verifiers/effects, authority-rejection tests, upstream storage controls and
lane reports. It made no production edits and ran no builds.
Its immutable-range `git diff --check` passed. Build/test outcomes belong to
the producing lanes and integration logs, not this review.

## Findings and challenged counterexamples

The change selects which existing ordered input consumes the first committed
tensor. It adds no source identity field, caller-authored forwarding role,
operation, guard predicate, storage model, runtime route or execution authority.

Native admission uses resolved descriptor identity inclusively. Existing
authentication, transparent-reference restrictions, adjacency and host/control
barriers, distinct outputs and descriptor self-alias rejection remain.
Builder and verifier preserve source operand order: lhs carry for C*D,
rhs carry for D*C, and the existing lhs carry plus late rhs read for C*C.
Different descriptors remain no physical noalias proof.

- Same-type operand swaps reject. The numerical D*C anchor is noncommuting.
- Current lhs/rhs supply output axes; operand and index mutations reject.
  Rectangular execution distinguishes geometry; captured extents remain dynamic.
- Carried values must come from commit0, with their declared input type;
  same-shaped stale precommit values reject.
- Noncarried reads must follow guard1 and commit0. Wrong role, guard0 borrowing,
  hoisting, erasure or unused reads with dual forwarding reject. The late-alias
  oracle views the same six floats as C(2x3) and D(3x2).
- Descriptor roots, roles, exact i64 argument indices and snapshot stages are
  checked before carried-role selection. Serialized edits or mutable frontend
  diagnostics cannot replace the native seal.
- Commit0 precedes guard1. Second-call dimension failure preserves the first
  write and untouched second output on the original route. An observer blocks
  admission but original execution must see C[0]=62.
- All four regions participate in actual named-to-generic conversion,
  canonicalization/CSE, symbol DCE, fresh-context parsing and source pairing.
  Verification checks semantics, not incidental operation spelling.
- The unchanged CPU lowerer rejects region IR despite forged authority labels.
  Standalone consistency is not source authentication.
- Stage-indexed arrays are protected by upstream/dialect verification, which
  rejects stages outside 0 and 1 before array use.

No runtime/provider implementation, guard catalog, ODS operation, sanitizer
registration shim, public API or ABI changed. Numerical witnesses execute only
the established per-call routes; no region or guard is executed/discharged.
No capacity/lifetime, allocation, zero-copy, fusion, performance, rollback or
target-support claim follows.

## Next boundary and stop condition

A connected region storage/commit consumer needs an explicit design decision.
Boundary bufferization interfaces and tensor islands with explicit destination
materialization carry different alias, allocation/failure and ownership
contracts. The existing upstream control's allocation/copy without resolved
ownership is not authenticated region lowering and cannot choose a model.

The owner's stop rule for consequential architectural alternatives applies.
Stop before implementing that consumer; inspection bufferization cannot grant
generated execution authority.

The inherited operator checkpoint required maintenance reconciliation; the
integration owner must update it after the exact engineering merge.
