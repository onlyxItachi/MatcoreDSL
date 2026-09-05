# Certified structured-GEMM vector-readiness lane

Date: 2026-09-05

Branch: `mdslc/vector-certified-v1`

Canonical base: `5983c2c1bf067bed9e69d0172b0944b7a4c14c00`

Vector-specific implementation:
`4a6711654f02dc50c4d4fcc2ae5cbafd16f61b75`

## Scope and dependency

This lane owns only an internal, default-off, inspection-only proof that a
bounded certified structured GEMM can be transformed by upstream MLIR into a
mechanically checked Vector representation. It changes no frontend operation,
installed API, ABI, driver stage, runtime, planner, provider, or execution
authority.

The branch also carries local dependency commit
`d29e1c347a45ba42e8bc413ee4e9fb4b9deb4fed`. Its stable patch ID
`31cb57f814faac7d520aef0375f139dc6fe19d2b` exactly matches shared
derived-source identity commit
`d3d4189aedb7346f853e0d5bd8d845d892612fdc` from draft PR #25. The
vector-specific commit must be restacked on that dependency's canonical merge
before integration; this branch is not a merge candidate as stacked locally.

The earlier experimental worktree
`MatcoreDSL-wt-structured-transform-vector-v1` remained clean and unchanged at
`d16f83cbc29dbd20a59ebeae9e056fa3378962c2`. It was implementation evidence,
not a source branch adopted wholesale.

## Surviving design

- Admission starts from exact verified certified structured GEMM, not a
  reconstructed or unauthenticated witness.
- Only positive, fully static, rank-2 f32 GEMM is admitted.
- The source module is cloned before applying upstream MLIR 21.1.8
  `transform.structured.vectorize_children_and_apply_patterns`.
- Shared operation-neutral derived-source certificates bind every result site
  to the exact source function type and semantic fingerprint and bind the
  ordered module site set. Paired verification recomputes those values from
  the supplied source.
- The GEMM-specific verifier checks a closed postcondition: full unmasked
  identity transfers, canonical contraction topology, positive-zero
  accumulator, no initial-C read, full write to the original destination
  tensor, exact result dataflow, exact retained semantic contract, and an
  exact unconsumed numerical-permission ledger.
- The resulting module remains analysis-only and inspection-only. The existing
  runtime lowerer rejects it without producing a partial execution record.
- No tile, vector width, target matrix shape, unroll factor, thread count,
  packing rule, or provider crossover is encoded.

The direct upstream dynamic control is important: the Transform interpreter
reports success but leaves the dynamic fill/matmul payload unchanged and
creates no Vector operation. The implementation therefore treats Transform
success only as a control result and rejects the dynamic artifact because it
does not satisfy the vector postcondition.

## Falsification coverage

The focused executable performs 280 checks. Its negative cases include
malformed structured input; source mutation; source substitution; multi-site
reorder/drop; aggregate and per-site fingerprint forgery; retained source type
and shape drift; module/function/block-argument/generated-operation location
drift; exact numerical-ledger drift; retained numerical-contract mutation;
square type-compatible map substitution; initial-C read; nonzero accumulator;
partial or masked transfer; result bypass; unauthorized attributes; authority
changes; and runtime-lowering attempts.

Positive controls cover a non-square golden, cross-context round trips,
two-site ordered pairing, unit M/N/K and all-unit rank-2 GEMM specimens, and the
observed dynamic upstream no-op before fail-closed admission.

The static fixtures are programmatically constructed verifier specimens, not
authenticated current-frontend static captures. The dynamic test parses the
reviewed committed capture; it does not independently reauthenticate the
original source bytes.

## Validation

All builds selected Ubuntu Clang/Clang++ 21.1.8 and the user-local MLIR 21.1.8
package named by `AGENTS.md`.

| Configuration | Result |
| --- | --- |
| Release, experimental option ON, full build | passed |
| Release, experimental option ON, full CTest | 66/66 passed, 184.26 s |
| Release, default OFF, full build | 136/136 build steps passed |
| Release, default OFF, full CTest | 65/65 passed, 208.31 s |
| Debug, experimental option ON, focused semantic chain | 3/3 passed |
| Direct vector-readiness executable | 280 checks, 0 failures |
| Option ON with Matcore MLIR OFF | configure rejected as required |
| Default-off target and test inventory | no vector target or test |
| Staged install/API firewall | no vector artifact exported |
| Diff whitespace check | passed |

The first opt-in full CTest run correctly caught dirty-checkout and stale
embedded-source-SHA provenance. After committing the documentation, cleanly
reconfiguring, and rebuilding `matcore-bench` at exact HEAD, the full 66/66
rerun passed. No test or provenance check was relaxed.

`clang-format-21` was unavailable. The three new C++ files were formatted with
the available LLVM 22.1.8 formatter and then compiled and tested with the exact
21.1.8 product toolchain.

## Rejected alternatives and non-claims

- Rejected: the old GEMM-specific reconstructed structured witness and
  whole-module byte-serialization equality. The shared certificate is the one
  provenance mechanism.
- Rejected: treating a successful Transform application as proof. The dynamic
  no-op falsifies that interpretation.
- Deferred: dynamic, tiled, scalable, masked, transposed, batched, GEMV, DOT,
  GER, and non-f32 vector forms.
- Unproven: physical output-buffer identity, in-place bufferization, allocation
  or copy freedom, zero-copy execution, target legality, profitability,
  performance, and BLAS parity.

Issue #15 remains partial and open. The next justified vector step first needs
explicit bounded tile/shape evidence and a reviewed tensor-versus-buffer
ordering, including remainder/mask legality and exact semantic consumption.
This lane stops before that decision.
