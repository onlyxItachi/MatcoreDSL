# Certified structured-GEMM vector-readiness lane

Date: 2026-09-05

Branch: `mdslc/vector-readiness-certified-v2`

Canonical base: `72f02d15e1a0d3a2dae2bab76ea0c1ee968e67de`

Restacked vector implementation: `6f63e47`

Restack hardening and hosted opt-in coverage: `2f9e687`, `8481482`

Reviewed pre-merge head: `7b9501ec12fe99575cfabdd3e738d32f817370df`

Canonical PR #27 merge: `8ac6c4189b6f79aadee150007b6d26894de02660`

## Scope and dependency

This lane owns only an internal, default-off, inspection-only proof that a
bounded certified structured GEMM can be transformed by upstream MLIR into a
mechanically checked Vector representation. It changes no frontend operation,
installed API, ABI, driver stage, runtime, planner, provider, or execution
authority.

Canonical PR #25 merge `09c96b980020f53bf1d352f9ee6c28fb470540ea`
provides the single shared derived-source identity implementation. The old
patch-equivalent local duplicate was dropped. Merge commit `fa074df` gives the
survivor explicit ancestry from canonical PR #26.

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
- The certificate/golden are explicitly restricted to exact MLIR 21.1.8.
  Combining the vector option with the portability branch's experimental
  toolchain selector fails closed. The vector library uses that branch's
  canonical per-target LLVM RTTI helper, as does the focused test target.

The direct upstream dynamic control is important: the Transform interpreter
reports success but leaves the dynamic fill/matmul payload unchanged and
creates no Vector operation. The implementation therefore treats Transform
success only as a control result and rejects the dynamic artifact because it
does not satisfy the vector postcondition.

## Falsification coverage

The focused executable performs 285 checks. Its negative cases include
malformed structured input; source mutation; source substitution; multi-site
reorder/drop; aggregate and per-site fingerprint forgery; retained source type
and shape drift; function/block-argument/generated-operation location drift;
exact numerical-ledger drift; retained numerical-contract mutation; square
type-compatible map substitution; initial-C read; nonzero accumulator; loss of
the full `in_bounds` proof; result bypass; unauthorized attributes; authority
changes; and runtime-lowering attempts.

Positive controls cover a non-square golden, cross-context round trips,
two-site ordered pairing, unit M/N/K and all-unit rank-2 GEMM specimens
(including nontrivial `M=2, K=1, N=4`), and the observed dynamic upstream no-op
before fail-closed admission.

The static fixtures are programmatically constructed verifier specimens, not
authenticated current-frontend static captures. The dynamic test parses the
reviewed committed capture; it does not independently reauthenticate the
original source bytes.

## Validation

The restacked survivor was validated with exact 21.1.8 as the product tuple and
the coherent exact 22.1.8 package as a vector-disabled compatibility control.

| Configuration | Result |
| --- | --- |
| Exact 21 Release, OpenBLAS required, buffer + vector ON, full build | 146/146 passed |
| Exact 21 Release, buffer + vector ON, full CTest | 70/70 passed, 191.51 s |
| Exact 21 focused composed semantic chain | 5/5 passed |
| Direct vector-readiness executable | 285 checks, 0 failures |
| Exact 22 Release, OpenBLAS and vector OFF, full build | 142/142 passed |
| Exact 22 Release, vector OFF, full CTest | 69/69 passed, 145.70 s |
| Exact 22 focused buffer sibling chain | 4/4 passed |
| Option ON with Matcore MLIR OFF | configure rejected as required |
| Option ON with experimental toolchain 22.1.8 | configure rejected as required |
| Exact 22/default-off target and test inventory | no vector target or test |
| Staged install/API firewall | no vector artifact exported |
| Diff whitespace check | passed |

The existing exact-21 Release workflow now enables vector readiness only in
the OpenBLAS-disabled, Matcore-MLIR-enabled matrix row, runs the exact vector
test with `--no-tests=error`, and then runs the complete suite. Existing check
names/counts and every other default-OFF configuration are preserved.
Independent review was GO with no remaining findings, and the reviewed head
passed all 19 hosted checks before normal integration through PR #27.

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
