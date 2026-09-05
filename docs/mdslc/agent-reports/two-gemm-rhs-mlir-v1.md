# Two-GEMM RHS mirror: MLIR lane

Canonical base: `8a825350ae8e3f24e55bc8d050375a0b2c0da3dd` (PR #31).
Branch: `agent/two-gemm-rhs-mlir-v1`.
Implementation/test commit: `2d0b35d0c449461554f35156c409b9c3bd105e59`.
For the focused build, frontend commit
`d3e37c12acda0f4bef9890b9f0dff6707b1ed224` was temporarily cherry-picked as
`a2b5c2e70937899a976d4ad3f8bbebd602150022`. Integrate the original frontend
commit once, not this duplicate.

## Decision and scope

The existing source-connected representation already separates an observable
output commit, its produced tensor value, and the next call's guarded memory
snapshot. None of that depends on the carried tensor being the second GEMM's
lhs. The extension changes operand selection, not the authority or effect model.

`MatcoreTwoGemmRegion.cpp` derives the carried input from the two sites' bound
descriptors. First-output identity matching second lhs takes precedence;
otherwise second rhs must match. No serialized role field or new frontend DTO
is authoritative. The builder forwards the first commit's tensor into that
mathematical input position and reads the other input after the second guard.
Output rows still come from current lhs axis 0, columns from current rhs axis 1.

```text
C = A * B; E = C * D     -> E = C_post * read_after_guard1(D)
C = A * B; E = D * C     -> E = read_after_guard1(D) * C_post
C = A * B; E = C * C     -> E = C_post * read_after_guard1(C)
```

The last case was already admitted at the base checkpoint. An exclusive-or
admission rule would incorrectly narrow that baseline. It remains a lhs carry
with a separate rhs read, not a new both-input SSA-forwarding implementation.
Different descriptors still may alias physical bytes. The second read must
not move before the first write even when its descriptor is distinct.

The verifier parses exact i64 descriptor argument/snapshot fields before using
them to derive the carried role. It then checks the appropriate required read,
guard, descriptor, type and ordered arithmetic. A carried tensor must have its
declared input type. The existing computation verifier continues to validate
Linalg indexing, reduction, numerical body, overwrite initialization and output
extent provenance without requiring named operations, source locations or a
fixed incidental operation count.

No guard-ledger catalog/algorithm, operation definition, memory-effect model,
runtime, provider, allocator-registration shim, public API or ABI changed.
All 36 per-call ledger rows retain output/lhs/rhs meaning; roles are never
reordered to match the carried input. There is no generated region execution,
guard discharge, allocation/zero-copy claim, fusion, transpose feature, static
shape inference, new target support or performance claim.

## Falsification and validation

The source fixture now contains the two original regions plus RHS-only and
`C*C` regions. All actual captured tensor extents remain dynamic. Tests reject
upstream-valid MLIR mutations that swap operands, substitute a stale tensor,
change the read role or guard descriptor, leave the other read unused, erase
that read, borrow an old guard, hoist a read across the first commit, alter each
output extent's operand or axis, or change the carried input's declared type.
For `C*C`, rejecting both-input forwarding is a boundary of this reviewed
representation, not a claim that every possible equivalent rewrite is illegal.

Existing tests now exercise all four regions through real named-to-generic
Linalg conversion, canonicalization/CSE/symbol DCE, fresh-context serialization,
source pairing and execution-authority rejection. The guard-ledger mutation
suite remains active and checks the new sites as well. The independent upstream
bufferization control remains a diagnostic, not an executable region.

A fresh coherent Clang/LLVM/MLIR 21.1.8 Release build with native and bootstrap
frontends enabled, Matcore MLIR enabled and OpenBLAS OFF built the focused
target using Ninja `-j2`:

```sh
cmake --build build-rhs-mlir-release \
  --target matcore_mlir_two_gemm_region_tests -- -j2
ctest --test-dir build-rhs-mlir-release \
  -R '^mlir.semantic.two-gemm-region$' --output-on-failure -V -j1
```

Result at combined local head `a2b5c2e70937899a976d4ad3f8bbebd602150022`:
**426/426 checks passed**, **1/1 CTest passed**, **0.07 seconds**.
The generated log is
`build-rhs-mlir-release/Testing/Temporary/LastTest.log`.
`git diff --check` passed. No build processes remain in this lane.

Actual asymmetric/noncommuting numerical witnesses and late physical-alias or
second-call failure witnesses belong to the integration owner's unchanged
authenticated CPU routes. This report does not infer their outcome, nor any
full Release/Debug/OpenBLAS/sanitizer/hosted result, from the focused test.
Independent review and broader integration validation are separate gates.
