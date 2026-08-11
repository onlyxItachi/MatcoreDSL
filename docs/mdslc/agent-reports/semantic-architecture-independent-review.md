# Semantic architecture independent review

Date: 2026-08-11
Reviewed branch: `mdslc/semantic-compiler-foundation-v1`
Accepted checkpoint: `b233017`

## Verdict

Milestone A is **accepted**. The final independent re-review found no
unresolved high- or medium-severity architecture issue.

The first review rejected acceptance with four medium findings. All four were
resolved and re-reviewed:

1. Matcore IR v1 alignment and no-alias entries are required preconditions,
   not facts about concrete runtime values. Optimization may consume them only
   after static proof or a dominating guard that rejects before packing or
   destination mutation.
2. `mdsl.gemm` returns the post-overwrite semantic value tied to the explicit
   write-only destination. It does not imply independent allocation;
   bufferization must preserve result/destination storage identity, and the
   observable write is not dead merely because the SSA result is unused.
3. `explicit-gemm-f32-v1` now specifies round-to-nearest-ties-even,
   non-trapping exceptions, exception-status limitations, gradual subnormals,
   and disabled FTZ/DAZ. The documentation explicitly does not claim that the
   current runtime already validates this environment; Milestone E must add a
   pre-mutation gate and backend conformance tests.
4. Linalg, Tensor, MemRef, Vector, and generic GPU dialects are HOW-level
   structured substrates while choices remain open. MACHINE begins when LLVM
   or a target-specific dialect/toolchain structurally commits the relevant
   target choice.

## Evidence checked

- Live base remained `origin/main` at `e5069758ad04bdb459de2026cad8498b47fda707`.
- PR 16 was merged; Issue 15 and Milestone 7 remained open and partial; no
  native-BLAS-parity completion tag existed.
- ADR-0009, `ROADMAP.md`, `STATUS.md`, `PRE_FREEZE_DECISIONS.md`, and
  `AGENTS.md` agreed on the corrected boundaries.
- `git diff --check e506975..b233017` passed.
- `bash tests/check_repository_hygiene.sh` passed.
- Active uncommitted Matcore MLIR implementation files were excluded from this
  architecture-only review.

This acceptance freezes the internal WHAT/HOW/MACHINE and semantic-preservation
rules. It does not freeze public API, ABI, backend contracts, or claim that
Milestones B through H are complete.
