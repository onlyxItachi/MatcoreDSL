# Correctness-first connected-region optimization campaign

## Starting checkpoint and scope

Started 2026-10-04 from clean canonical `main`
`43ce3cb2557aad4dc3b810ea715c00531c76628f`, independently compared with live
GitHub before mutation; no open PRs. The preceding multi-target campaign is
recorded in [its reconciliation](MULTITARGET_CORRECTNESS_CAMPAIGN_V1.md).
Historical worktrees and raw evidence remain untouched.

The owner approved parallel implementation with Sol workers at max/ultra effort
and independent adversarial review. Correctness of optimization is the objective;
speedups and Native BLAS Parity are not completion requirements. GEMV, sparse and
other mathematical forms remain broader product directions, not implied coverage
of this bounded campaign. No public API/ABI stability is promised.

## Terminal capability

PROPOSED: authenticated C++ source can enter a separately verified derived
execution plan, retain original semantic evidence and ordered checks/effects,
and execute a genuinely fused, bounded CPU segment for an existing two-GEMM
mathematical chain. The full private intermediate can be replaced by bounded
row-panel scratch without changing either operation's numerical meaning.

This is not implemented by the starting checkpoint. Whole-region MLIR is
currently an exact untransformed witness, and sealed Program orchestration calls
compiler-issued individual GEMM candidates. Successful upstream transformations
or a prototype executable alone cannot authorize a source-program replacement.

## Dependent engineering boundaries

1. **Authenticated publication-to-read forwarding.** Retain original Program and
   exact source witness. Issue a small internal derived plan; independently check
   same resource/version, dominating successful publication and no intervening
   possible-alias write. Keep every read's requested-shape/extent/resource
   validation at its original frontier before reusing a retained immutable value.
   Distinct descriptors are not proof of disjointness. Branches may conservatively
   invalidate reuse until a stronger proof is implemented and tested.
2. **Strict two-GEMM structured candidate.** For `C=A*B; E=C*D`, initially admit
   only input reads preceding an adjacent strict pair with a private, single-use
   intermediate and no intervening publication, observation or branch. Use
   upstream Transform/Linalg to compute a producer row panel once, consume it
   across the second operation's columns, then continue. Preserve producer f32
   rounding, both increasing reduction orders and every logical-extent check.
   Do not replace `(A*B)*D` by `A*(B*D)`, infer FMA permission or recompute the
   producer once per consumer column tile.
3. **Connected source execution and reconciliation.** Bind candidate derivation,
   source identities, original failure locations, private scratch/output,
   completion/FP checks and authenticated generated artifacts. Execute original
   source with the installed driver, test hostile inputs and preserve every
   existing target/provider/compatibility path. Record bounded footprint and
   predeclared same-source fused/unfused measurements, including regressions.

Each accepted engineering boundary gets a focused reviewed PR, full affected
qualification and normal merge, then a compact `CURRENT_STATE.md` update.
Forwarding is the first merge, not the entire campaign's termination point.

## Mechanical acceptance and falsifiers

- Original source admission and exact paired witness remain authoritative;
  forged/stale derivations, changed source bindings, changed frontier mappings
  and foreign candidate artifacts fail closed.
- Forwarding rejects intervening overlapping publication, wrong requested
  shape/capacity/access, untaken-arm publication and insufficient descriptor
  identity. Saved old values and owning observations retain their contents.
- A transformed segment preserves required check order and failure attribution.
  A later invalid input/profile cannot hide an earlier required failure. Earlier
  successful publications/observations survive a later failure; later effects
  and resource accesses do not occur after sticky failure.
- Only the existing [resource contract](FOUNDATION_RESOURCE_DECISION_V1.md)
  permits allocation opportunities to disappear. Required checks cannot be
  deleted, nor may speculative later failures retire before earlier effects.
- Strict f32 tests cover separate multiply/add, cancellation, operation-boundary
  rounding, subnormals, signed zero, nonfinite values, zero-K, tails, zero extents
  and overflow. Full caller FP controls/status are restored on normal return.
- Structural evidence must show actual fused producer/consumer computation,
  reduced intermediate allocation and no unintended producer recomputation.
  Two unchanged Session GEMM calls are not a fusion result.
- Publication remains backed by private completed output; failed publication
  keeps its own destination unchanged under the established normal-return
  preconditions. A row-panel intermediate does not authorize early output writes.
- Source/installed execution, independent oracles, fault injection, proven
  sanitizer instrumentation, Release/Debug, provider on/off, native ARM and
  Windows compatibility, GPU qualification and repository hygiene retain their
  appropriate evidence scopes. Unsupported environments are reported, not passed.

## Parallel ownership and resource discipline

Sol implementation owns a dedicated forwarding branch. A second Sol lane owns
an isolated upstream fusion prototype; its executable is research evidence only.
Independent Astra review attacks admission, runtime and failure invariants.
Integration owns shared builds, cross-branch decisions, hosted CI and merges.
Parallel tasks do not edit each other's worktrees; semantic interfaces and
source authority changes are reconciled before dependent implementation.

Use coherent LLVM/Clang/MLIR 21.1.8 initially. Upstream already supplies the
needed [Transform fusion mechanisms](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/docs/Tutorials/transform/Ch1.md)
and [tensor-first bufferization](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/docs/Bufferization.md).
Newer LLVM remains a separate investigation only for a demonstrated gap. No
LLVM/Rust source build, casual system install or global compiler mutation.
Build/TMPDIR use SSD-backed task locations, at most two total compilation jobs
and serialized heavy links. No source edits during a frozen qualification run.

## Explicit non-goals

No new sparse/GEMV/epilogue operation surface, general rank-N/view language,
all-target fusion, persistent device residency, asynchronous source effects,
automatic performance policy, public API/ABI freeze, Metal qualification,
universal performance claim or Issue #15/#20 closure. Unsupported optimization
patterns keep the existing checked route or fail clearly when explicitly forced;
they do not silently inherit new execution authority. Negative results remain
part of the durable campaign record.
