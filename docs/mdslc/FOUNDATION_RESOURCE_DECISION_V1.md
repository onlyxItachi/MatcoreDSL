# Foundation decision: immutable values and bounded host publication

Status: **bounded host adapter validated and normally merged in PR #39**.
This is not yet authenticated source-region or generated-kernel execution.
Starting canonical main: `fd64850a0c8cb7d2c0801a64dd94d515dd714130`
([operator PR #38](https://github.com/onlyxItachi/MatcoreDSL/pull/38)); latest
engineering merge: `f6a5d9718a8c704231ad37d9e315f7d2d6912a39`
([host-admission PR #37](https://github.com/onlyxItachi/MatcoreDSL/pull/37)).

The owner has delegated consequential technical decisions and evidence-backed
implementation/normal merges. Historical owner-approval pauses remain historical
records, not the current execution plan. Neither that delegation nor this record
turns an existing inspection specimen into executable authority.

## Decision and alternatives

Retain C++ as the host and admit a closed mathematical sublanguage through Clang.
The demonstrated grammar and frontend-neutral semantic graph survive. Build a
synchronous host-storage adapter, then connect authenticated source to statically
compiled orchestration and a generated CPU candidate. Do not interpret ASTs or
serialized inspection IR at runtime.

Two independent advocates investigated isolated values/strong publication and
selective borrowing/partial publication. They agree that immutable value meaning
does not prescribe allocation. They disagree about the default failure guarantee
for external publication. The selected first **host adapter** has a strong
per-publication normal-return guarantee; this is not a universal guarantee for
all external resources.

| Alternative | Decision and falsifier |
| --- | --- |
| Immutable values with implementation-selected realization | KEEP. Old values must survive publication through an alias; borrowing is legal only while that meaning survives. |
| Unconditionally pointer-backed logical values | REJECT. An old value changes when its borrowed resource is overwritten. |
| Snapshot every input at region entry | REJECT. A late read through an alias must see an earlier publication. |
| Snapshot at each ordered read initially | ACCEPT as conservative realization, not language semantics or a performance claim. |
| Whole-region transaction/rollback | REJECT. A first published/observed result remains observable when the second operation fails. |
| Direct fallible candidate output into eventual external destination | REJECT as default. Failure can follow partial mutation, or an intervening check can fail before publication is reached. |
| Strong publication for validated synchronous host storage | ACCEPT. All recoverably failing preparation precedes a non-failing byte-copy phase under the stated caller preconditions. |
| Universal atomic publication for devices/files/networks | REJECT. A transfer/export itself can fail after partial effects. It requires a different explicitly declared adapter contract. |

## Semantic contract selected for the adapter

1. A read denotes immutable `f32` contents at its ordered frontier, not an
   allocation or an address. A read after publication sees the updated resource.
2. External resource views describe shape, bounded capacity and access. Distinct
   descriptors remain MAY-alias, including partially overlapping subranges.
   Valid objects, truthful capacity/lifetime/access and freedom from conflicting
   concurrent access remain caller preconditions; inspecting a pointer cannot
   establish them.
3. Shape checks and required observations cannot disappear merely because a
   mathematical result is unused. Source-required failure ordering is separate
   from the physical schedule.
4. Fallible candidates produce private logical results. Candidate completion,
   numerical legality and provider-state checks precede exposure of that result.
   An arbitrary callback or imported certificate is not a trusted candidate.
5. Host publication validates destination shape/access/capacity and performs all
   potentially failing bookkeeping before copying. On a defined normal return,
   a failed publication leaves its own destination unchanged. Successful earlier
   publications are not rolled back. This is not concurrent atomicity, signal
   safety, crash recovery or protection against invalid pointers.
6. Observation captures an immutable current-resource record. Readiness alone
   is not observation, and an uncontrolled host callback is not a closed effect.
7. Status identifies failure and the completed effect prefix. The first failure
   is sticky: later statements must neither access resources nor publish. No
   exception crosses the private adapter's status-returning boundary.
8. Implementation failures, such as allocation exhaustion, can vary with the
   realization. They may return only an allowed completed effect prefix. A
   speculative later failure cannot retire ahead of earlier required effects.
   Optimization may remove an allocation opportunity, not a source-required
   check.
9. The new mathematical execution scope uses nearest-even `f32`, gradual
   underflow and masked machine exceptions. It restores the caller's complete
   floating-point environment on successful and failing normal returns. Internal
   sticky flags are not implicit host observations. Strict GEMM has increasing-K
   separate multiply/add; reassociation permission is per operation and does
   not erase each operation's `f32` rounding boundary.

The initial runtime trusts conforming allocation/deallocation routines to
preserve FP state, including on failure and destruction outside the numeric
scope. Arbitrary interposed allocator hooks with host effects are not sandboxed
or covered by that guarantee. Session use is thread-confined; synchronous
reentry detection is not concurrent locking.

Zero extents are already admitted by the closed semantic model. Empty accesses
must not dereference null storage; a zero-K nonempty product has the specified
positive-zero initial accumulator. Dimensions and products still require
representability checks, including when another extent is zero.

The old mutating `mdsl::gemm` and its runtime/provider ABI retain their existing
semantics. The private inspection dialect's general partial-publication outcome
is not silently rewritten. The new synchronous host adapter is an explicit
stronger resource contract; other resource classes are not implemented by it.

## Evidence that selected the design

**OBSERVED in canonical source:**

- `compiler/lib/mlir/MatcoreClosedRegion.cpp` already separates read-at-frontier
  immutable values, resource epochs, checks and publication/observation order.
- `compiler/lib/runtime/cpu_openblas.cpp` invokes CBLAS before its post-call
  provider-state checks. `cpu_openblas.h` expressly permits output mutation
  before a provider-state error. Scratch output is necessary for isolating this
  candidate failure; it does not prove input/global-state purity by itself.
- `compiler/lib/platform/fp_environment_v1.cpp` restores controls, not the whole
  FP environment. Its Linux `fnclex` clears x87 exception flags. This is a
  compatibility-contract fact, not an automatic bug in that older contract.
- Existing native targets permit contraction. They are not automatically
  implementations of closed `strict_f32`, which forbids FMA contraction.

**OBSERVED in bounded throwaway experiments, not product execution:**

- An isolated-value advocate's ASan/UBSan probe passed 13 checks. It distinguishes
  old values from late reads, demonstrates partial candidate mutation, and
  preserves the first publication across later failure.
  `publication_probe.cpp` SHA-256:
  `3ea416681ecae914f0d18207817cc26712b88862e8affcf4e18cf4ab4fdd93d1`.
- A borrowing advocate's independent ASan/UBSan probe passed 19 checks. It also
  disproves entry snapshots, eager later-check retirement and the inference
  that scratch computation makes a fallible external transfer atomic.
  `contract_counterexamples.cpp` SHA-256:
  `123075a3d137a221148d15817455ef70415ace963763da48cbbb400292004b97`.
- A separate exact-21.1.8 MLIR experiment lowered dynamic Linalg through standard
  loops and LLVM IR to an ordinary object and executed 28 checks on Linux x64,
  including rectangular/degenerate geometry, special values and an FMA
  discriminator. This is upstream feasibility, not authenticated source
  execution. Omitting CF-to-LLVM conversion correctly failed translation.
- Merely linking ASan into generated LLVM IR did **not** instrument its loads and
  stores: a deliberate out-of-bounds access escaped detection. Adding
  `sanitize_address` to generated functions made that negative control fail
  with an actual ASan heap-buffer-overflow. Generated-code validation must test
  instrumentation, not infer it from an ASan runtime symbol.

Temporary probes are research evidence, not durable regression coverage. Their
relevant falsifiers must be incorporated in committed tests before accepting an
engineering checkpoint.

## Upstream ownership and architectural implications

- MLIR One-Shot Bufferize preserves tensor SSA meaning while selecting reuse or
  copies. Its tensor/buffer `restrict` boundary needs established alias facts;
  Matcore cannot assert it from descriptor inequality. This supports immutable
  values without mandatory materialization. [Pinned MLIR 21.1.8 design](https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/mlir/docs/Bufferization.md)
- XLA separates compile-time alias configuration from actual runtime donation;
  absent donation it can protect inputs by copying. Reuse authority is not a
  consequence of a value's type name. [XLA aliasing](https://openxla.org/xla/aliasing)
- Standard MLIR conversion and LLVM translation already provide the lowering
  machinery and private ranked-memref C wrapper ABI. Matcore should establish
  semantic/adapter preconditions, not recreate loop lowering or instruction
  selection. [LLVM target interface](https://mlir.llvm.org/docs/TargetLLVMIR/)

These are upstream contracts and architectural implications, not proof that
Matcore has implemented or validated every supported upstream transformation.

## Execution program and merge gates

The immediate dependent boundaries are host adapter conformance, strict AOT CPU
candidate derivation, authenticated source-to-static-orchestration connection,
then validated candidate coexistence and installed tooling. Independent branches
may implement separable pieces; each integration needs its own exact validation
and adversarial review. A passing candidate kernel alone does not finish the
source compiler.

The first adapter gate includes saved-old/late-read alias counterexamples,
partial overlap and reuse, ordered second failure after publication, private
candidate partial mutation, allocation failure before all effects, actual
observation snapshots, zero/overflow shapes, caller FP control **and status**,
no callback authority/reentrancy escape, and affected compatibility regressions.

No planner crossover, performance claim, zero-copy claim, accelerator support,
Issue #15 completion or public stability commitment follows from this decision.
Windows compatibility remains supported independently; a new Linux-only proof
must not be called Windows generated execution.

## Host-adapter integration evidence

Implementation origin: `6bc3069c5f5ee49469061fdd07feaf962a96e4b8` on
`mdslc/closed-host-adapter-v1`, preserved as
`b33a1dd7621b9d60c2940c783340b9a8c8c93a7c` in the integration branch.
Only new private runtime/test files are introduced; no legacy implementation,
installed API or driver route is replaced.

Focused integrated Release CTest: **5/5 passed** (1.13 seconds), including
production smoke, 137 implementation checks, 216 independent adversarial checks,
131 production allocation-failure checks and compile/link authority controls.
See the [test contract](../../compiler/tests/closed_host/README.md) and
[independent ACCEPT review](reviews/CLOSED_HOST_ADAPTER_INDEPENDENT_V1.md).

The first full Release run on the deliberately uncommitted integration returned
**78/83 passed** (150.50 seconds). Five existing consumer/benchmark tests rejected
the dirty source provenance as intended. This is retained as negative evidence,
not waived: rebuild at a clean committed checkpoint and rerun the complete suite
before integration. The first clean rebuild gave 82/83 because the package test
also authenticates cleanliness at CMake configuration time. After clean
reconfiguration its isolated retry passed (53.74 seconds); the final complete
Release run passed **83/83** (195.32 seconds) at clean
`53f390d3cab5ed69d9347748d44793d6e05c87bd`.

Affected integrated ASan/UBSan scope passed **32/32** (24.86 seconds), including
all five new adapter tests and the existing allocator-protocol negative control.
No sanitizer exclusion or provenance guard was disabled. Independent review
also accepted the CMake, workflow and evidence integration.

Normal merge: **`f988882710ac0b3677d908b3442841e3a8986b81`**,
[PR #39](https://github.com/onlyxItachi/MatcoreDSL/pull/39),
2026-09-06T18:42:39Z. Parents are the starting canonical `fd64850...` and reviewed
head `53f390d3cab5ed69d9347748d44793d6e05c87bd`; the merge tree equals that head.
All **19/19 exact-head hosted checks succeeded** before the normal merge,
covering Release OpenBLAS on/off, MLIR on/off, Debug, ASan/UBSan, TSan, Windows,
legacy build/tests and repository hygiene. The Copilot quota notice was not
counted as an approval. See the
[validation record](https://github.com/onlyxItachi/MatcoreDSL/pull/39#issuecomment-5561282175)
and [merge gate](https://github.com/onlyxItachi/MatcoreDSL/pull/39#issuecomment-5561334781).

This merged boundary is an independently instantiated host adapter, not yet a
connected source compiler. Subsequent candidate/source branches retain their
own reviews, authority boundaries and integration gates.
