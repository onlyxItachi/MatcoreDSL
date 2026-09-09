# Reassociate-profile register-retained GEMM research

Status: **PROVEN WITHIN A BOUNDED RESEARCH CONTRACT**, not a registered candidate,
source execution authority, performance result, or product-target claim.

Starting canonical checkpoint: `0b2c21f126288501fe27afea9320ec9fce392dcc`.
Research branch: `research/mlir-reassociate-register-v1`.
Fixture/runner: [mlir_reassociate_register_v1](../../../compiler/experiments/mlir_reassociate_register_v1/).
Reviewed source contract: [region numerical permissions](../REGION_COMPILER_V1.md#numerical-permissions-and-candidate-choice)
and `compiler/lib/mlir/MatcoreClosedRegion.cpp`, `gemmContract`.

## Verdict

**Yes:** pinned upstream MLIR generates a dynamic GEMM with an actual
register-carried output tile, without private vector lowering or blanket
fast-math. This is a better next *measurement hypothesis* than another strict
cache-tile guess: it changes the inner-loop accumulator/storage structure that
the earlier losing cache schedule did not fix. It is not yet evidence to select
a new default, add a production candidate, or infer BLAS parity.

The candidate's fused full-tile and unfused tail realizations are both permitted
by the existing **per-GEMM** `reassociate_f32` contract. Strict-profile rejection
is mandatory. No source syntax or numerical permission was added here.

## Actual pipeline and ownership

The complete reproducible commands are in `run.sh`. They use the existing
coherent Ubuntu 21.1.8 / `6ubuntu1` tuple, with no dependency installation:

```text
dynamic memref GEMM: canonical +0 fill, A*B accumulation into private C
  -> upstream Transform tile [4,8,1]
  -> upstream SCF peel N and M: exact full tiles, separate scalar tails
  -> canonicalize; vectorize static Linalg children
  -> upstream vector.contract / outerproduct lowering
  -> remaining Linalg -> scalar loops; fold memref alias operations
  -> upstream hoist redundant vector transfers
  -> Vector / SCF / MemRef / Arith -> LLVM -> x86-64-v3 object
```

The scheduled full tile has a `vector<4x8xf32>` SCF loop-carried accumulator.
Its K loop increases by one. C is read once before and written once after that
loop. The zero fill remains earlier. There is no reduction padding, allocation,
copy, parallel distribution, cross-operation rewrite, or hidden provider call.
The physical O3 inner loop uses four YMM `vfmadd231ps` accumulators, four scalar
broadcasts from A and one contiguous B vector load per K step. No C load/store
or accumulator spill occurs inside that inspected loop. This is an artifact
observation, not a guarantee of register allocation on all compilers/targets.

Tile sizes are research schedule constants, not Matcore semantic facts. Inputs
may alias each other; C must be private/disjoint. This artifact encodes **no**
LLVM `noalias` attributes. Descriptor identity is not a data-disjointness proof.

Upstream mechanics used:

- [SCF peel contract, exact 21 tag](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td#L143-L187).
- [Linalg vectorization and transfer-hoisting interfaces](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td#L2327-L2517).
- [Outer-product/FMA construction](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/Vector/Transforms/LowerVectorContract.cpp#L111-L147).
- [Vector FMA to LLVM FMulAdd](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVM.cpp#L1105-L1131).

## Hoisting preconditions are not optional

The pinned [hoister implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/Linalg/Transforms/Hoisting.cpp#L175-L366)
is conservative around views and explicitly unsuitable for general distributed
memref loops. Its interface is marked obsolete upstream. This experiment does
not turn it into a general Matcore alias or effect solver.

The fixed serial pipeline removes all destination subview users before hoisting.
The optional nonzero-trip proof is disabled: for **this** GEMM, K=0 moves only a
C read/write outside K, still inside nonempty full-M/full-N tile loops, after the
dominating fill. Those accesses are valid, initialized private storage; A/B
loads remain inside K. Null inputs at zero footprint are exercised. M=0 or N=0
does not execute a tile transfer. The raw leaf has no source-visible publication
or failure frontier; it is not called on arbitrary user buffers by this study.

This reasoning would **not** authorize hoisting an input load from an arbitrary
possibly-empty loop, moving an observation, using shared output storage, or
distributed execution. A product implementation must check the fixed envelope,
retain this exact precondition, and re-evaluate the upstream dependency; passing
the transform is not an independent semantic proof.

## Exact numerical interpretation

Full-tile cells compute increasing-K f32 FMA accumulation from +0; tail cells
compute increasing-K separately rounded f32 multiplication/addition. The
per-GEMM permission permits both. No `fast`, `reassoc`, `nnan`, `ninf`, `nsz`,
`arcp`, `afn`, `contract`, reduced-precision, or flush-to-zero LLVM flag is added.
NaNs/infinities and signed zero remain meaningful; NaN payload bits are not
compared. Each GEMM retains its complete f32 result before a dependent GEMM.

LLVM [FMulAdd](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/docs/LangRef.rst#L17809-L17848)
allows, rather than mandates, hardware fusion. This run explicitly selects
`-march=x86-64-v3` on the already execution-validated local AMD Ryzen AI 9 HX 370;
assembly and a rounding discriminator prove actual FMA here. The runner refuses
to run unless `MDSLC_RESEARCH_RUN_V3=1` is explicitly set. It does not establish
runtime capability gating or a portable product ISA contract.

Do not interpret a different *permitted* result as a source violation: the
full/tail difference is deliberate. Conversely, do not infer that any compiler
rewrite is allowed just because the profile is called reassociate. The oracle
checks this exact chosen realization, not tolerance against arbitrary real
arithmetic, and not the entire allowed-expression family.

## Execution and falsification evidence

Reproduction on 2026-09-09:

```sh
MDSLC_RESEARCH_RUN_V3=1 bash \
  compiler/experiments/mlir_reassociate_register_v1/run.sh \
  /tmp/mdslc-reassociate-register.atT1oW/reproduce
```

| Scope | Actual outcome |
| --- | --- |
| O3 generated object, O2 independent harness | 1,417 cases / 109,632 checks, zero failures |
| O1 generated ASan object, ASan+UBSan harness | Same 1,417 / 109,632, zero failures |
| Deliberately corrupted output, both binaries | Exit 1; 1,715 check failures each |
| Deliberately require strict result, both binaries | Exit 1; 7,749 check failures each |
| Deliberately invalid full-tile input capacity | ASan heap-buffer-overflow in `research_gemm` |
| Full-tile FMA rounding discriminator | Nonzero fused result; same-operation tail cells +0 |

The harness covers 700 deterministic random rectangular/dynamic cases with
ordinary and IEEE-special values; 704 M/N/K boundary/zero combinations; two
dependent GEMMs with lhs/rhs carries; same and partially overlapping input
storage; output canaries; unchanged input/descriptor bytes; odd K endpoints;
and exact-dyadic independent double-oracle cells. Normal-return FP setup and
restoration belong to the **research harness**, not this raw generated leaf.
The production adapter's existing FP/failure/publication behavior was not
replaced or newly validated by these direct calls.

ASan is attached to the generated LLVM functions before object compilation,
not merely to the C++ harness. The successful intentional full-tile OOB control
is required by the runner. No benchmark, hosted CI, independent full review,
installed package, source-authenticated candidate, or new platform gate was run.

Exact emitted identities (reproduced twice):

| Artifact | SHA-256 |
| --- | --- |
| Scheduled MLIR | `69e3120cb4320f53aeb4dc84d0086830d12924c0239c1da94cd159bf114c6923` |
| LLVM IR | `a16ae9e926ff67fa25a50df6aec80b0e22a4bb6b601793d8ee3ce1a1687e5aa8` |
| O3 v3 object, text 6,178 bytes | `453ed371fb64dc7df300b341eb8a08083ed683b3c976f9f7c9008aa496867eb9` |
| O1 v3 ASan object | `1014a1cffd56a43f4ae3e20de1c01b6d65e52a3e2fd29d023eefc3b5b4b25124` |

Entry: `_mlir_ciface_research_gemm`, three standard rank-2 memref descriptor
pointers. No generated binaries or IR outputs are committed.

## Negative alternatives and next boundary

The retained `tiled_contract.mlir` uses masked `[4,8,4]` vectorization. It compiles
and lowers, but this exact pipeline leaves scalarized masked reductions and
does not produce the intended outer-product/register-carried form. The runner
preserves that negative structural observation. An initial `[4,8,1]` masked
experiment obtained SSA retention but simplified away contraction into separate
operations. Neither result justifies private replacement of upstream machinery.
Peeling the parallel dimensions exposes the useful full-tile form directly.

**Next justified boundary:** independently review and measure this frozen
*reassociated primitive* against the best strict row/noalias artifacts in a
coordinated window. Do not weaken the existing strict benchmark oracle to admit
it. Only if that evidence survives should a separately gated generated
reassociate candidate be engineered, with source-profile refusal, capability
requirements, retained private-output/FP guarantees and actual source execution
tests. The current default and strict candidate remain unchanged.

## Independent generated-read sanitizer tightening

The independent review of `caa2364` verified the recorded source/IR/object hashes
and accepted the stated numerical/hoisting envelope, but found that the original
invalid-capacity case faulted on a **4-byte A read** in the generated function.
That is useful scalar-load instrumentation evidence, not evidence for the wide
contiguous B load. The original result is not relabeled as a wide read.

Starting from research head `48adacfac75d8751133fe7a64871661c2fea5cf4`, the
research harness now retains that A case and adds a separate B case: A is a
truthful 4x1 allocation, private C is a truthful 4x8 allocation, and B advertises
1x8 while containing only one f32. The runner requires configured ASan exit 1,
heap-buffer-overflow, the exact READ width, and `research_gemm` as frame #0.
Merely finding the function elsewhere in a trace no longer suffices.

Correctness-only reproduction (no benchmark):

```sh
MDSLC_RESEARCH_RUN_V3=1 bash \
  compiler/experiments/mlir_reassociate_register_v1/run.sh \
  /tmp/mdslc-register-read-controls.IWAEiy
```

Both normal and ASan/UBSan executions again passed **1,417 cases / 109,632
checks**. Corrupted output and strict-result requirements retained their 1,715
and 7,749 failures in both executables. The A adversary produced exit 1 with a
generated top-frame **READ of size 4**; the B adversary separately produced
exit 1 with a generated top-frame **READ of size 32**. Nine permanent classifier
negatives reject wrong exit statuses (0/2/99/134), swapped widths, a WRITE,
an unrelated top-frame owner, and the wrong sanitizer error kind.
Independent read-only review accepted the two-file test delta; it did not run
or rebuild the tests and did not infer source legality from the sanitizer fault.

The scheduled MLIR, LLVM, O3 v3 object and O1 v3 ASan object hashes are unchanged
from the earlier table. Only the research harness/control machinery changed.
No measured artifact was overwritten and no timing series was rerun.

| New artifact in `/tmp/mdslc-register-read-controls.IWAEiy` | SHA256 |
| --- | --- |
| `execution` | `3d1b9f4bf4f6cb2a23864ccb11558489cf46a0084c5d283d348a857235ce68ed` |
| `execution-asan` | `12b8f7ca285dc2c7037f252d9b916c2b0187bb77c4bf50368dd0d958dd95c1a8` |
| `run.log` | `b00bd17431836e5e716c1d318b50b19657b7aae22e439e3a657d2fdbec475376` |
| `asan-a-negative.stderr` | `402e0aeff6239b18550d034f1073043fbe686cf9d3e74f0c10ed5ef8584c1940` |
| `asan-b-negative.stderr` | `2e9e307fd65a725920aa16d392d0cd23ebec6453052d3869bd6d4577eba9a01e` |

Remaining production acceptance gaps are unchanged:

- A compiler-issued `reassociate_f32` candidate and actual strict-source refusal;
  `--require-strict` remains only a numerical oracle discriminator.
- An exact lowering-envelope verifier, not structural grep or hoister success:
  preserve initialized private C, complete M/N bounds, zero-trip input safety,
  no fictitious K padding and per-GEMM f32 boundaries.
- A correct minimum ISA/capability contract and isolated target compilation;
  existing AVX2/FMA detection alone does not establish all x86-64-v3 features.
- Existing FP adaptation, allocation/shape checks, helper/DSO identity and the
  selected adapter's stronger normal-return publication guarantee must survive
  actual source execution and installed-package integration.
- Independent whole-source/storage/failure and capability-refusal gates are
  still needed. These direct-leaf controls establish neither descriptor guarding,
  source authority, general performance superiority nor BLAS parity.
