# Parameterized strict GEMM output pattern v1

Starting canonical main: `4c222dd68176de19fcf82696939b74be73f80968`.
Status: normally merged through [PR #91](https://github.com/onlyxItachi/MatcoreDSL/pull/91),
canonical engineering checkpoint `c09a9d0e4384441238bf169e7e1122bf13ea5e23`
from exact premerge head `4fde88b8c3e6b09bc88ce466a0bd28d9dafe6600`.

## Architectural verdict

The existing architecture supplies the right boundaries, but not a general HOW
implementation. This increment completes one reusable decomposition boundary;
it does not rename the existing collection of candidates into a general optimizer.

| Existing component | Observed scope at the starting checkpoint |
| --- | --- |
| Clang admission, immutable Program and paired MLIR witness | Source authentication, shape/numerical facts and ordered effects; whole-region witness is untransformed. |
| Matcore semantic dialect and contraction model | Operation contracts and topology; not execution authority for arbitrary contractions. |
| Built-in strict GEMM stages | Shared semantic -> Tensor/Linalg -> checked One-Shot destination path, already reused by CPU and GPU issuers. |
| CPU Transform schedules | A fixed M/K/N interchange and separately bounded reassociating/vector and fused-pair recipes. |
| GPU realization | Exact one-output-per-block scalar GEMM and separate qualified pair recipes; launch and completion laws are pinned. |
| Candidate selection | Trusted implementation identities, capability/profile checks and artifact ownership; not a general measured schedule search. |
| MLIR/LLVM | Structured transformation and machine-lowering machinery already present; no replacement backend is needed. |

The smallest useful extension is a parameterized **output-domain decomposition**
of the same GEMM. A single upstream Transform recipe accepts M/N tile sizes,
constructs the tiled IR and interchanges the local traversal to M/K/N. The
numerical reduction is not tiled. This is neither algebraic equivalence search
nor a collection of manually written arithmetic kernels.

## Closed pattern and realization contract

`gemm_pattern::OutputTilePatternV1` carries two private HOW parameters, not source
tensor dimensions. Each lies in `[1,64]`; non-power-of-two and rectangular tiles
are deliberately included. This is a bounded initial qualification envelope,
not a hardware limit, a cost model or a claim that these are optimal tile sizes.
Runtime M/N/K remain dynamic and include zero. Upstream subviews bound tails;
no padding, speculative out-of-bounds arithmetic or masked extra reduction is
introduced. The original positive-zero fill covers the complete destination,
including K=0, outside the tiled contraction.

At the private raw-leaf boundary, signed tile-loop increments must remain
representable: `M <= INT64_MAX-(tile_m-1)` and likewise for N. The unchanged
source adapter proves this for nonempty results through its checked output-byte
extent bound, and bypasses generated invocation for empty results or K=0.
Consequently no new source-size restriction, guard or failure frontier is added.

The issuer retains the original semantic, structured and bufferized witnesses.
The Transform program runs only on a verified clone. Parameter-bound replay and
structural postconditions defend the generated graph; failed candidates are
discarded. Per output, the full scalar K sequence is increasing with one
unflagged f32 multiply then add. No new allocation, copy, external call, input
no-alias assumption, publication or failure frontier belongs to this pattern.

The existing CPU issuer specializes the result for its closed target/ISA set;
the same parameters have identical pre-machine IR across target choices. The
LLVM backend owns instruction selection and independent-lane vectorization.
An ISA feature string is not numerical permission, hardware qualification or a
performance result. Different ISA objects still require actual hardware/OS
guards before execution.

There is no new source grammar, dialect, optimizer JSON, certificate framework,
public API or executable IR importer. The issuer takes only checked parameters
and constructs the canonical built-in primitive internally. Inspection verifiers
do not admit caller-supplied IR for execution. Source admission, sealed Program,
ordered runtime checks/effects and private generated-code ownership stay intact.

## Selection and artifact lifecycle

The existing private issuer gains `--output-tiles=M,N`, mutually exclusive with
its old row-contiguous and reassociating schedules. Exact parameters, Transform,
scheduled payload, target and final LLVM identities are recorded with the issued
artifact. Only the requested instance is generated; the compiler does not
enumerate a matrix-size catalogue.

The explicit internal build selection is:

```text
-DMDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE=output-tiled
-DMDSLC_EXPERIMENTAL_CPU_OUTPUT_TILE_M=3
-DMDSLC_EXPERIMENTAL_CPU_OUTPUT_TILE_N=5
```

This selects the existing generated-strict primitive in the build's private
candidate library. Default scalar and existing ISA/native/provider/GPU routes
are unchanged. The public source driver does not accept arbitrary schedules or
select tiles automatically. Production issue-at-build-time remains: source-time
on-demand specialization is still a subsequent artifact-lifecycle boundary.

## Acceptance and falsification

Required evidence is separate for:

- identical original witnesses and different derived IR for multiple tile pairs;
- exact parameter/target binding and rejection of malformed or unsupported choices;
- mutation rejection for loop bounds/steps, indexing/subviews, arithmetic,
  destination, extra effects/attributes and a mismatched parameter tuple;
- normal and generated-code sanitizer execution of rectangular, skinny, tail,
  zero-M/N/K, input-alias, cancellation/order/FMA, subnormal and non-finite cases;
- existing authenticated source execution, late reads, publication/observation,
  failure prefixes, provider coexistence and relocated installed consumers;
- unchanged default artifacts and regression paths; hosted target evidence is
  distinct from local execution and hardware-unavailable skips.

Local qualification on coherent Linux x64 21.1.8:

- New focused suite: **32/32 passed, zero skips**. Issuer: 242 checks,
  22 well-formed IR corruptions rejected. CLI: 17 refusals and four accepted
  parameter sets. Each of six normal/ASan leaf profiles executed 269 cases;
  all deliberate corruption and generated-code ASan controls passed.
- Existing generated CPU and candidate regression surface, including the
  source/effect oracles and extended huge-empty/overflow guards: **120/120
  passed, zero skips**. This build selected the actual `3x5` production leaf,
  with authenticated OpenBLAS 0.3.32 enabled and GPU issuers compiled.
- Retained scalar, row-contiguous, AVX, AVX2, AVX512F, reassociating and AArch64
  baseline issuers: all **49** LLVM/MLIR/manifest artifacts byte-identical to
  the prior clean `9f7f9af` issuer (its relevant sources match starting main).
- Object inspection observed separate packed multiply/add using YMM for the
  `4x8` AVX2 test artifact and ZMM for `4x16` AVX512F. This is instruction and
  correctness evidence, not a speed comparison or an automatic tile policy.
- Repository hygiene and `git diff --check` passed.
- Separate configure controls rejected seven malformed/out-of-range values on
  each axis at the exact parameter diagnostic (14 refusals); bounds 1 and 64
  configured successfully on both axes with `BUILD_TESTING=OFF` (four accepts).

The full clean standalone suite passed **299/299 with zero skips**, including
the production 3x5 source/build-inaccessible installed route. All **26 hosted
engineering checks succeeded**, including native ARM64 tiled source qualification
(126/126, zero skips), Release/Debug, focused sanitizers, Windows and legacy CI.
Some x86 runners skipped unavailable AVX512 tests; they executed locally and
on other hosted workers. Exact counts, run/head identities, skip sets and the
28 unchanged GPU issuer artifacts are in the
[merged qualification record](agent-reports/parameterized-pattern-qualification-checkpoint-v1.md).
The
[independent fixtures report](agent-reports/parameterized-pattern-adversarial-tests-v1.md)
and [implementation report](agent-reports/parameterized-gemm-output-pattern-implementation-v1.md)
record the bounded contracts and retained construction/test failures.
An [independent core audit](agent-reports/parameterized-pattern-independent-core-audit-v1.md)
accepted immutable implementation `a5d94e9` and rechecked final wiring `4fde88b`
within this bounded contract, with no unresolved correctness or authority blocker;
it did not substitute for tests.

## Deliberate exclusions and next boundaries

No speed, optimal schedule, BLAS parity, matrix-instruction, padding, K tiling,
general fusion, target-residency or cross-accelerator performance claim follows.
The earlier 4x64x32 cache research lost its sampled comparisons; this new
output-only parameterization is a correctness/reuse increment, not reversal of
that performance evidence. See [the retained experiment disposition](ROW_CONTIGUOUS_CPU_SCHEDULE_V1.md#exact-integration-and-subsequent-experiment-disposition).

GPU use of this output decomposition needs its own thread/block mapping,
grid/ABI binding, target limits and physical/completion qualification. Existing
NVVM/ROCDL paths are not silently retargeted. [SPIR-V Issue #89](https://github.com/onlyxItachi/MatcoreDSL/issues/89)
and draft PR #84 remain independent; this increment has no SPIR-V dependency.

The next pattern-specific boundary is source-time selection and ownership of a
bounded requested instance, reusing the same issuer and existing private-artifact
authentication. No cost model or expanded numerical permission is implied.
Its acceptance must bind requested parameters, target/toolchain and source closure
to the private artifact; reject substituted/stale instances or forged submitted IR;
and retain tail/numerical/source-effect and relocated-consumer controls. Generate
only requested instances, without turning shape sizes into a kernel catalogue.
Broader operations, GPU mapping and schedule search remain separately reviewable.
External compiler interoperability is bounded by the
[official-interface audit](COMPILER_INTEROP_BOUNDARIES_V1.md), not by shared use
of MLIR alone. In particular StableHLO dot cannot silently replace strict GEMM.
