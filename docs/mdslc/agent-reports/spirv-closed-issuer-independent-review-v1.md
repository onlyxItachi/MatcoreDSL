# Independent closed static SPIR-V issuer review v1

Scope: Issue #89's compiler-private bounded-static positive strict-f32 GEMM
issuer. Canonical starting point was verified as
`b2597bed52de46604574f76eb33663788a5ab994`. Research PR #84 remained draft at
`5dc14db686b6b2668a2d3a47f5ef24c0e2bb4b97` during this review. Full repository
guidance, `CURRENT_STATE.md`, Issue #89 and its comments, the research specimen,
and the source-frontier audit were read before reviewing implementation.

The final source review includes issuer-lane hardening through
`82bd1db14bd106b5816289610402adf86c34612c` and the canonical BuiltIn-decoration
fix `8c16d6ac39dbc6935b0a17552297e75aac886b35`. Those are local engineering
identities, not claims that canonical main already contains this work.

## Disposition and evidence limits

No remaining high/medium numerical or ABI finding was identified by source
inspection at that checkpoint. Acceptance still depends on the integration
owner's actual compiled tests, upstream SPIRV-Tools validation and separately
labelled physical/refusal receipts. This reviewer did not start an LLVM or MDSLC
build, dispatch a shader, mutate a device or system package, or execute a source
candidate. No performance, universal-target, installed-consumer, source/runtime
authority or general imported-IR verification claim follows.

Exact artifact replay binds every stage string, full word vector, shape, manifest
and typed ABI. It does not independently prove the correctness of a shared issuer
bug. The supplementary graph/ABI checks and independent arithmetic controls are
therefore material: replay is not substituted for them.

## Concrete falsifiers and resulting hardening

The initial supplementary classifiers inspected discardable attributes but
missed some native properties and type-member decorations. These were not
artifact-import or execution-authority bypasses: complete replay still rejected
changed artifact fields. They were gaps in the claimed independent envelope.

| Validity-preserving change | Why it matters | Reviewed response |
| --- | --- | --- |
| Add `Aligned 8` to a scalar f32 load/store | The unchanged index may name an odd f32 element; the ABI grants no such alignment promise | Final native attribute allowlists reject memory-access/alignment modifiers |
| Add buffer-member `RelaxedPrecision` | Arithmetic opcode counts and NoContraction remain unchanged, but precision permission changes | The checker explicitly rejects extra member decorations |
| Replace the entry interface's WorkgroupId reference with a storage variable | Generic MLIR verification checks that a referenced global exists, not the exact required interface | The checker binds the actual WorkgroupId symbol, not only interface count |
| Add `gpu.block_id x upper_bound 1` while dispatch M is 2 | This explicit bound overrides contextual bounds and makes a legal dispatched block undefined | The outlined checker now rejects native upper bounds |
| Set `memref.load` nontemporal true | A native property escaped the former discardable-only check | The outlined checker now requires the closed false-valued property |

Independently executed controls used only the existing pinned tools and streaming
standard input/output; no generated file replaced a production artifact:

- The historical strict 2x3x4 assembly with an Aligned8 f32 load assembled and
  passed `spirv-val --target-env vulkan1.2`.
- The same assembly with member-0 RelaxedPrecision on A's struct assembled and
  passed that upstream Vulkan validation.
- Pinned MLIR 21.1.8 accepted the outlined upper-bound-1 mutant. The precise
  undefined-behavior rule is in the pinned
  [GPU operation definition](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td).

Thus generic validity is not being counted as strict closed-graph correctness.
The independently authored
[`adversarial_test.cpp`](../../../compiler/tests/generated_spirv/adversarial_test.cpp)
first checks unchanged MLIR operation inventory, generic validity and (for final
shaders) production serialization, then calls the structural classifier directly.
It does not obtain rejection merely from edited strings disagreeing with replay.
The foreign-interface control is MLIR-valid but intentionally not Vulkan-valid;
the optional exports contain only the other three expected Vulkan-valid shaders.
Test-only commits are `52e3b14ce354c8cdcdd718d0cd1d145fda59a6c9` and
`de5e392dd88fa2ad9432ee1347875c40e1efdfea`; compilation/execution belongs to the
integration owner's receipts, not this review's independently executed evidence.

## Independent shape and numerical reasoning

Let L=33,554,432. The division-based positive-family checks imply MK, KN and MN
are at most L before a product is formed. For dispatched x<M, y<N and scalar
k<K, the complete addresses satisfy:

```text
A: x*K+k <= M*K-1 < L
B: k*N+y <= K*N-1 < L
C: x*N+y <= M*N-1 < L
```

Consequently element addressing and four-byte ranges fit signed i32; K itself
is at most L, so the next induction value cannot overflow. M/N at most 65535
match the guaranteed Vulkan dispatch-axis minimum; each range at most 128 MiB
matches the guaranteed storage-buffer minimum. These are issuance bounds, not
allocation success, optional float-property support or a physical-work budget.
An admitted 5792x5792x5792 instance is already about 194 billion multiply/add
pairs; no claim that all admitted tuples were physically run is justified.

Independent pinned-tool scheduling controls confirmed all three graph forms:
1x1x1 uses zero indices with no K loop; 1x3x4 removes only x; 2x3x1 removes only
y. The checker must accept these proven forms, not require a fixed 2x3x4 graph.

An integer/rational software binary32 calculation, using no host floating-point
arithmetic, independently established the following anchors. Inputs/results are
hex binary32; B is all 3f800000 unless explicitly given; every reduction starts
at positive zero.

| Control | A; exceptional B | Strict result / counterfactual |
| --- | --- | --- |
| FMA | bf800000,3f800001,0; B=3f800000,3f7ffffe,3f800000 | 00000000; fused second step a8800000 |
| K order | 4b800000,3f800000,cb800000 | 00000000; reverse 3f800000 |
| Overflow/order | 7f7fffff,7f7fffff,ff7fffff | 7f800000; regrouping can return 7f7fffff |
| Subnormal input | 00000001,0,0 | 00000001; negative input gives 80000001 |
| Inexact underflow/RNE tie | 00800000,0,0; first B=3f000003 | 00400002; RTZ gives 00400001 |
| Positive seed | all A=80000000 | 00000000; negative-zero seed remains 80000000 |
| RTE versus RTZ | 3f800000,33c00000,0 | 3f800001 |
| Tie-to-even high | 3f800001,33800000,0 | 3f800002 |

Inf*0, opposed infinities and quiet NaN require NaN classification, not payload
identity. Signed-zero padding must remain negative-zero products for the seed
discriminator; trailing positive-zero products would erase a wrong negative seed.
Last-K-only and nonuniform rectangular controls are necessary because repeated
dot witnesses alone can mask truncated reductions or address permutations.

The APFloat oracle and Vulkan probe were also read in full. Their arithmetic
oracle is independent of MLIR lowering and device FP state, though from the same
LLVM project; fixed analytic anchors are the second arithmetic oracle. Expected
outputs freeze before Vulkan loading. Integer-prefix range checks, distinct
NaN/non-NaN output sentinels, immutable inputs, canaries and ordered coherent
barriers/fenced completion are separate controls. Source inspection of those
controls is not a claim of device execution or reusable adapter failure proof.

Pinned upstream [ABI lowering](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/SPIRV/Transforms/LowerABIAttributesPass.cpp)
and [serialization](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Target/SPIRV/Serialization/Serializer.cpp)
support the native metadata route. The
[Vulkan precision rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision)
support the narrow scalar f32 multiply/add claim under the required controls,
not arbitrary dot/matrix/extended-instruction substitution. Optional properties,
actual ICD/layer identity and source adapter/artifact/completion laws remain
separately owned subsequent boundaries.
