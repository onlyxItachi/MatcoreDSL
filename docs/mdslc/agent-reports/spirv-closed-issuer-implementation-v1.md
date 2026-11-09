# Closed bounded static strict GEMM SPIR-V issuer v1

Status: implementation and adversarial compile qualification in progress;
**compiler-private issuance only, no source/runtime/candidate authority**.
Canonical starting point is `b2597bed52de46604574f76eb33663788a5ab994`.
This report covers `MatcoreSpirvGemmCandidate.{h,cpp}`, the narrow positive-static
overload in `MatcoreCpuGemmCandidate.{h,cpp}`, and the focused C++ issuer tests.
The integration owner supplies default-off CMake, a private qualification CLI,
SPIRV-Tools/physical fixtures and the complete integration receipts.

## Closed construction and retained numerical law

The issuer accepts only signed `(M,K,N)`. It does not accept a module, shader,
callback, filesystem path, manifest or certificate as construction input.
`buildStrictGemmStagesV1(context, shape)` reuses the original compiler-owned
semantic primitive, verified structured construction, and upstream One-Shot
bufferization. A literal-dimension specialization is checked against the same
canonical GEMM maps, ordered operands, exact destination, positive-zero fill,
and separate strict f32 multiply/add. The existing no-argument dynamic path
remains unchanged; the new overload checks every matrix byte product before
upstream static-type construction.

Upstream Linalg parallel-loop conversion, GPU mapping, outlining and
GPU-to-SPIR-V conversion produce the arithmetic, indexing and control flow.
There is no handwritten GEMM shader. The fixed schedule emits two ordered
dispatches: positive-zero fill, then increasing scalar-K GEMM. Both use local
size `(1,1,1)` and dispatch `(M,N,1)`. Singleton output axes become literal zero;
K=1 becomes the single scalar multiply/add body without a reduction loop.
This canonicalization is explicitly checked rather than excluded from the family.

Production `getEntryPointABIAttr` / `getInterfaceVarABIAttr` APIs provide entry
and descriptor ABI data. In MLIR 21.1.8, `targetWidth=32` emits only
`SignedZeroInfNanPreserve`; the issuer creates the remaining `DenormPreserve`
and `RoundingModeRTE` execution-mode operations on the exact same entry.
The final separate FMul/FAdd results receive `no_contraction` UnitAttrs, which
the production serializer emits as NoContraction. No assembly text is patched,
and no test-only ABI pass is linked into the issuer.
See pinned upstream [ABI lowering](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/SPIRV/Transforms/LowerABIAttributesPass.cpp),
[GPU conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/GPUToSPIRV/GPUToSPIRVPass.cpp)
and [serialization](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Target/SPIRV/Serialization/Serializer.cpp).

## Derived static family, not device availability

Admitted tuples satisfy all of:

- M,K,N are positive; M,N <= 65535.
- K < INT32_MAX, including its next induction value.
- MK, KN and MN each <= 33,554,432 f32 elements (128 MiB).

The last test is division-based after positivity checks, so validation never
forms an overflowing signed product. Each actual linear element index is below
its array extent; even byte extents fit signed i32. Thus row-stride multiply,
index add and the scalar reduction increment cannot wrap. M/N use the guaranteed
Vulkan dispatch-axis bound. The static range uses the guaranteed storage-buffer
limit; three storage descriptors and one invocation per workgroup are below the
baseline guaranteed resource limits. These bounds are independently derived
from [Vulkan limits](https://docs.vulkan.org/spec/latest/chapters/limits.html),
not CUDA/HIP limits or the historical 2x3x4 outputs.

No allocation-success, optional float-control, device implementation or
performance guarantee follows. A later runtime must check the actual selected
device's required f32 controls, buffer/descriptor/dispatch limits, capacities,
isolation, fill-to-GEMM memory dependency and checked completion. The three modes
are requirements on the eventual implementation, not evidence that it supplies
them. The arithmetic law is bounded to scalar f32 FMul/FAdd, not arbitrary dot,
matrix, FMA or extended instructions; see the
[Vulkan SPIR-V precision environment](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision).

## Full identity binding plus independent structural checks

The artifact records semantic, structured, bufferized, outlined-GPU and final
fill/GEMM SPIR-V IR; both exact serialized word vectors; and typed shader ABIs.
The fill shader's binding 0 is C. GEMM bindings 0/1/2 are A/B/C. The ABI fixes
descriptor set 0, exact byte ranges, array stride 4, member offset 0, entry,
local size and dispatch. Its writable flags describe checked shader writes,
not host descriptor access authentication.

The manifest binds every stage, both full binaries and ABI serialization.
`verifyStrictGemmArtifactV1` independently reissues from the checked shape and
compares every artifact field. A caller cannot replace a graph and compensate
with an edited digest. This is inspection-only full replay, not an import or
source/execution certificate. Full word equality includes integer constants,
CFG edges, AccessChain operands, symbol names and every other serialized field;
it is not an opcode-count/float-mode gate.

Supplementary independent validators check the outlined host's dispatch order,
operand binding, unchanged destination return, exact local/grid sizes, strict
scalar DAG, K-loop bounds/step, per-output addresses and absence of other effects.
The final shader checker validates Logical/GLSL450, one GLCompute entry/function,
exact capability/extension/mode sets, storage types/strides/ranges/offsets,
the precise builtin interface symbol, four-block reduction CFG, scalar dataflow
and complete affine linear addresses. Symbolic coefficient arithmetic uses
checked overflow rejection. Native and discardable attributes are allowlisted;
extra alignment/volatile/fast-math permissions, member decorations, imports,
functions, storage and effects fail. The actual serialized output is also
deserialized with the upstream production API and rechecked structurally.

## Focused test design and current evidence boundary

Pure shape tests include positive family faces, zero/negative inputs,
dispatch-axis overflows, each range overflow and extreme signed values.
Compilation cases cover the 2x3x4 control, 1x1x1, rectangular/skinny/tail shapes,
both axis maxima, maximum K range and exact 128 MiB output range. Boundary
compilation does not allocate or execute those matrices.

Mutation tests change individual binary constants, AccessChain operands,
binding/stride, branch edges, strict modes and NoContraction. The first five
retain instruction counts and float modes. Artifact fields are independently
tampered to exercise full replay. Structural tests mutate loop bounds,
operand order, FMA permission, argument attributes, output indexing, storage
layout, load/store memory modifiers and initialization. Mutants claimed as
valid shader controls must pass `mlir::verify` before their expected refusal.

The first integrated compile exposed a signed-to-unsigned literal narrowing;
the integration owner fixes it after the positive-shape check. Initial control
issuance exposed the upstream GPU function's `sym_name` discardable attribute;
the allowlist now preserves that authenticated name. A subsequent compile
exposed the pinned GPU function `getArgAttrs` overload difference; the integration
owner adapts that API without relaxing the no-extra-argument-permissions check.
Independent review also identified valid Aligned8 loads and RelaxedPrecision
member decorations that discardable-only checks miss; native-property and member
decoration checks now reject them. These failures are recorded, not passing
qualification claims.

Final compile/unit/external/physical receipts will be recorded by the integration
owner after the complete current implementation is compiled and executed where
required. This implementation lane has run no physical workload, system package
mutation, LLVM build, source-driver route, runtime adapter or performance test.

## Intentionally unsupported subsequent boundaries

Dynamic/zero shapes are refused by this issuer without changing their existing
source mathematics. There is no source candidate, auto dispatch, residency,
fusion, installed shader import, public API or performance milestone. Source
connection still needs an explicit finite-catalog-versus-source-time artifact
lifecycle decision, unchanged Program/witness/site/profile binding, original
Read/extent/effect/failure frontiers, selected ICD/layer-chain authentication,
joined-worker completion/quarantine/poison and relocated-install qualification.
PR #84 research, existing CPU/GPU runtime routes and frontend remain separate.
