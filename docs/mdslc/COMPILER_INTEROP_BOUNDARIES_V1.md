# Compiler pattern and interchange boundaries v1

Upstream documentation checked: 2026-10-11. Repository audit base:
`4c222dd68176de19fcf82696939b74be73f80968`.

Status: internal architecture/research boundary, not an interop implementation,
installed format, public ABI, or target qualification. No StableHLO, JAX, XLA,
Cerebras SDK, or new runtime dependency is introduced by this record.

## Ownership and the first pattern boundary

Keep the existing LLVM/MLIR implementation. Matcore owns authenticated semantic
admission, numerical/effect legality, retained facts, candidate selection and
checked orchestration. Upstream MLIR owns structured transformations; LLVM and
target backends own machine lowering. A pattern is a compiler-private HOW
proposal, not a second semantic IR, executable authority, or serialized plugin.

The selected first boundary is CPU-first, checked parameterized M/N output
tiling of the existing isolated GEMM issuer through upstream Transform/Linalg.
K remains unpartitioned; the original positive-zero accumulator, increasing-K
order and separate f32 multiply/add stay intact. Target/ISA realization is a
separate specialization of that proposal, using the existing issuer/runtime
checks. GPU tiling and SPIR-V are deferred. This selection does not establish
new performance, physical-target or external-consumer support.

Patterns must distinguish semantic preconditions, numerical permissions,
schedule parameters, structural postconditions, and target/runtime requirements.
Neither a tile parameter nor a successful transformation discharges any of the
other categories. Retain the unchanged semantic/source witness and discard a
failed transformed clone. Existing source, failure, private-output and artifact
ownership contracts remain in
[REGION_COMPILER_V1.md](REGION_COMPILER_V1.md),
[ROW_CONTIGUOUS_CPU_SCHEDULE_V1.md](ROW_CONTIGUOUS_CPU_SCHEDULE_V1.md), and
[STAGED_GPU_CANDIDATES_V1.md](STAGED_GPU_CANDIDATES_V1.md).

## What coherent MLIR 21.1.8 already supplies

The pinned upstream implementation provides generalization/interchange,
`structured.tile_using_for` with static or parameter/payload-computed sizes,
complementary `structured.split`, packing and tile/fuse operations. A zero tile
size means no tiling; all-zero sizes can therefore succeed without changing
the payload. Split-reduction and reduction-tiling operations explicitly create
partial reductions and a merge: they are arithmetic-changing choices, not
strict schedule aliases. Vectorization can succeed without vectorizing any op.
These are reusable mechanisms, not Matcore legality certificates.
[MLIR 21.1.8 Linalg Transform definitions](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td)

Upstream SCF peeling supplies main/remainder loops.
[MLIR 21.1.8 SCF Transform definitions](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td)
Matcore still must check complete domain coverage, bounds/tails, destination
identity, accumulator flow, exact numerical body, no added effects/allocation,
and stage/artifact identity. Future K tiling requires an ordered carry proof;
split-K/partial sums require adequate explicit reassociation permission. Padding
must not invent extra observable FP operations or reads: a zero-padded product
is not harmless for every non-finite input. No private replacement tiler,
bufferizer, vectorizer or machine backend follows from these needs.

## Actual external interchange surfaces

| Upstream surface | Documented boundary | Boundary it does not supply |
| --- | --- | --- |
| StableHLO | Specified ops plus compatibility serialization/deserialization into a targeted portable artifact. | Arbitrary `mdsl`, Linalg, Transform, LLVM or vendor-dialect acceptance. |
| XLA / PJRT | Program submission, compilation and runtime device APIs; XLA imports its supported HLO-family representation. | Semantic understanding of Matcore contracts merely because the program format is named `mlir`. |
| JAX export / FFI | Exported-JAX artifact roundtrip; separately registered external-operation handlers. | A plain foreign MLIR file automatically becoming a callable, differentiable, authenticated JAX program. |
| Cerebras SDK / training | CSL compiler/runtime lane; separately a Cerebras PyTorch model lane. | A publicly documented arbitrary Matcore/MLIR/StableHLO importer. |

StableHLO compatibility applies to artifacts produced by its portable APIs, not
ordinary textual dumps or arbitrary MLIR bytecode. Its stated windows are five
years backward and two years forward, the latter excluding newly introduced
features. Unregistered attributes, numerical accuracy and libStablehlo source
API compatibility are outside those guarantees. Retaining `mdsl.*` attributes
on an external module is therefore not a portable semantic preservation law.
[StableHLO compatibility](https://openxla.org/stablehlo/compatibility)

PJRT's `mlir` program format describes a container, not an arbitrary-dialect
lowering contract. The examined XLA parser registers HLO-family/support dialects,
upgrades versioned StableHLO, performs its CHLO/Shardy handling and converts to
HLO. It does not register or lower Matcore operations. That source observation
is revision-sensitive, not a stable extension promise.
[PJRT program definition](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h),
[XLA MLIR-to-HLO implementation](https://github.com/openxla/xla/blob/main/xla/pjrt/mlir_to_hlo.cc)

JAX's documented roundtrip is `export` -> `Exported.serialize()` ->
`deserialize()` -> `.call()`. The artifact includes more than StableHLO:
trees, shapes/dtypes, platforms, sharding/device metadata, effects and calling
convention. Main/inner signatures can include platform indexes, dimension
arguments and ordered-effect tokens. JAX's stated compatibility is six months
backward/three weeks forward; a direct `compiler_ir()` dump is outside it.
Custom-call compatibility is separately allowlisted; disabling a safety check
does not confer compatibility. Deserialized bytes are trusted executable input.
[JAX export contract](https://docs.jax.dev/en/latest/export/export.html),
[Exported metadata API](https://docs.jax.dev/en/latest/jax.export.html)

For external code, the documented JAX route is registered FFI targets, not
arbitrary-dialect import. Typed FFI uses custom-call API version 4, explicit
result metadata, layout/alias choices and side-effect indication. A side-effect
flag ensures retention even if results are unused; it alone does not establish
Matcore's ordered failure/publication law. Differentiation needs separate rules.
[JAX FFI call API](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.ffi_call.html),
[JAX FFI guide](https://docs.jax.dev/en/latest/ffi.html)

## Exact numerical, effect and runtime gaps

StableHLO `dot_general` defines its sum through `reduce`, whose order is
implementation-defined. `precision_config` is explicitly underspecified;
`DotAlgorithm` describes input/accumulation precision and algorithm properties,
not increasing-K separate multiply/add rounding. Unsupported algorithms should
error, not silently substitute another. Consequently neither `HIGHEST` nor an
f32 accumulation algorithm is a lossless carrier for Matcore strict GEMM.
StableHLO tokens impose dataflow ordering, while `custom_call` semantics and its
metadata remain implementation-defined.
[StableHLO dot/reduce/custom-call/execution specification](https://openxla.org/stablehlo/spec)

This is an architectural inference, not an execution result: a future ordinary
dot projection needs an explicitly compatible numerical profile and consumer
capability contract; strict export must reject that projection unless a different
representation and its consumer are separately proved. Algebraic equality or
matching storage dtype is insufficient. Source reassociation permission also
does not authorize reduced input precision, imprecise accumulation, altered
subnormal behavior or cross-operation reassociation by implication.

XLA custom calls require separately registered implementations, platform/ABI,
layout, argument/result and error contracts. The official FFI guide still labels
its ABI experimental; GPU handlers enqueue work on the supplied platform stream.
A handler return is not automatically Matcore's checked device completion.
[XLA custom calls](https://openxla.org/xla/custom_call)
Likewise compile-time input/output aliasing and runtime donation are distinct;
without donation XLA may allocate/copy. Neither alias metadata nor donation is
Matcore's caller-storage/no-hidden-copy proof.
[XLA aliasing](https://openxla.org/xla/aliasing)

Matcore's descriptor checks, all-MAY-alias host resources, source-ordered failure
frontiers, late reads, retained successful publications, owning observations,
FP-state policy, unknown-completion quarantine, frozen source closure and
compiler-private implementation ownership are not supplied by generic tensor
dataflow, token presence, custom-call flags or portable serialization. A future
runtime adapter needs a separate proof of those obligations; an imported module
must not forge Clang/source authentication or inherit serialized authority.

## Cerebras: vendor lane, not a shared-MLIR shortcut

The current official SDK documents `cslc <csl_filename>` (normally a layout
program), WSE-2/WSE-3/fabric/parameter choices and output ELF artifacts. Its
`--import-path` resolves CSL modules, not a general IR importer.
[CSL compiler](https://sdk.cerebras.ai/csl/csl-compiler)
`SdkRuntime` loads the compiled artifacts and performs explicit transfers and
launches, with copy/streaming modes, per-PE layout and blocking/nonblocking calls.
Nonblocking return can precede operation start; memcpy infrastructure reserves
fabric resources and cannot detect every resource conflict.
[SDK host runtime and streaming](https://sdk.cerebras.ai/tensor-streaming)
Appliance compilation submits CSL sources through `SdkCompiler` and returns an
artifact path for the cluster runtime.
[SDK appliance mode](https://sdk.cerebras.ai/appliance-mode)

The separate current training documentation ports PyTorch model/dataloader/YAML
programs through Model Zoo. Its AutoGen page describes vendor-generated kernels
and loss decorators, not user-supplied arbitrary IR; historical release notes
also bound availability. Neither page establishes a StableHLO importer or the
Matcore strict profile.
[Cerebras PyTorch porting](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/migration/porting-pytorch-models-to-cerebras),
[Cerebras AutoGen](https://training-docs.cerebras.ai/rel-2.5.0/fundamentals/kernel-autogeneration-with-autogen)
This audit found no documented direct interchange in those surfaces; it does
not claim a vendor-private route cannot exist. Any future vendor realization
needs source/kernel generation or an explicitly documented vendor ingestion
contract plus placement, transfers, numerical and completion/error qualification.
SDK simulation would not substitute for physical WSE execution evidence.

## Future explicit projection gate

Generic HOW can later support a bounded export only through an explicit verifier
that accounts for every semantic fact: represented structurally, enforced by an
authenticated adapter, retained as an independently checked obligation, or
rejected as unsupported. Name the accepted op/type/shape/profile/effect subset,
producer/consumer versions, custom-call targets and runtime ownership. Separate
portable math inspection from execution admission; imported facts cannot issue
a Matcore plan or waive runtime checks.

The first useful external experiment would be a pure value-level subset with
loss diagnostics, independent differential controls and zero execution authority.
Even then, successful parse/roundtrip/compile is not source authentication,
strict numerical parity, link/runtime support or performance. This document
does not freeze an export schema, choose an adapter, or authorize an integration.
