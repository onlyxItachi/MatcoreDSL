# MDSLC Legacy Reuse Map

## Purpose and scope

This document records how the current Python/JIT/native-extension implementation
relates to the additive standalone MDSLC compiler. It is based on repository
state `351075e4d8af1880330b7c0474d701ca76776dfa` on
`feature/device-resident-tensors`.

The existing implementation is evidence and future backend substrate. It is not
the source-language frontend, bootstrap IR, stable C ABI, driver, or build
layout for MDSLC. The standalone path must preserve the legacy implementation
without taking a Python, nanobind, JIT, CUDA, or MLIR dependency in its CPU v0
compiler/runtime core.

The repository rules under `rules/` continue to govern the legacy build and
backend routes. In particular, they preserve explicit target routes and forbid
silent CPU fallback. The standalone compiler is intentionally isolated so that
the legacy Python bridge and root MLIR 18 build requirements do not leak into
`compiler/`.

## Classification vocabulary

- **Reuse unchanged**: suitable for direct use without changing its contract.
- **Wrap behind a new interface**: valuable implementation that must be hidden
  behind a standalone contract.
- **Adapt later**: useful substrate after semantic gaps are closed.
- **Conceptually useful only**: preserve the idea, not the current code/API.
- **Legacy-specific**: belongs to the Python/JIT product path.
- **Dangerous to couple into the frontend**: would make Clang capture depend on
  runtime, MLIR, Python, or target-specific machinery.
- **Delete never / preserve for compatibility**: retain unless a separately
  approved legacy migration proves it can be retired.

## Component map

| Component | Classification | Evidence and disposition |
| --- | --- | --- |
| `context.md` | Legacy-specific; conceptually useful only; delete never | Describes `@mc.jit`, RegionV1, packed BlockAttnRes, f32, and an NVIDIA-only JIT route. Preserve as historical implementation context, not as the standalone architecture contract. |
| Root `CMakeLists.txt` | Legacy-specific; dangerous to couple; delete never | Builds only the nanobind `_matcore_native` module, requires Python, nanobind, and MLIR 18.1.3, and writes the extension into `matcore/`. MDSLC must configure independently with `cmake -S compiler ...`; do not make the root build a prerequisite for v0. |
| `include/matcore/kernel_ir.h` | Adapt later through an explicit conversion boundary; delete never | Contains useful target, dtype, tensor, graph, and region concepts. It combines LinearV1, GraphV2, and RegionV1 in one C++ aggregate and lacks source locations, explicit memory spaces, mutability, alias/effect rules, policy/fallback, synchronization, and capability requirements. Its C++ containers and variants are not a stable C ABI or the JSON v0 schema. |
| `matcore/frontend.py` and `matcore/graph.py` | Legacy-specific; conceptually useful only; dangerous to couple | Python AST/tracer capture produces dictionaries for `@mc.kernel`, `@mc.fused`, and `@mc.jit`. It is not canonical Clang declaration recognition and must not participate in `.mdsl` parsing or rewriting. Preserve its explicit-rejection and residency semantics as test ideas. |
| `src/bindings.cpp` | Legacy-specific; dangerous to couple; delete never | A nanobind bridge that parses Python dictionaries and NumPy/DeviceTensor objects, retains Python objects for pointer lifetime, and releases the GIL before JIT work. Its parsers are not reusable as a C++ frontend. Dtype, shape, contiguity, and mixed-residency validation are useful behavior to restate in the new verifier/runtime. |
| `include/matcore/device_buffer.h` and Python `DeviceTensor` | Conceptually useful ownership model; adapt later; delete never | `DeviceBufferHandle` uses pointer, byte size, and monotonic `alloc_id` to reject stale/double-free handles. Upload, download, and zero operations are synchronous. The ownership-token idea is reusable, but the CUDA memory-pool implementation and Python lifetime wrapper are not part of CPU v0. |
| `src/mlir_engine.cpp` | Wrap behind a future Matcore-IR backend interface; dangerous to couple into frontend | Builds MLIR directly from legacy `KernelIR` plus runtime tensor metadata and immediately selects/lower routes. It uses unknown MLIR locations and mixes construction, validation, planning, and runtime-shaped specialization. It can become downstream substrate only after a verified conversion boundary exists. |
| `src/jit_runner.cpp` | Legacy-specific JIT orchestration; dangerous to couple; delete never | Owns MLIR `ExecutionEngine`, memory/disk caches, `dlopen`, runtime symbol registration, external shared-object linking, output zeroing, and CUDA graph resources. It is not the normal `.mdsl -> .o` driver and must not be invoked by `mdslc++` v0. |
| `src/executor.cpp` | Conceptually useful ABI evidence; adapt later | Implements MLIR ranked-memref descriptors, dtype dispatch, packed invocation, ranks 1-4, and shared-library dispatch for up to 16 tensors. It depends on MLIR C-interface conventions, C++ variants/exceptions, and function-pointer casts. It is not the public status-returning C ABI. |
| `src/cache_manager.cpp` and `src/cache_manager.h` | Adapt later; exclude from v0 frontend | Provides explicit cache versioning and structure-aware hashing for JIT `.so` artifacts. The cache is extension-location-dependent, runtime-shaped, and not an AOT object cache contract. Preserve versioning/determinism ideas only. |
| `src/runtime_capabilities.cpp` | Wrap behind a future capability provider | Useful fail-closed behavior: CPU feature discovery, lazy CUDA/HIP probing, compute capability, device/runtime presence, and a process-sticky NVIDIA/AMD backend claim. Its model is too shallow for MDSLC planning and its subprocess probes (`nvidia-smi`, `rocm-smi`, `rocminfo`, `dmesg`) must not run in the Clang frontend. |
| `src/target_registry.cpp` | Wrap/adapt later | Useful canonical parsing and execution requirements for x86 tiers, NVIDIA SM, AMD gfx, and NPU routes. It is an eligibility layer, not a capability record, cost model, or library-versus-generated implementation planner. |
| `src/cpu_lowering.cpp` | Adapt later as downstream MLIR substrate | Contains f32 matmul tiling/vectorization plus a generic loop-to-LLVM route and requests MLIR C wrappers. It is coupled to MLIR memrefs/JIT invocation and has no library-call planner. CPU reference GEMM v0 remains a separate stable runtime implementation. |
| `src/gpu_nvvm_lowering.cpp` | Adapt later; exclude from CPU/bootstrap milestones | Captures valuable NVIDIA transform and NVVM pass-order knowledge. The pipeline is deliberately ordered around host/device type conversion and binary serialization. It is generated-kernel infrastructure, not a cuBLAS library backend and not a v0 requirement. |
| `src/gpu_amd_lowering.cpp` | Future/adapt later; never claim from dialect presence alone | Has a small gfx capability table, dtype legality checks, and a ROCDL/HIP lowering sequence. It defaults an unspecified chip to `gfx90a` and stages host memrefs. It requires independent compile/link/execute validation before being described as supported. |
| `src/lowering_pipeline.cpp` | Conceptually valuable; dangerous monolith; bridge later | Owns route selection, extensive custom NVIDIA scheduling/MMA passes, Region/Fusion special routes, pass diagnostics, and target binary packaging. It contains hidden host-device staging and target-specific assumptions, so it must remain downstream of explicit residency/legalization checks. |
| `src/region_verifier.cpp` | Best verifier substrate to adapt later | Checks IDs, producers, topological order, inputs/outputs, dtype/rank/static shapes, BlockAttnRes attrs, and runtime descriptor agreement. It is fixed to RegionV1, throws C++ exceptions, and has no source location, alias/effect, policy, or fallback checks. Reuse algorithms, not the API. |
| `src/region_emitter.cpp` and `src/region_emitter_block_attn_res.cpp` | Legacy-specific NVIDIA emitter; preserve | Accepts a single RegionV1 op, rejects multi-op lowering, emits a direct f32 NVIDIA `gpu.launch`, and assigns unknown source locations. It is evidence for future explicit-launch backends, not the GEMM bootstrap path. |
| `tests/` | Reuse unchanged as a regression surface; add a separate standalone suite | Existing tests exercise Python tracing, JIT, Region verification, dtype/shape checks, device residency, GPU kernels, and benchmarks. They contain no `.mdsl`, Clang/Sema, rewrite, normal object, install, consumer-CMake, or source-diagnostic coverage. Do not repurpose them as the MDSLC acceptance suite. |

No named production component is safe to reuse unchanged inside the standalone
frontend or C ABI runtime. “Reuse unchanged” applies to preserving the legacy
code and regression surface while new boundaries are introduced additively.

## Python, nanobind, and JIT assumptions

The current execution contract begins with Python capture:

1. Python decorators and tracer objects build dictionary IR.
2. `bindings.cpp` converts Python objects into C++ `KernelIR` and
   `RuntimeTensorView` values.
3. NumPy `__array_interface__`, dtype objects, contiguity flags, and Python
   object lifetime establish the host pointer contract.
4. Python `DeviceTensor` exposes a native allocation handle and residency flag.
5. `MlirEngine` constructs and lowers MLIR using runtime tensor shapes.
6. `jit_runner.cpp` creates or loads an in-process execution object.
7. `executor.cpp` invokes an MLIR C-interface entrypoint.

This path is valuable compatibility surface but cannot be a dependency of
`mdslc++`, `matcore-extract`, JSON IR verification, generated stubs, or the CPU
reference runtime.

JIT-specific assumptions that must stay out of v0 include:

- `mlir::ExecutionEngine` as the primary artifact producer;
- `_mlir_ciface_<kernel>` as the invocation contract;
- dumping JIT objects and relinking them into cache-local `kernel.so` files;
- discovering the Python extension location with `dladdr`;
- registering allocator and GPU runtime symbols into the JIT;
- runtime output zeroing required by accumulation-form `linalg.matmul`;
- CUDA module loading and graph capture inside a precompiled plan.

## Hardcoded toolchain and runtime paths

The legacy path contains host-local defaults that must not enter installed
MDSLC CMake files or generated sources:

- `/usr/bin/clang` and `/usr/bin/clang++`;
- `/usr/lib/llvm-18/lib/cmake/mlir`;
- `/usr/share/nanobind/cmake`;
- `/usr/lib/llvm-18/lib/libmlir_runner_utils.so`;
- `/usr/lib/llvm-18/lib/libmlir_c_runner_utils.so`;
- `/usr/lib/llvm-18/lib/clang/18`;
- `/usr/local/cuda/include`;
- `/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so`;
- `/lib/x86_64-linux-gnu/libcuda.so`;
- `/lib/x86_64-linux-gnu/libamdhip64.so`;
- CUDA libdevice candidates below `/usr/local/cuda`, `/usr/local/cuda-13.2`,
  and `/usr/lib/cuda`;
- `/usr/bin/clang++` as the final legacy linker fallback.

The standalone preflight must select one coherent Clang/LLVM frontend tuple.
Any later MLIR bridge must independently select a coherent MLIR/LLVM tuple and
must not mix frontend libraries from one version with core/backend libraries
from another.

## ABI-like structures and why they are not the C ABI v0

Existing ABI evidence includes:

- `DeviceBufferHandle { ptr, size_bytes, alloc_id }`;
- `RuntimeTensorView` with pointer, dtype, shape, strides, contiguity,
  residency, and quantization;
- `KernelArgumentDesc` and `LoweredModule` argument metadata;
- executor-local rank-specific MLIR memref descriptors;
- `extern "C"` `_mlir_malloc` and `_mlir_free` symbols;
- generated MLIR `_mlir_ciface_*` wrappers.

These structures do not define the required standalone ABI because they use
C++ containers, variants, templates, exceptions, or MLIR-specific calling
conventions. `runtime_c.h` must instead define fixed-layout versioned
descriptors, explicit enums, policy/target/fallback fields, status codes, and a
no-exception boundary.

## Device ownership, copies, and synchronization

The legacy device-buffer implementation provides strong ownership evidence:

- allocations carry a monotonic ownership token;
- frees validate the token and reject stale/double-free handles;
- upload, download, and zero operations are explicit and synchronous;
- `MatcorePlan` owns its compiled execution bundle and CUDA graph resources;
- graph replay rejects changed tensor addresses.

The legacy GPU lowering also inserts `GpuDataStagingPass` for host memrefs,
which allocates and copies device storage. That behavior is valid only for the
legacy API. It conflicts with the MDSLC rule against hidden host/device copies.
The future bridge must receive already-legal residency decisions and must reject
mixed or unsupported memory spaces before invoking existing lowering.

## Capability and target handling

Useful current behavior:

- x86 features are detected through LLVM host feature APIs;
- NVIDIA probing is lazy and records driver/device/SM information;
- ROCm probing is lazy and distinguishes library presence from device presence;
- NPU runtime detection checks candidate libraries;
- requested targets are canonicalized and converted to execution requirements;
- unsupported targets and insufficient hardware fail explicitly;
- the process cannot silently switch between NVIDIA and AMD runtime ownership.

Missing from the requested long-term model are instruction-family/tile legality,
subgroup constraints, vector width, shared/local memory, registers, async copy,
barriers, address spaces, alignment, launch limits, binary formats, runtime
versions, and library availability/cost. The legacy capability code must
therefore be wrapped as one evidence source, not treated as the complete device
record or planner.

## Verifier and lowering-order evidence

The RegionV1 verifier is the clearest reusable validation pattern. A future
high-level verifier should retain its “validate before emission” ordering and
extend it with source locations, mutability, aliasing, read/write effects,
memory space, policy/fallback, and capability requirements.

Pass ordering is part of the current backend contract:

- CPU: tile/vectorize, canonicalize/CSE, Linalg-to-loops, memref expansion,
  affine lowering, residual vector legalization, Vector-to-SCF, then LLVM
  conversions and requested C wrappers.
- NVIDIA: scheduling and MMA transforms precede data staging; GPU outlining and
  device conversion precede binary serialization; GPU-to-LLVM must occur while
  staging operands still have compatible memref types; host function
  finalization follows.
- AMD: Linalg parallelization, GPU mapping, staging, outlining, target attach,
  AMDGPU/ROCDL conversion, then host lowering and binary packaging.
- Region and Fusion modules are selected through `matcore.kernel_type`; they do
  not pass through the generic matmul transform route.

These sequences are backend implementation details. The Clang frontend must
never know or configure them.

## Cache version and key observations

The legacy disk cache version is:

```text
matcore-phase4-cache-v10-regionv1-block-attn-res-cta-reduce
```

The version participates in the MD5 artifact-directory hash. The structural key
contains detailed GraphV2/RegionV1 nodes and attrs, target, graph mode, and CPU
feature tags. Runtime tensor suffixes are limited to the first three tensors and
record only two shape dimensions. Metadata writes the version, target,
entrypoint, route, tensor count, and output-zeroing flag, but cache load does
not explicitly validate the metadata version and treats metadata parse failures
as non-fatal.

MDSLC JSON IR and generated artifacts require a separate schema version,
deterministic serialization, verifier, and explicit conversion boundary. The
legacy JIT cache must not be reused as that contract.

## Tests and regression implications

The strongest reusable test ideas are:

- explicit rejection before backend dispatch;
- dtype/rank/shape/stride agreement;
- mixed residency rejection;
- Region ID/producer/topological validation;
- unsupported target and capability diagnostics;
- zero-copy pointer and output behavior;
- independent numerical oracles.

Limitations of the current suite:

- it is almost entirely Python/NumPy/Torch based;
- many cases require the native extension or an NVIDIA GPU;
- several scripts hardcode `sm_89`;
- some scripts catch errors and print them without failing;
- some `test_*.py` files execute assertions only from `main`, so pytest does
  not collect meaningful cases from them;
- root CMake does not register a standalone CTest artifact matrix.

Legacy tests remain a regression gate, but MDSLC needs independent CTest tests
for Clang capture, JSON goldens, rewrite safety, C ABI status behavior, normal
objects/linking, installation, and the external consumer.

## Explicit bootstrap-IR-to-backend bridge

The future bridge must be one-way, versioned, and downstream of semantic
analysis:

```text
valid C++ .mdsl
    -> Clang parsing and Sema
    -> canonical matcore::mdsl operation capture
    -> verified Matcore JSON IR v0
    -> capability-aware planner
       |-> CPU v0 C ABI reference/library backend
       `-> future LegacyMlirBridge (explicit conversion)
              -> existing/future Matcore MLIR construction
              -> existing target lowering substrate
              -> AOT host/device artifacts
    -> ordinary host/object/shared-library link
```

Bridge requirements:

1. **The frontend emits only target-independent Matcore IR.** It must not
   include `kernel_ir.h`, MLIR headers, nanobind, runtime probes, or backend
   pass configuration.
2. **JSON IR v0 is verified before conversion.** The verifier owns operation,
   tensor, source-location, dtype, shape, layout/stride, mutability, alias,
   effects, memory-space, target, fallback, and synchronization checks.
3. **Planning precedes backend conversion.** A selected implementation records
   whether execution uses a library call, reference implementation, generic
   structured lowering, or generated accelerator kernel. No target silently
   becomes CPU.
4. **Conversion is explicit and loss-checked.** A future bridge module converts
   exactly one Matcore IR version to a documented legacy `KernelIR` or Matcore
   MLIR dialect version. If legacy structures cannot represent a required
   property, conversion fails rather than dropping it.
5. **Source diagnostics survive the boundary.** Existing emitters use unknown
   MLIR locations. The bridge must attach original `.mdsl` locations or retain
   a stable site-ID diagnostic table through lowering.
6. **Residency is already legal on entry.** The bridge may not invoke hidden
   `GpuDataStagingPass` for an operation whose policy forbids migration.
7. **AOT artifact production replaces JIT orchestration.** Reuse of MLIR passes
   does not imply reuse of `jit_runner.cpp`, disk-cache `.so` loading, Python
   extension discovery, or `_mlir_ciface_*` as the public ABI.
8. **Capability discovery is injected.** The planner consumes a versioned
   capability record. Legacy probes can populate part of it through an adapter,
   but they do not run inside Clang capture.
9. **Backend support is acceptance-test based.** A bridged target is supported
   only after compile, link, execute, correctness, diagnostics, and declared
   artifact tests pass on that target.
10. **The long-term Matcore MLIR dialect is the structured transformation IR.**
    JSON v0 remains the inspectable bootstrap contract; it must not grow into a
    second undocumented optimizer IR alongside legacy `KernelIR` and MLIR.

The first bridge checkpoint should be deliberately narrow: verified rank-2,
row-major, host-resident f32 GEMM with `target=cpu` and `fallback=error`. It
should compare the bridge result against the standalone C ABI reference backend
before any NVIDIA, AMD, fusion, or custom scheduling work begins.

## Files excluded from the v0 standalone dependency graph

The v0 `mdslc++`, `matcore-extract`, JSON verifier, generated stubs, and CPU
runtime must not link or import:

- `src/bindings.cpp` or any Python module;
- nanobind or Python development libraries;
- `src/jit_runner.cpp`, `src/executor.cpp`, or the JIT cache;
- MLIR `ExecutionEngine` or MLIR runner utility libraries;
- GPU runtime symbol registration or CUDA graph code;
- `GpuDataStagingPass`;
- custom NVIDIA MMA/WGMMA lowering;
- AMD/ROCm lowering;
- the root `_matcore_native` target.

This exclusion is additive, not destructive: all listed legacy paths remain in
place for compatibility and future bridge work.
