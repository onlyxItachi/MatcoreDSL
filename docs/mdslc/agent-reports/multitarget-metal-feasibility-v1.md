# Optional Metal feasibility: research, not a candidate

Starting canonical main: `9c2149a25e12f5192c168f3e047097505263fc1a`.
Research branch: `research/mdslc-metal-feasibility-v1`.
Worktree: `/home/hamza-usta/mdslc-work/multitarget-v1/metal`.
No production code, runtime registry, language semantics or target authority changes.

## Decision

**GO for a bounded upstream/physical research probe. NO-GO for advertising the
current strict-f32 Matcore candidate on Metal.** Ordinary compilation success
cannot establish the missing numerical guarantee. Do not silently broaden the
existing `reassociate_f32` permission to permit denormal flushing either.

The useful route is Linalg → GPU → SPIR-V → SPIRV-Cross → MSL → Apple Metal
compiler/runtime. It reuses existing structured lowering; it is not a direct
LLVM target backend, and it is not evidence that MLIR owns a complete Metal
runtime. Nothing here replaces the CPU/native/provider path or the separately
qualified NVIDIA/AMD work.

## Mechanically observed locally

The small [reproducer](../../../compiler/experiments/multitarget_metal_v1/reproduce.py)
and [static specimen](../../../compiler/experiments/multitarget_metal_v1/static_specimen.mlir)
instantiate the established bufferized `linalg.fill(+0)` + `linalg.matmul` shape
at M=2, N=4, K=3. This is **a research instantiation**, not a new authenticated
Matcore issuer or an admitted source program. The arithmetic is not handwritten
Metal. All three [MSL goldens](../../../compiler/experiments/multitarget_metal_v1/golden)
were read completely and compared byte-for-byte with reproduced output after
the documented EOF-newline normalization.

1. MLIR 21.1.8 Linalg parallel-loop conversion, GPU mapping/launch/outlining,
   SPIR-V conversion and ABI lowering produced two modules: positive-zero fill
   and increasing-K GEMM. Each output has one workgroup with one thread.
2. Both serialized modules pass `spirv-val --target-env vulkan1.1` and translate
   through SPIRV-Cross to MSL 2.1 source. Validation is of the binary structure,
   not a statement that Vulkan or Metal devices satisfy every numerical mode.
3. Default `arith.mulf/addf` conversion produced SPIR-V `OpFMul/OpFAdd` but **no
   NoContraction, DenormPreserve or RoundingModeRTE**. The resulting MSL collapses
   the source expression to `C += A * B`, which requires explicit compilation
   controls before claiming separate rounding.
4. The controlled counterexample adds valid NoContraction decorations and the
   DenormPreserve, RoundingModeRTE and SignedZeroInfNanPreserve execution modes
   to that same binary. It still validates. SPIRV-Cross accepts it, but its MSL
   is **byte-identical to NoContraction-only translation**. Merely adding the
   missing modes does not make this bridge preserve their meaning.
5. NoContraction translates to SPIRV-Cross's `optnone` scalar helpers implemented
   using explicit `fma` calls. This is upstream's implementation, not an MDSLC
   workaround. It prevents one source contraction form; its signed-zero and
   denormal behavior must still be checked. Unused generated matrix helper
   templates are present in the golden but never instantiated by this GEMM.
   In particular, `fma(x,y,+0)` is not a universal replacement for multiply:
   it can change a negative-zero product to positive zero. This GEMM's initial
   positive-zero accumulation may hide that difference, so this specimen must
   not be generalized into a scalar arithmetic proof.
6. The **actual dynamic outlined NVIDIA research specimen** was also tested
   with the SPIR-V Shader/storage-buffer target environment and entry ABI.
   It fails at `memref.store` for `memref<?x?xf32>`. Upstream
   `getVulkanElementPtr` requires static strides/offsets; this is not evidence
   that all dynamic SPIR-V buffers are impossible. A dynamic rank-1 buffer ABI
   with explicit dimensions/index arithmetic, or bounded static specialization,
   would be separate derivations, not permission to erase descriptors.

20 tool invocations were recorded: **19 returned success, one returned the
expected dynamic-stride rejection**. Four SPIR-V binary variants validated;
three source goldens and the MLIR fixture match their recorded hashes. The
host oracle's nine discriminating controls passed a separate small Linux
Clang 21.1.8 compile/run. Objective-C++/Metal compilation and physical execution
are **not locally validated** on the Linux host. Hosted results, when available,
must be appended with exact run/head/device details rather than inferred.

## Exact upstream evidence

Apple's [MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
4.1, dated 2026-06-04, §§1.6.3, 8.1–8.5: safe/precise math does not by itself
disable contraction; subnormal operands/results may flush; either nearest-even
or toward-zero arithmetic rounding is permitted; flushed-zero sign is not
guaranteed. `-ffp-contract=off` or the Metal contraction pragma addresses
contraction, **not all the other differences**. The relevant rendered PDF pages
368 and 376 were visually checked against the extracted text. A passing finite
normal-number sample cannot replace these missing guarantees.

- [MLIR 21.1.8 GPU conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/GPUToSPIRV/GPUToSPIRVPass.cpp)
  owns progressive operation conversion and clones the original GPU module.
- [SPIR-V pointer/index conversion](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/SPIRV/Transforms/SPIRVConversion.cpp#L1219)
  explains the dynamic-stride rejection. The same restriction was inspected
  at LLVM main `be60b244d1f9fa798c8922e313679035ad9c2ee5`, not executed as a new
  toolchain. Current-main inspection does not imply 21/22/main compatibility.
- [ArithToSPIRV](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Conversion/ArithToSPIRV/ArithToSPIRV.cpp)
  uses ordinary elementwise FAdd/FMul conversion. The same pertinent patterns
  were inspected in that frozen current-main source.
- [Khronos float controls](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_float_controls.html)
  define explicit denormal, rounding and signed-zero requirements; these are
  target capabilities, not promises conferred by successful serialization.
  [Float controls2](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_float_controls2.html)
  adds finer permissions but is not an automatic solution to Metal's limits.
- [SPIRV-Cross MSL at the SDK source tag](https://github.com/KhronosGroup/SPIRV-Cross/blob/fb0c1a307cca4b4a9d891837bf4c44d17fe2d324/spirv_msl.cpp)
  documents the helper implementation used by the small packaged translator.
  Current upstream main `be71ee8c12cd7dc5ca8fa9581f708c2e8561fe2a` was independently
  inspected: `spirv_parser.cpp`, `spirv_glsl.cpp`, `spirv_msl.cpp`; no MSL
  denormal/rounding realization was found in those implementations. The
  byte-identical-output claim above is limited to the actual tested package.
- [MoltenVK's capability implementation](https://github.com/KhronosGroup/MoltenVK/blob/4aaf714aa1b3e78e26ecfcefa9c75e9a576c500b/MoltenVK/MoltenVK/GPUObjects/MVKDevice.mm#L858)
  explicitly reports `shaderDenormPreserveFloat32=false` and
  `shaderRoundingModeRTEFloat32=true`. That exact current upstream source was
  read, not built. MoltenVK is a relevant alternate host/runtime route for the
  same SPIR-V, but does not supply the missing strict denormal guarantee. A
  future Vulkan candidate must consume actual device float-control properties;
  successful SPIR-V validation cannot substitute for that capability check.

## Runner and runtime experiment

It would be false to say GitHub universally has no Metal-capable machines.
[Standard runner documentation](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
lists Apple Silicon macOS runners without a blanket Metal guarantee.
[Larger runner documentation](https://docs.github.com/en/actions/reference/runners/larger-runners)
explicitly advertises M2 XLarge GPU acceleration, but requires separate paid
provisioning. This experiment enables **no paid runner or account change**.

The branch-only [workflow](../../../.github/workflows/mdslc-metal-research.yml)
uses standard `macos-15`. It records the exact host/compiler/display inventory,
checks golden identity, optionally compiles MSL offline when Apple's compiler
component exists, and builds the small [research host](../../../compiler/experiments/multitarget_metal_v1/probe.mm).
No compiler components are installed or downloaded by that workflow.

The host calls both device enumeration and
[MTLCreateSystemDefaultDevice](https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice%28%29).
Apple explicitly requires CoreGraphics linkage for such command-line device
discovery, so this probe links it. An actual nil device gives
`SKIP_NO_METAL_DEVICE`, not a fabricated pass. Non-unified storage is explicitly
unimplemented in this tiny Apple-Silicon probe, not declared unsupported by Metal.

The emitted fill and GEMM run with explicit shared input/private-result staging,
separate ordered encoders and checked
[completion](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted%28%29).
The 10 cases cover rectangular indexing, input/output subnormals, negative
subnormals, signed zero, a real FMA discriminator, K-order, RTNE-versus-RTZ,
infinity and quiet NaN (NaN class, not payload identity). Each has eight output
comparisons. Host input immutability and destination guards are checked.
The host uses fast math disabled and an explicit no-contraction pragma for both
plain and NoContraction-generated shaders. Both offline and runtime compilation
explicitly select MSL 2.1. Artifact identity or compilation failures prevent
the execution step; only evidence upload uses unconditional workflow execution.

Numerical mismatches are **successful falsification observations**, retained as
`EXECUTED_STRICT_COUNTEREXAMPLE` with exact expected/actual bits; they do not
masquerade as target support. Even a clean sample says
`EXECUTED_BOUNDED_SAMPLE_MATCH` and `strict_contract_qualified: false`.
API/compile/guard/oracle errors fail the research job. Missing devices can skip
only after genuine discovery. No performance measurements are made.

## Ownership and next boundary

Matcore retains mathematical/numerical authority, shape/range legality, source
identity, candidate qualification and ordered publication/failure semantics.
MLIR owns Linalg/GPU/SPIR-V transformations and interface lowering; SPIRV-Cross
owns its MSL translation; Apple's toolchain owns machine code. A future Matcore
Metal adapter would still need exact artifacts, resource lifetimes, transfer
visibility, synchronization/failure containment and publication only after
checked completion. Shared storage in this diagnostic is neither zero-copy
MDSLC support nor persistent device-residency semantics.

**Next justified boundary: run and preserve this exact standard-macOS diagnostic.**
If arithmetic fails, retain the counterexample and stop strict-Metal admission.
If it matches, investigate a precise hardware/toolchain contract before product
integration. Neither result licenses silently weakened numerics. Dynamic ABI
flattening and a production adapter are deferred independent engineering work.
Do not handwrite another GEMM shader to circumvent the observed seam.

## Reproduction identity

Local artifacts: `/home/hamza-usta/mdslc-work/multitarget-v1/builds/metal/reproduce`.
`manifest.json` records every command/result, generated SHA-256, source identity
and exact executable hashes; `commands.json` survives a failed rerun.
Task TMPDIR: `/home/hamza-usta/mdslc-work/multitarget-v1/tmp/metal`.

```sh
TMPDIR=/home/hamza-usta/mdslc-work/multitarget-v1/tmp/metal \
python3 compiler/experiments/multitarget_metal_v1/reproduce.py \
  --mlir-bin /home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/bin \
  --spirv-bin /home/hamza-usta/mdslc-work/multitarget-v1/builds/metal/tool-root/usr/bin \
  --output /home/hamza-usta/mdslc-work/multitarget-v1/builds/metal/reproduce \
  --dynamic-outlined /home/hamza-usta/mdslc-work/multitarget-v1/builds/nvidia/final-reproduce/outlined.mlir \
  --check-golden
```

The two Ubuntu packages were downloaded (3,860,766 bytes combined), checked and
extracted only into `builds/metal/tool-root`, **not installed**:

| Evidence | SHA-256 |
| --- | --- |
| `spirv-cross_2021.01.15+1.4.335.0-1_amd64.deb` | `50d11b7efc263240d04b015fecfd419377a4a3e2a9cb59387f2229aae03e4e7f` |
| `spirv-tools_2026.1-1_amd64.deb` | `24e972ed4f2e92ada6f64b32ff40550fda02038385656736af871ca3dcb2b867` |
| Apple MSL 4.1 PDF | `41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5` |

No heavy source builds, production changes, system toolchain mutation, history
rewrite, performance claim or MDSLC Metal target claim occurred in this lane.

## Independent pre-push review

Root reviewed the complete host, all three goldens, workflow, reproducer and
record. It required fail-closed workflow ordering after artifact identity,
explicit MSL-version pinning, exact negative return codes (not any crash), and
60-second child-process timeouts. All were incorporated before hosted execution.
A separate CPU-lane reviewer found the first K-order witness was too weak:
`[2^24,-2^24,1]` does not distinguish both parenthesizations. The replacement
`[2^24,1,-2^24]` produces strict zero but cancellation-first one, and the host
now explicitly proves its wrong-order oracle differs before launching anything.
This is ordinary engineering review, not native Apple validation or a security
audit. Rejected weak fixtures remain visible in normal commit history.
