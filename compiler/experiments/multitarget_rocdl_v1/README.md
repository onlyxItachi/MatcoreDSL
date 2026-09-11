# ROCDL / gfx1150 correctness experiment

This opt-in research executable proves a real **compiler-issued strict tensor
GEMM -> MLIR -> AMDGPU code object -> physical gfx1150** route. It does not add a
registered MDSLC candidate, source admission, public storage mutation, residency
semantics or performance policy. No HIP kernel source is handwritten.

## Reproduce

Use an already-built canonical `matcore-cpu-gemm-candidate`, coherent MLIR/LLVM
21.1.8 development tools, and a working gfx1150 HIP installation. The script
installs nothing and builds only two small host utilities, sequentially. Use
SSD-backed build and temporary directories, not `/tmp` or `/dev/shm`.

```sh
bash compiler/experiments/multitarget_rocdl_v1/run.sh \
  /home/hamza-usta/mdslc-work/multitarget-v1/builds/amd/repro \
  /home/hamza-usta/mdslc-work/multitarget-v1/builds/base/bin/matcore-cpu-gemm-candidate \
  /home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21 \
  /opt/rocm
```

This is a physical test: missing hardware or an execution failure returns
nonzero, never a pass through CPU fallback. `HSA_OVERRIDE_GFX_VERSION` is rejected.
The script does not benchmark or claim performance. Do not automatically enable
it on hardware-free hosted runners.

## Actual pipeline

1. Existing private issuer emits semantic witness and dynamic tensor
   `linalg.fill` + `linalg.matmul`; no copied semantic mirror is maintained here.
2. One-Shot Bufferize converts function boundaries with identity-layout memrefs.
3. Upstream Linalg-to-parallel-loops leaves K serial. GPU mapping, parallel-loop
   conversion, index sinking and outlining create a fill kernel and GEMM kernel.
4. Affine/SCF and GPU-to-ROCDL conversion preserve separate f32 multiply/add.
5. The small C++ utility clones parsed device LLVM operations without changing
   their bodies, preserves target data layout, sets AMDGPU target triple, and
   verifies both modules. This is extraction, not source authentication.
6. MLIR LLVM translation, LLVM `llc` targeting gfx1150 and `ld.lld -shared`
   produce actual AMD HSA code objects. FMA is disabled and f32 denormals use IEEE
   preservation. The disassembly and target metadata are checked.
7. The host loads both modules through HIP, allocates guarded private device
   buffers, explicitly uploads immutable inputs, launches fill then GEMM,
   synchronizes, copies output back and checks all results/guards/inputs.

The bounded schedule uses one block per output element, one workitem per block;
it is deliberately not an HPC schedule. M/N/K are at most 65535 in this fixture.
Descriptors are the internal upstream expanded rank-2 ABI (7 fields each), not
a new Matcore ABI. No physical alias disjointness is inferred between inputs.

## Evidence

The physical runner includes seeded rectangular mathematics, empty M/N/K,
actual device input/input alias, NaN classification, infinity and invalid
products, signed zero, FMA and increasing-K discriminators, and subnormal
inputs/results. It compares strict results bitwise, except NaN payload bits.
The FMA-enabled mutation and flushed-subnormal mutation must both fail their
specific numerical discriminator after successful GPU execution.

Artifacts, source/issuer hashes, output, ISA and numerical negative-control
logs remain in the build directory. See the [detailed report](../../../docs/mdslc/agent-reports/multitarget-rocdl-v1.md).

This does not yet prove the source-to-executable driver, candidate-library
identity, ordered region failures, GPU resource cleanup under driver failure,
sanitizer coverage or hardware other than the inspected gfx1150 system.
