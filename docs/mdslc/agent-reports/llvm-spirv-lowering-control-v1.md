# LLVM 21.1.8 SPIR-V lowering-contract control

Report-only research from clean main `13283c440fc4fdad42606bb723dfcedde4f07644`.
No MDSLC production, CMake, target integration or public contract changed. Raw
inputs, binaries and disassembly remain external at
`/var/tmp/mdslc-llvm-spirv-control.HLFv0ozl`.

## Bounded result

Five serial, tiny `llc-21` invocations, their five `spirv-val` checks and their
five disassemblies returned 0 in author execution. No compiler abort/refusal
occurred. All emitted modules are SPIR-V 1.4 **Kernel / Physical64 / OpenCL**,
not the MLIR research route's Logical / GLSL450 Vulkan compute shaders.

| Control | Input / extension flag | ContractionOff | f32 global float modes | Per-op RTE |
| --- | --- | --- | --- | --- |
| plain | `plain.ll`, none | yes | none | no |
| plain-ext | `plain.ll`, KHR enabled | yes | none | no |
| modes | `modes.ll`, KHR enabled | **no** | all three | no |
| modes-full | `modes-full.ll`, KHR enabled | yes | all three | no |
| constrained | `constrained.ll`, KHR enabled | yes | none | both arithmetic results |

Every module contains one separate `OpFMul` followed by one `OpFAdd`; none has
an operation-level `NoContraction` decoration. The plain variants are byte
identical: enabling `SPV_KHR_float_controls` alone does not request its modes.
The three explicit modes are `DenormPreserve 32`,
`SignedZeroInfNanPreserve 32`, and `RoundingModeRTE 32`; their matching
capabilities and KHR extension appear in both metadata variants.

Pinned [SPIRVAsmPrinter.cpp:460–523](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Target/SPIRV/SPIRVAsmPrinter.cpp#L460-L523)
explains the observed edge: automatic Kernel `ContractionOff` is emitted only
when neither `spirv.ExecutionMode` nor `opencl.enable.FP_CONTRACT` named
metadata exists. The explicit float-mode list therefore must also carry mode
31 to retain that declaration. This is a future issuer negative-guard
obligation, not a universal upstream bug claim. Absence of `NoContraction`
alone is not evidence that this Kernel route lacks an anti-contraction mode.

Constrained RTE/ignore-exception intrinsics compile to separate arithmetic
with `FPRoundingMode RTE` on each result, but do not imply the three global
preservation modes. No ablation isolates the contribution of each ordinary
function attribute. Only these inputs at `-O0` were inspected; latest LLVM,
other optimization levels, environments and hardware are not checked.

## Exact inputs

`plain.ll` (UTF-8, LF, final newline):

```llvm
; Throwaway lowering-contract control only. No MDSLC or device authority.
target triple = "spirv64-unknown-unknown"

define spir_kernel void @fp_control(float %a, float %b, float %c,
                                    ptr addrspace(1) %out) #0 {
entry:
  %product = fmul float %a, %b
  %sum = fadd float %product, %c
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { strictfp "denormal-fp-math"="ieee" "denormal-fp-math-f32"="ieee" "unsafe-fp-math"="false" }
```

`modes.ll` is otherwise byte-identical to `plain.ll`, replacing the first line
with `; Same throwaway arithmetic as plain.ll; explicit LLVM SPIR-V execution modes.`
and appending these lines immediately after the attributes line:

```llvm
!spirv.ExecutionMode = !{!0, !1, !2}
!0 = !{ptr @fp_control, i32 4459, i32 32}
!1 = !{ptr @fp_control, i32 4461, i32 32}
!2 = !{ptr @fp_control, i32 4462, i32 32}
```

`modes-full.ll` instead uses first line
`; Explicit mode list includes the Kernel-specific noncontraction requirement.`,
adds `!3` to that named list and appends `!3 = !{ptr @fp_control, i32 31}`.
The mode IDs are pinned in
[SPIRVSymbolicOperands.td:665–677](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Target/SPIRV/SPIRVSymbolicOperands.td#L665-L677).

`constrained.ll`:

```llvm
; Explicit RTE/ignore-exceptions lowering control, not a runtime strict profile.
target triple = "spirv64-unknown-unknown"

define spir_kernel void @fp_control(float %a, float %b, float %c,
                                    ptr addrspace(1) %out) #0 {
entry:
  %product = call float @llvm.experimental.constrained.fmul.f32(
      float %a, float %b, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  %sum = call float @llvm.experimental.constrained.fadd.f32(
      float %product, float %c, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
attributes #0 = { strictfp "denormal-fp-math"="ieee" "denormal-fp-math-f32"="ieee" "unsafe-fp-math"="false" }
```

## Commands and provenance

The installed `/usr/bin/llc-21` reports Ubuntu LLVM 21.1.8, including registered
`spirv`, `spirv32` and `spirv64` targets. Its route is documented in pinned
[SPIRVUsage.rst](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/docs/SPIRVUsage.rst).
The already-extracted SPIRV-Tools binaries report v2026.1, unknown source hash,
2026-02-06T09:16:33+00:00. No installation or LLVM build occurred.

Replay commands preserve the exact compiler flags and input contents, using
the author artifact directory as working directory. The constrained control
retains the child-only core-disable/30-second-timeout safety envelope:

```sh
/usr/bin/llc-21 -O0 -mtriple=spirv64-unknown-unknown -filetype=obj -fp-contract=off plain.ll -o plain.spv
/usr/bin/llc-21 -O0 -mtriple=spirv64-unknown-unknown -filetype=obj -fp-contract=off -spirv-ext=+SPV_KHR_float_controls plain.ll -o plain-ext.spv
/usr/bin/llc-21 -O0 -mtriple=spirv64-unknown-unknown -filetype=obj -fp-contract=off -spirv-ext=+SPV_KHR_float_controls modes.ll -o modes.spv
/usr/bin/llc-21 -O0 -mtriple=spirv64-unknown-unknown -filetype=obj -fp-contract=off -spirv-ext=+SPV_KHR_float_controls modes-full.ll -o modes-full.spv
bash -c 'ulimit -c 0; exec timeout 30 /usr/bin/llc-21 -O0 -mtriple=spirv64-unknown-unknown -filetype=obj -fp-contract=off -spirv-ext=+SPV_KHR_float_controls constrained.ll -o constrained.spv'
```

For each `<name>` above, validation and disassembly were:

```sh
/home/hamza-usta/mdslc-work/multitarget-v1/builds/metal/tool-root/usr/bin/spirv-val <name>.spv
/home/hamza-usta/mdslc-work/multitarget-v1/builds/metal/tool-root/usr/bin/spirv-dis --raw-id <name>.spv -o <name>.spvasm
```

Validation used the default universal SPIR-V environment, **not Vulkan
validation or physical-device execution**. Exit statuses are author tool-call
evidence, not an independent review/rerun or a committed automated test.

SHA-256:

```text
935bdae46915daba23b279efebc04400c085c56a8536dea633f1f5a96c9fd382  /usr/bin/llc-21
85367fefdb7e93ae45654255ac2b7f8dc7056b6df78a6fdeb03ce395c7477239  spirv-val
e4d0ff4856346a3190f6c289b108cf00be1ee768041f49dec5c61555b353f69d  spirv-dis
a6ce7887b6dba924b6fb09bc9b7c9fce5f853fef6ea18e14c4ee475f3f8ecc27  plain.ll
488e46eaa8ce524c256bc43061e56faf780c4814ed4f6c56f1521418ba0aa4f3  modes.ll
a5277742a95eb6dacb8ac2e6d59e443195b7947293045e1da791dfce6ee37bb4  modes-full.ll
c42ec6ad009cd19f73eabaa1887d06d4f76cb488da0141f28c940a7707241791  constrained.ll
69f82d9ddfb7fa50b589877066a350f32b88a9c34c86115cd5b53c4f5b3e3210  plain.spv / plain-ext.spv
a1873e9e82e0fcbf27be4aa1e0c292dfc560cbdae0eb2df7bc81eb14e9eef73b  modes.spv
4b7d4f9fdf9df4c4b4b05ae2aec338197f7ccb540ae9440251c30b1aa224a243  modes-full.spv
646b8e9610bf8ea937aa7934feda89d4781717822288c7db76ca3676ba4045b1  constrained.spv
```

No mathematical oracle, kernel execution, runtime capability discovery,
Vulkan adapter, MDSLC source authority, package qualification or performance
result follows. No LLVM issue was created. This control separates LLVM-SPIR-V
Kernel lowering from the independently bounded MLIR-SPIR-V shader research;
future authority must verify the complete required mode set and the actual
execution environment, not infer it from a compiler flag or one decoration.
