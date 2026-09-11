# CPU ISA and ARM correctness campaign v1

Base: clean `9c2149a25e12f5192c168f3e047097505263fc1a`, branch
`mdslc/cpu-isa-arm-correctness-v1`. This is implementation-owner evidence, not
independent approval, product support or performance evidence.

## Initial read-only audit

Read `AGENTS.md`, generated strict/reassociate contracts, current region guide,
closed-source contract and candidate-coexistence runtime contract before edits.

- `compiler/CMakeLists.txt` and `compiler/lib/regions/CMakeLists.txt` explicitly
  gate experimental executable regions/candidate construction to Linux x86-64
  and coherent LLVM/Clang/MLIR 21.1.8.
- `MatcoreCpuGemmCandidate.h` fixes the strict target to
  `x86_64-pc-linux-gnu`; the issuer stamps it before MLIR-to-LLVM conversion.
  Target generalization belongs in a closed compiler target selection, not in
  semantic IR permissions or arbitrary imported LLVM.
- `closed_host_v1.cpp` only admits Linux x86-64, and its complete FP scope saves
  MXCSR/x87 control/status, normalizes privately, checks controls and restores
  full state. ARM requires independently checked FPCR/FPSR handling. The legacy
  `fp_environment_v1` control-only helper is not a substitute.
- `ArtifactSymbolOwnership.cpp` requires Linux x86-64 host LLVM and ELF
  `EM_X86_64` DSOs. ARM needs the same ELF64 little-endian symbol-ownership proof
  matched to `EM_AARCH64`, not removal of the target check. Native ABI checks
  must retain the exact 56-byte, 8-aligned LP64/i64 memref descriptor layout.
- Generated reassociate is a separate `+avx2,+fma` candidate with explicit
  source permission and direct hardware/OS XMM/YMM guards, not x86-64-v3.
  Strict permits no FMA on any ISA. Capability-v2 has no separate AVX wire bit;
  do not silently add one to that versioned record or reuse AVX2 as AVX evidence.
- Object fixtures assume ELF x86-64/SSE opcodes. Existing hosted setup includes
  `g++-multilib`, which must not be copied indiscriminately to Linux ARM.

Exact source integration seam: sealed source/paired semantic graph enters
`emitClosedHostV1`, generated orchestration calls `Session::gemm`, per-operation
`candidateLegality` precedes private allocation, and `closed_host_v1.cpp` invokes
the compiler-issued `_mlir_ciface___matcore_strict_gemm_f32_v1` after FP setup.
`regions/CMakeLists.txt` owns issuer -> `.ll` -> isolated object -> private DSO;
the driver retains frozen host, ABI-thunk and artifact-symbol ownership checks.
Leaf portability alone does not discharge those source/runtime obligations.

## Dependency evidence

Live local Clang, LLVM, development libraries and compiler-rt report 21.1.8,
Ubuntu package `1:21.1.8-6ubuntu1`. MLIR tools report 21.1.8 at
`/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`.
No system packages or LLVM sources were built/changed by this lane.

GitHub lists native `ubuntu-24.04-arm` and `ubuntu-22.04-arm` standard runners:
[official specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
[apt.llvm.org](https://apt.llvm.org/) documents ARM64 LLVM/Clang/MLIR packages.
The live [Noble LLVM21 ARM64 index](https://apt.llvm.org/noble/dists/llvm-toolchain-noble-21/main/binary-arm64/Packages.gz)
contains `clang-21`, `libclang-21-dev`, `libclang-cpp21-dev`, `llvm-21-dev`,
`libclang-rt-21-dev`, `libmlir-21-dev` and `mlir-21-tools` at one version:
`1:21.1.8~++20251221032922+2078da43e25a-1~exp1~20251221153059.70`.
This is repository-index availability, not an installed/native-tested ARM tuple.
Future CI must authenticate packages and verify all executable/CMake versions.

## Isolated generated-leaf prototype

Only `compiler/experiments/multitarget_cpu_v1` and this report changed in this
first commit. It reuses the real closed MLIR issuer and unchanged permanent
174-check strict fixture, plus an independent exact double oracle for 15,444
output elements across 198 cases. All generated artifacts stay on the designated
SSD, task TMPDIR is explicit and compilation is strictly sequential.

Executed command:

```sh
python3 -B compiler/experiments/multitarget_cpu_v1/run.py \
  --issuer /home/hamza-usta/mdslc-work/multitarget-v1/builds/base/bin/matcore-cpu-gemm-candidate \
  --output /home/hamza-usta/mdslc-work/multitarget-v1/builds/cpu-arm/leaf-matrix-01 \
  --tmp-dir /home/hamza-usta/mdslc-work/multitarget-v1/tmp/cpu-arm
```

Fresh root-built issuer SHA256:
`46137779068edfbf944975adcf47e23f718eca788086b11989400b1ca10c0fba`.
Issued row-contiguous LLVM SHA256:
`9b42c60b2a4f5b2e87dc1c768c108a798baf4d988a76d126df8792660a683dd8`.

| Isolated target request | Object check | Physical execution | Observed widths |
| --- | --- | --- | --- |
| baseline x86-64 | PASS; separate arithmetic/no FMA | 15,618 checks PASS | XMM, no YMM/ZMM |
| AVX | PASS; separate arithmetic/no FMA | 15,618 checks PASS | YMM, no ZMM |
| AVX2 | PASS; separate arithmetic/no FMA | 15,618 checks PASS | YMM, no ZMM |
| AVX2 + FMA permitted | PASS; separate arithmetic/no FMA | 15,618 checks PASS | YMM, no ZMM |
| AVX512F permitted | PASS; separate arithmetic/no FMA | 15,618 checks PASS | YMM, no ZMM |
| AArch64 armv8-a override | PASS; ELF64/fmul/fadd/no FMA | NOT RUN: cross-compile only | ARM object only |

Total x86 numerical checks: **78,090**, no failures/skips. Import checks admit
only `memset`; both issued strong function identities are present. Eight
synthetic classifier tests pass, including fused arithmetic, foreign object,
extra function and provider-import rejection. `git diff --check` passes.

Important negative finding: AVX2, AVX2+FMA and AVX512F requests produced the
**identical** object SHA256
`a2cd868b5dacb02f960b1306f1e3765dd0f4fc58597153bb1c3ed745803636f2`.
Therefore this does **not** demonstrate AVX512 or FMA instruction execution.
No vector-width preference, tuning or timing was introduced.

AArch64 object SHA256:
`fa4ed944828076b0eb35d790bc391ddee0e3b5fad9c8f5142f914286ae015c8f`.
Its explicit Clang target override is experimental and does not change the
production issuer's target authority or establish native ARM support. Complete
commands/diagnostics, assembly and JSON results remain in `leaf-matrix-01`.

Next smallest production boundary: fixed native Linux AArch64 strict issuer
target plus full FPCR/FPSR scope and matched ELF ownership, then real native ARM
source execution/negative tests in CI. Existing numerical, provider, automatic
selection, publication and source authority contracts must remain unchanged.
