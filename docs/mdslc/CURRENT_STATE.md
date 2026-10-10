# MDSLC current state

Canonical engineering checkpoint: `c09a9d0e4384441238bf169e7e1122bf13ea5e23`, normal merge of
[PR #91](https://github.com/onlyxItachi/MatcoreDSL/pull/91) from exact premerge
`4fde88b8c3e6b09bc88ce466a0bd28d9dafe6600`.

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema admission, source-visible helpers, frozen source/header/toolchain
  -> immutable values + separate all-MAY-alias host resources
  -> dynamic shapes, per-operation numerics, ordered checks/effect frontiers
  -> exact untransformed Matcore MLIR paired witness
  -> static orchestration from the unchanged sealed Program (no interpreter)
Compiler-private derived plan -> source replay + exact consumption check
Opt-in forwarding retains values; publications/branches invalidate MAY-alias facts.
2–8 original source TUs -> per-file authentication -> checked host program link
Original Clang host -> sealed ABI-checked entry thunk + isolated helper LLVM
Build-issued GEMM -> Linalg/One-Shot -> CPU schedules / GPU outlining
  -> LLVM x64 baseline/AVX/AVX2/AVX512F, ARM64; NVVM sm_89 / ROCDL gfx1150
Private CPU output-tile pattern -> checked M/N parameters, full increasing scalar K
Per-operation permission -> separate AVX2/FMA realization (x64 only)
Opt-in strict lhs pair -> panel4 CPU / qualified NVVM / ROCDL (Linux x64)
GPU combined candidate -> f1 guards; f2 invocation/completion/quarantine
Trusted registry -> guarded candidates, private staging/output, checked completion
  -> ordered publication, owning observation, sticky failure; private DSO ownership
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM and separate Linux/ARM64/Windows/Python lanes remain intact.
```

## Material change

PR91 adds one reusable strict GEMM output-decomposition pattern through upstream
Transform: private M/N tiles in 1..64, checked tails, unchanged full K and f32
multiply/add order. Opt-in build selection connects a requested instance to real
source/installed execution; defaults, runtime/provider policy and GPU paths are unchanged.
Local clean suite: 299/299, zero skips. Native ARM64 tiled source qualification:
126/126, zero skips. Full hosted/review provenance and capability skips are in
the [qualification record](agent-reports/parameterized-pattern-qualification-checkpoint-v1.md).
See [contract and pattern-specific follow-up](PARAMETERIZED_GEMM_PATTERN_V1.md),
[interop limits](COMPILER_INTEROP_BOUNDARIES_V1.md), and the preserved
[PR86 GPU pair checkpoint](agent-reports/gpu-pair-source-qualification-checkpoint-v1.md).

## Unsupported or unproven

Experimental API/ABI and coherent 21.1.8 remain. This pattern is not source-time
specialization, automatic tuning, a cost model, GPU tile mapping or a performance win.
No parity, general fusion/rank-N/views, automatic reuse, residency/asynchrony,
zero-copy, Metal/NPU, generated-region Windows or universal compiler interchange.
GPU support retains its [exact target/work limits and failure law](STAGED_GPU_CANDIDATES_V1.md);
real CUDA pair host-ASan failed prelaunch at cuInit2; HIP/global host-ASan remains
unqualified/configure-refused; AMD has no device-sanitizer claim. Nothing here repairs these.
Opaque imports/cross-region optimization remain unsupported; other ISA/provider/toolchain
paths require their own qualification. Caller lifetimes/race freedom, conforming runtime/
allocation, trusted loading and provider/manual-link limits remain, not sandbox/crash atomicity.
[#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15) stays partial/open,
[#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) design-only/open;
[#77](https://github.com/onlyxItachi/MatcoreDSL/issues/77) has no performance qualification.

## Exactly one next boundary

**[Issue #89](https://github.com/onlyxItachi/MatcoreDSL/issues/89): qualify a closed parameterized bounded static-shape SPIR-V strict GEMM issuer.**
This remains the independent engineering frontier, not a dependency or consequence
of CPU tiling. Falsify the reusable positive rank-2 family, complete graph/ABI/strict
controls and independent target evidence; fixed 2x3x4 is only a control.
No source/runtime authority. [Research PR #84](https://github.com/onlyxItachi/MatcoreDSL/pull/84)
remains draft; its [source-frontier audit](https://github.com/onlyxItachi/MatcoreDSL/blob/d214b017cad93ede7445148988f7aad36bd4573f/docs/mdslc/agent-reports/spirv-source-frontier-audit-v1.md)
separates issuer proof from Vulkan failure/identity/artifact obligations.
