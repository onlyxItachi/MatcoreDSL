# MDSLC current state

Canonical engineering checkpoint: `523f8637bb135b7b5e6b4f08f563a2c93a2ebfee`, the normal merge of
[PR #86](https://github.com/onlyxItachi/MatcoreDSL/pull/86) from exact premerge head `fb70f7d`.

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema admission, source-visible helpers, frozen source/header/toolchain
  -> immutable values + separate all-MAY-alias host resources
  -> dynamic shapes, per-operation numerics, ordered checks/effect frontiers
  -> exact untransformed Matcore MLIR paired witness
  -> static orchestration from the unchanged sealed Program (no interpreter)
Compiler-private immutable derived plan -> source replay + exact consumption check
Opt-in forwarding retains values; publications/branches invalidate MAY-alias facts.
2–8 original source TUs -> per-file authentication -> checked host program link
Original Clang host -> sealed ABI-checked entry thunk + isolated helper LLVM
Build-issued GEMM -> Linalg/One-Shot -> CPU Transform or GPU outlining
  -> LLVM x64 baseline/AVX/AVX2/AVX512F, ARM64; NVVM sm_89 / ROCDL gfx1150
Per-operation permission -> separate AVX2/FMA realization (x64 only)
Linux x64 opt-in strict lhs pair -> row-panel4 CPU / NVVM sm_89 / ROCDL gfx1150
GPU combined candidate -> f1 guards only; f2 invocation/completion/quarantine
Trusted registry -> guarded candidates, private staging/output, checked completion
  -> ordered host publication, owning observation, sticky failure prefix
Private DSO owns implementation; Runtime owns provider policy; driver pins artifacts.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM and separate Linux/ARM64/Windows/Python lanes remain intact.
```

## Material change

PR86 connects `--optimization strict-fused-pair` with explicit
`generated-nvvm`/`generated-rocdl` to one closed strict two-GEMM kernel on Linux x64
sm_89/gfx1150. It retains the immutable Program/witness, both original guard bundles,
full logical C extent, private panel/output and prior effects. Shared poison/quarantine
prevents output or sibling reuse after uncertain completion; K=0 still evaluates `+0*D`.
This is deliberately serial correctness, not performance or general GPU fusion.
See the [qualified source/runtime checkpoint](agent-reports/gpu-pair-source-qualification-checkpoint-v1.md)
and [combined realization contract](GPU_STRICT_FUSED_PAIR_V1.md).
Local qualification belongs to compiler/AGENTS-identical clean `9f7f9af`; final `fb70f7d`
hosted acceptance includes the [Release budget-only correction](agent-reports/gpu-pair-release-ci-budget-v1.md), not removed commands or scope.
Earlier [CPU source](STRICT_FUSED_PAIR_SOURCE_V1.md) and [issuer/image qualification](agent-reports/gpu-pair-issuer-merged-checkpoint-v1.md) retain their independent evidence.
Default/native/provider routes and independent [forwarding](PUBLICATION_READ_FORWARDING_V1.md) are unchanged.

## Unsupported or unproven

Experimental API/ABI and coherent 21.1.8 remain; GPU recipes obey [exact target/work limits](STAGED_GPU_CANDIDATES_V1.md), not arbitrary hardware.
Real CUDA pair host-ASan failed prelaunch at cuInit2; real HIP/global host-ASan remains unqualified/configure-refused.
API mocks, CUDA device Memcheck and normal physical execution are separate scopes; AMD has no device-sanitizer claim.
No performance/parity, broad fusion, automatic reuse, residency/asynchrony, zero-copy, general rank-N/views, Metal/NPU or generated-region Windows claim.
Other ISA/provider/toolchain paths require their own qualification; opaque imports and cross-region optimization remain unsupported.
Caller lifetimes/race freedom, conforming runtimes/allocation and trusted loading remain prerequisites, not sandbox/crash-atomic guarantees.
Provider/manual-link boundaries remain; [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15) is partial/open, [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) design-only/open and [#77](https://github.com/onlyxItachi/MatcoreDSL/issues/77) has no performance qualification.

## Exactly one next boundary

**[Issue #89](https://github.com/onlyxItachi/MatcoreDSL/issues/89): qualify a closed parameterized bounded static-shape SPIR-V strict GEMM issuer.**
Fixed 2x3x4 is only its control; declare/falsify the reusable positive rank-2 family,
complete graph/ABI/strict controls and independent target evidence. No source/runtime authority in this milestone.
[Research PR #84](https://github.com/onlyxItachi/MatcoreDSL/pull/84) remains draft;
the [d214 source-frontier audit](https://github.com/onlyxItachi/MatcoreDSL/blob/d214b017cad93ede7445148988f7aad36bd4573f/docs/mdslc/agent-reports/spirv-source-frontier-audit-v1.md)
separates issuer proof from later Vulkan failure/identity/artifact obligations. GPU pair qualification does not transfer.
