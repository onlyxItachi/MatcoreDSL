# MDSLC current state

Canonical engineering checkpoint: `52fb20a30cae0d001daebf5f71035ba071541181`, normal merge of
[PR #100](https://github.com/onlyxItachi/MatcoreDSL/pull/100) from exact premerge
`b52f068b53ca09045b7044beb9acd67faec2be4d`. Its harness/report files are unchanged
by merge; main also retains [PR #99](https://github.com/onlyxItachi/MatcoreDSL/pull/99),
`243f6772ebf8cc7600d05776a2adbb8d8a0515d0`, and Windows prerequisite PR #97.

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema, source-visible helpers, frozen source/header/toolchain closure
  -> immutable values, all-MAY-alias Storage, dynamic shapes, per-op numerics
  -> ordered checks/effects + exact untransformed Matcore MLIR witness
Sealed Program -> static orchestration; private plans -> replay/consumption checks
Forwarding is opt-in; publications/branches invalidate MAY-alias bindings.
Existing 2–8 TU program -> authenticated sources + checked host link
Original frozen host -> sealed ABI thunk + intrinsically isolated helper LLVM
Built-in GEMM -> Linalg/One-Shot -> CPU schedules / GPU outlining
LLVM targets: x64 ISA variants, ARM64; NVVM sm_89 / ROCDL gfx1150
Private static SPIR-V issuer -> checked shader bytes; no source/runtime route
Source-time M/N tile request 1..64 -> same strict issuer + pinned PIC relink kit
  -> localized request-bound DSO + persistent no-clobber .mdslc-private bundle
Source/entry/request/target/profile + produced LLVM -> typed final publication
Strict tiles retain full increasing scalar K and separate f32 multiply/add.
Reassociate permission -> separate AVX2/FMA realization (x64 only)
Opt-in strict lhs pair -> panel4 CPU / qualified NVVM / ROCDL (Linux x64)
Guarded registry -> ordered publication, owning observations, sticky failure
Matcore owns meaning/legality; MLIR transformation; LLVM machine lowering.
Legacy mutating GEMM and separate platform/Python lanes remain intact.
```

## Material change

PR100 adds an authenticated source-call performance harness and retains one
fixed campaign: all correctness/identity gates passed, overall timing remains
**inconclusive**. The only qualified stock comparison, M/K/N=33/35/37, found
5x3 taking **2.45145x** the stock interval. This is a loss, not a speedup or parity.
All 26 hosted checks passed. See the [reviewed evidence](../performance/cpu/source-pattern-p1-2026-11-10.md).
No compiler/runtime/default/provider behavior changed. Earlier PR99's private
SPIR-V [contract](SPIRV_CLOSED_ISSUER_V1.md) and
[qualification](agent-reports/spirv-issuer-integration-qualification-v1.md) remain separate.

## Unsupported or unproven

Experimental API/ABI and coherent 21.1.8 remain. Source-time tiles support only
explicit baseline `generated-strict` single-TU Linux x64/AArch64 executables with
`none` or checked forwarding. No `-c`, program-mode/per-operation selection,
other ISA candidates, fused-pair composition, executable IR imports or cache reuse.
Keep the emitted private bundle and runtime paths; post-emission relocation is
unsupported. No automatic tuning, cost model, GPU tile mapping or performance win.
No parity, general fusion/rank-N/views, automatic reuse, residency/asynchrony,
zero-copy, Metal/NPU, generated-region Windows or universal compiler interchange.
Existing [GPU target/work/failure and sanitizer limits](STAGED_GPU_CANDIDATES_V1.md)
remain unchanged. New SPIR-V physical samples are test-only: no Vulkan source
candidate, production completion/quarantine protocol, dynamic/zero issuer cases,
full-envelope physical qualification, ICD-chain/install authority or performance.
Opaque imports/cross-region optimization remain unsupported. Caller lifetimes/race
freedom, runtime/allocation, trusted loading and provider/manual-link preconditions
remain, not sandbox or crash atomicity. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
is partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) design-only/open;
[#77](https://github.com/onlyxItachi/MatcoreDSL/issues/77) has no performance qualification.

## Exactly one next boundary

**[Issue #102](https://github.com/onlyxItachi/MatcoreDSL/issues/102): qualify a separate private strict full-N/tail realization.**
The [reviewed compile-only trace](https://github.com/onlyxItachi/MatcoreDSL/blob/4f3cd4c/docs/mdslc/agent-reports/strict-output-tile-peeling-prototype-v1.md)
shows upstream peeling can expose register accumulators for 5x3; it also changes
inlining/code size, so no speed inference follows. Before adoption, independently
prove both domains and test full/tail numerical, alias, empty and sanitizer cases.
Keep v1/source/default routes unchanged; source connection and new timings are
later gates. See the [performance roadmap](PERFORMANCE_EVIDENCE_ROADMAP_V1.md).
PR #84 remains the unchanged draft research track, not Vulkan execution authority.
