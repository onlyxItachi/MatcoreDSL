# MDSLC current state

Canonical engineering checkpoint: `243f6772ebf8cc7600d05776a2adbb8d8a0515d0`, normal merge of
[PR #99](https://github.com/onlyxItachi/MatcoreDSL/pull/99) from exact premerge
`06affd2ffea4a5459158a7ddcd0c4e75009ea90e`; their Git trees are identical.
Its Windows-test prerequisite is [PR #97](https://github.com/onlyxItachi/MatcoreDSL/pull/97),
merge `a90e82ae947efcae77d1b1c78c38d87ea4369884`.

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

PR99 closes [#89](https://github.com/onlyxItachi/MatcoreDSL/issues/89)'s bounded
positive-static strict-f32 SPIR-V issuer: complete graph/ABI/strict-control and
serialized-byte checks, independent adversarial review, AMD RADV physical samples
and explicit unsupported-device refusals. It is default-OFF, private and not
installed or source-connected. See the [contract](SPIRV_CLOSED_ISSUER_V1.md) and
[qualification](agent-reports/spirv-issuer-integration-qualification-v1.md).
PR97's [diagnostic transport fix](agent-reports/windows-diagnostic-capture-v1.md)
changes test byte handling only. Existing CPU/GPU/provider routes are unchanged.

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

**[Issue #98](https://github.com/onlyxItachi/MatcoreDSL/issues/98): qualify authenticated strict source-time CPU tile performance evidence.**
The existing correct source/instance/ownership route now permits a matched
stock-versus-requested-tile experiment; more operations are not prerequisites.
Keep full call/lifetime accounting, artifact identity and independent output
checks, including losses and inconclusive outcomes. [PR #100](https://github.com/onlyxItachi/MatcoreDSL/pull/100)
records a bounded negative/inconclusive campaign pending canonical integration.
See the [performance roadmap](PERFORMANCE_EVIDENCE_ROADMAP_V1.md). No tuning or
default change follows. [PR #84](https://github.com/onlyxItachi/MatcoreDSL/pull/84)
remains the unchanged draft research track, not a Vulkan execution authority.
