# MDSLC current state

Canonical engineering checkpoint: `1f7c491f230251d04d42b5d4a0af5eaf29bcb685`, normal merge of
[PR #103](https://github.com/onlyxItachi/MatcoreDSL/pull/103) from exact frozen head
`dfc35f9e13653db16c7f9a825db2bb1b2a82ea22` into main `52fb20a...` (PR #100).
The merge tree is exactly the qualified private tree
`c07a780e28ed14b1669e4a50c898203cce22cac3`; no source/default route is switched.

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
Private full-N/tail M/N tiles 1..64 -> verified strict recipe; no source selection
LLVM targets: x64 ISA variants, ARM64; NVVM sm_89 / ROCDL gfx1150
Private static SPIR-V issuer -> checked shader bytes; no source/runtime route
Existing unpeeled source-time tiles -> pinned PIC kit + local DSO/no-clobber bundle
Source/entry/request/target/profile + produced LLVM -> typed final publication
Strict tiles retain full increasing scalar K and separate f32 multiply/add.
Reassociate permission -> separate AVX2/FMA realization (x64 only)
Opt-in strict lhs pair -> panel4 CPU / qualified NVVM / ROCDL (Linux x64)
Guarded registry -> ordered publication, owning observations, sticky failure
Matcore owns meaning/legality; MLIR transformation; LLVM machine lowering.
Legacy mutating GEMM and separate platform/Python lanes remain intact.
```

## Material change

PR103 adds the separately explicit private `--output-tiles-peel-n=M,N` recipe,
using upstream full-N/tail peeling, branch-aware proof before exact replay and a
static no-tail identity for unit N. Strict increasing K, separate f32 multiply/add,
destination ownership and old artifacts are preserved. Clean Release: 352/352,
zero skips; additive private family: 44/44; all 26 hosted checks succeeded,
including native ARM. See the [qualification record](agent-reports/strict-output-peel-integration-qualification-v1.md).
This is private correctness/ownership, not source integration or speed evidence.
PR100's [P1 campaign](../performance/cpu/source-pattern-p1-2026-11-10.md) remains
**inconclusive** overall; its sole qualified stock comparison at M/K/N=33/35/37
found 5x3 **2.45145x slower**, not a win or parity. Defaults/provider policy stay intact.

## Unsupported or unproven

Experimental API/ABI and coherent 21.1.8 remain. Source-time tiles support only
explicit baseline `generated-strict` single-TU Linux x64/AArch64 executables with
`none` or checked forwarding. No `-c`, program-mode/per-operation selection,
other ISA candidates, fused-pair composition, executable IR imports or cache reuse.
Keep the emitted private bundle and runtime paths; post-emission relocation is
unsupported. No automatic tuning, cost model, GPU tile mapping or performance win.
The new private peeled recipe has no canonical source-time selection yet;
PR105 is under qualification, not merged support. Raw leaf preconditions remain
distinct from the adapter's conditional nonempty-byte guards and empty/K=0 bypass.
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

**[Issue #104](https://github.com/onlyxItachi/MatcoreDSL/issues/104): qualify separately explicit authenticated source-time selection.**
Keep old requests/defaults unchanged; bind the new realization through source,
exact full/tail consumption, private DSO, typed compilation and final publication.
Require source numerical/effect/refusal, generated sanitizer and relocated
producer-inaccessible installation evidence, not just raw-leaf passes.
[PR #105](https://github.com/onlyxItachi/MatcoreDSL/pull/105) is that pending gate.
Later comparison-harness preparation is not timing or qualification; see the
[performance roadmap](PERFORMANCE_EVIDENCE_ROADMAP_V1.md).
PR #84 remains the unchanged draft research track, not Vulkan execution authority.
