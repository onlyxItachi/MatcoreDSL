# MDSLC current state

Canonical maintenance checkpoint: `a90e82ae947efcae77d1b1c78c38d87ea4369884`, normal merge of
[PR #97](https://github.com/onlyxItachi/MatcoreDSL/pull/97) from exact premerge
`dd64a369d4a4c730d0db09aca1edf86a16cda149`.
The last compiler-capability checkpoint remains [PR #93](https://github.com/onlyxItachi/MatcoreDSL/pull/93),
`e8e8008cb61b61a585d8bc1584ca320c6326472d`.

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
Source-time M/N tile request 1..64 -> same strict issuer + pinned PIC relink kit
  -> localized request-bound DSO + persistent no-clobber .mdslc-private bundle
Source/entry/request/target/profile + produced LLVM -> typed final publication
Strict tiles retain full increasing scalar K and separate f32 multiply/add.
Reassociate permission -> separate AVX2/FMA realization (x64 only)
Opt-in strict lhs pair -> panel4 CPU / qualified NVVM / ROCDL (Linux x64)
GPU pair -> f1 guards; f2 invocation/completion/quarantine
Guarded registry -> ordered publication, owning observations, sticky failure
Matcore owns meaning/legality; MLIR transformation; LLVM machine lowering.
Legacy mutating GEMM and separate platform/Python lanes remain intact.
```

## Material change

PR97 makes synchronous/asynchronous Windows test diagnostics byte-preserving
under Python's UTF-8 default; it changes no compiler/runtime semantics or target
authority. Independent review and all 26 hosted checks passed; see the
[qualification record](agent-reports/windows-diagnostic-capture-v1.md).
The authenticated source-time HOW pattern remains governed by its
[bounded contract](SOURCE_TIME_GEMM_PATTERN_V1.md) and
[PR93 qualification](agent-reports/source-pattern-qualification-checkpoint-v1.md).

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
remain unchanged; this checkpoint provides no new GPU execution qualification.
Opaque imports/cross-region optimization remain unsupported. Caller lifetimes/race
freedom, runtime/allocation, trusted loading and provider/manual-link preconditions
remain, not sandbox or crash atomicity. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
is partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) design-only/open;
[#77](https://github.com/onlyxItachi/MatcoreDSL/issues/77) has no performance qualification.

## Exactly one next boundary

**[Issue #89](https://github.com/onlyxItachi/MatcoreDSL/issues/89): qualify a closed parameterized bounded static-shape SPIR-V strict GEMM issuer.**
This remains the independent frontier, not a dependency or consequence of CPU
source-time tiling. Falsify the reusable positive rank-2 family, complete graph/ABI/
strict controls and independent target evidence; fixed 2x3x4 is only a control.
No source/runtime authority. [Research PR #84](https://github.com/onlyxItachi/MatcoreDSL/pull/84)
remains draft and unchanged; its [source-frontier audit](https://github.com/onlyxItachi/MatcoreDSL/blob/d214b017cad93ede7445148988f7aad36bd4573f/docs/mdslc/agent-reports/spirv-source-frontier-audit-v1.md)
separates issuer proof from Vulkan failure/identity/artifact obligations.
