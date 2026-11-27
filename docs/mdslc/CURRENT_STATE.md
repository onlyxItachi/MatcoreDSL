# MDSLC current state

Latest canonical engineering/evidence checkpoint:
`1c1221e89272eb2ad6ff058b8e26de4653f3a93d`, normal merge of
[PR #107](https://github.com/onlyxItachi/MatcoreDSL/pull/107).
The source/product feature checkpoint remains
`6d30a204aa6f7224b736c2035de15ce20b0e6b3d` ([PR #105](https://github.com/onlyxItachi/MatcoreDSL/pull/105)).
P5's executed producer/SDK stays frozen at
`1ad7fe56b5fcb10efe37e523206228e9ef303f6a`; the later merge is not P5's producer.
See the [integration record](agent-reports/source-peel-canonical-integration-v1.md).

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema and frozen source/header/toolchain closure
  -> immutable Values, all-MAY-alias Storage and per-operation numerics
  -> ordered checks/effects and exact original Matcore MLIR witness
Sealed Program -> orchestration; private plans -> replay/consumption proofs
Optional forwarding retains guards and invalidates MAY-alias bindings.
Existing 2–8 TU program path remains separate from source-time tile requests.
Original host -> sealed ABI thunk + intrinsically isolated helper LLVM
Built-in GEMM -> Linalg/One-Shot -> CPU schedules or GPU outlining
Explicit old/peeled source tiles -> pinned PIC kit + local DSO/no-clobber bundle
Source/entry/request/profile + actual artifacts -> typed final publication
Strict full/tail tiles retain increasing scalar K and separate f32 mul/add.
LLVM targets retain x64 ISA/AArch64 and bounded NVVM/ROCDL realizations.
Reassociate permission -> separate x64 AVX2/FMA realization
Opt-in strict lhs pair -> qualified CPU / NVVM / ROCDL routes
Private static SPIR-V -> checked shader bytes; no source/runtime candidate
Guarded execution -> ordered publication, owning observations, sticky failure
Matcore owns meaning/legality; upstream MLIR/LLVM own lowering.
Legacy mutating GEMM and separate platform/Python lanes remain intact.
```

## Material change and evidence

PR105 connects separately explicit `--gemm-output-tiles-peel-n M,N` requests
through existing authenticated ownership, with a distinct schema/binding domain.
Old/default requests and numerical/provider policy stay unchanged. Local Release
356/356 and committed x64 source ASan 4/4 passed; final source CI had all 26 checks
successful. Selected x64 baselines retained unrelated AVX512 skips; new source
gates had none. ARM source qualification is normal, not source-ASan.
See the [bounded qualification](agent-reports/source-output-peel-qualification-v1.md).
PR107's final head also passed all 26 fresh hosted checks; actual scopes and
explicit skips are retained in its [final CI record](agent-reports/source-peel-final-ci-v1.md).

[P5](../performance/cpu/source-pattern-p5-2026-11-10.md) is complete but
`INCONCLUSIVE_TIMING`: all 18 ratios null, 78/270 measured aggregates rejected
solely for nonzero involuntary switches, 0/15 pairs qualified. No win/loss ratio
or clock defect follows. P1 remains globally inconclusive with its one bounded
2.45145x old-5x3 loss unchanged; neither campaign is reclassified or repeated.

## Unsupported or unproven

Source-time tiles remain explicit baseline generated-strict, single-TU Linux
x64/AArch64 complete executables with none or checked forwarding, using coherent
Clang/LLVM/MLIR 21.1.8. No -c/program
tile selection, other ISA requests, fused-pair tile composition, executable imports,
cache reuse, post-publication relocation or automatic tuning. Caller lifetime/race,
allocator/runtime/trusted-loading preconditions and huge-empty/K=0 bypass remain.
No new GPU/Vulkan source mapping, residency/asynchrony, general rank-N/views,
Metal/NPU, generated-region Windows, universal interchange or performance policy.
Existing GPU work/failure/sanitizer limits and private SPIR-V authority limits stay.
Experimental API/ABI is not frozen. Issue #15 remains partial/open; #20 design-only/open.

## Exactly one next boundary

**[P6 / Issue #108: qualify one source-call timing method](https://github.com/onlyxItachi/MatcoreDSL/issues/108).**
Test a fixed wall-clock method, including scheduling delays, against an identical
binary/argv pair and verified extra CPU work. Freeze what is timed, allocation/
returned-ownership lifetime, sample inclusion and order before fresh data; use a
fresh confirmation block and independent raw-data audit. The method may fail.
Even a pass cannot rescue P1/P5 or establish a source-variant speedup, ranking or
automatic policy. No P6 implementation or execution is recorded.
