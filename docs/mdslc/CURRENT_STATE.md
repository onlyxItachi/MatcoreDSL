# MDSLC current state

Engineering checkpoint: canonical merge
`1c8f534bba81760aaa211f13b0695254c1d93e60`, [PR #73](https://github.com/onlyxItachi/MatcoreDSL/pull/73).
This identifies the latest engineering merge; documentation-only updates may follow.

## Architecture

```text
Ordinary C++ host + explicit closed mathematical regions
  -> Clang/Sema admission, source-visible helpers, frozen source/header/toolchain
  -> frontend-neutral immutable values + separate all-MAY-alias host resources
  -> dynamic shapes, per-operation numerics, ordered checks/effect frontiers
  -> exact untransformed Matcore MLIR paired witness
  -> static orchestration from the sealed semantic Program (no interpreter)
2–8 original source TUs -> per-file authentication -> checked host program link
Original Clang host -> sealed ABI-checked entry thunk + isolated helper LLVM
Strict GEMM primitive -> verified Linalg/One-Shot -> optional Transform M/K/N
Permitted GEMM -> separate Transform/Linalg/Vector AVX2/FMA (x64 only)
  -> LLVM strict x64 baseline/AVX/AVX2/AVX512F or ARM64; hardware/OS guards
Trusted registry -> generated/native strict or numerically legal legacy/provider
  -> private candidate output, checked completion/FP state, immutable value
  -> ordered host publication, owning observation, sticky failure prefix
Reads snapshot at their frontier; snapshots are realization, not value identity.
Private candidate DSO owns implementation; canonical Runtime owns provider policy.
Installed mdslc-region pins artifacts; coherent dynamic loading remains trusted.
Matcore owns meaning/legality; MLIR transformations; LLVM machine lowering.
Legacy mutating GEMM -> IR v1/MLIR -> existing CPU/runtime route is unchanged.
Linux x64/ARM64 regions, standalone Windows and Linux Python/JIT are separate lanes.
```

## Material change

Three explicitly forced strict ISA candidates now preserve the same mathematical
and structured recipe while producing actual YMM/ZMM arithmetic. Exact CPU/OS
guards, private ownership and wide ASan controls accompany them; default dispatch
and numerical permissions are unchanged. Hosted Release **229/229** and
**230/230**, plus native ARM **97/97** and **99/99**, passed without skips.
Global ASan/UBSan had **161 passes + 14 eligibility skips** on a different runner;
local instrumented AVX512F execution is separate evidence. All qualifying hosted
lanes passed. See [exact evidence and exclusions](agent-reports/strict-cpu-isa-candidates-v1.md#qualified-canonical-checkpoint).

## Unsupported or unproven

Experimental region execution is native Linux x86-64 / ARM64, coherent 21.1.8; syntax and
API/ABI are unfrozen. No transformed whole-region execution, fusion, automatic
reuse, general views, asynchronous/device/export effects or generated-region Windows.
No canonical GPU/NPU, general rank-N execution, zero-copy, universal performance or
BLAS-parity claim. These three recipes do not qualify every AVX extension or AMX.
ARM legacy/provider and reassociate candidates, SVE/SME and
cross-compiled execution remain unsupported. Opaque mathematical imports and cross-region optimization are unsupported. Valid caller
objects/lifetimes, race-free storage, conforming allocation and trusted library
loading remain preconditions; this is not a sandbox or crash-atomic transaction.
Manual archive/object linking does not inherit the complete driver contract.
Uncoordinated provider users/duplicate adapters are outside shared policy exclusion.
Provider conformance is bounded, not universal. [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15)
remains partial/open; [#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) remains design-only/open.

## Exactly one next boundary

**Complete staged GPU candidate qualification and canonical integration.**
The same checked private GEMM boundary must preserve source meaning while owning
fallible device resources, transfers and completion. [PR #72](https://github.com/onlyxItachi/MatcoreDSL/pull/72)
is a reviewed CPU/GPU composition with corrected-head local **379/379** passing,
awaiting final hosted qualification; it remains noncanonical. No fusion, residency language, default
GPU selection or performance policy follows from that integration.
