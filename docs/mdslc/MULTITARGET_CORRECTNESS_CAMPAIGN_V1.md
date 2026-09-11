# Correctness-first MLIR multi-target campaign

## Starting checkpoint and authority

Canonical `main`: `9c2149a25e12f5192c168f3e047097505263fc1a` (PR #69,
documentation checkpoint after engineering PR #67). Local/GitHub identity,
clean main, no open PRs and successful native/Windows/hygiene/legacy workflows
were checked on 2026-09-11 before this campaign. Existing worktrees are preserved.

The owner authorized a correctness-first multi-target implementation campaign,
normal GitHub integration, independent agents and target-specific MLIR recipes.
Performance and BLAS parity are not acceptance gates. Target support remains a
claim earned by source-to-artifact-to-execution evidence, not by a dialect name.

## Product flow

```text
Authenticated C++-hosted mathematical regions
  -> authoritative frontend-neutral values/resources/effect/numerical contracts
  -> legal structured Tensor/Linalg candidate
  -> explicit, reusable target lowering recipes
       +-> Linalg/Transform/LLVM -> native x86 ISA variants
       +-> Linalg/LLVM -> native ARM64
       +-> GPU/NVVM/LLVM -> NVIDIA device binary
       +-> GPU/ROCDL/LLVM AMDGPU -> AMD device binary
       +-> separate optional Metal feasibility experiment
  -> checked candidate execution, completion and private result adoption
  -> original ordered publication/observation/failure trace
```

This is the campaign design, not a list of already supported targets. The
whole-region paired semantic witness remains authoritative and is not loosened
to admit arbitrary transformed input. Generated artifacts are compiler-issued;
importing MLIR, matching a symbol or supplying a function pointer does not grant
execution authority. Existing native/provider execution and default selection
remain intact while new candidates are explicitly forced and validated.

## Independent lanes

| Lane | Initial evidence target | Integration gate |
|---|---|---|
| Common contract | Reuse the existing strict FP32 GEMM semantic/structured issuer and private output boundary | No duplicated mathematical definition or lost effect frontier |
| NVIDIA | Actual RTX 4060 Laptop `sm_89`, exact documented MLIR/NVVM path | Generated code executes independent-oracle and adversarial cases |
| AMD | Actual Radeon 890M `gfx1150`, ROCDL/AMDGPU path | Real AMD execution, no CPU fallback or architecture spoofing |
| x86 ISA | Baseline, AVX/AVX2 and supported AVX-512 subsets | Hardware + OS state + emitted implementation + real execution; strict does not acquire FMA |
| ARM64 | Native ARM CPU CI and coherent toolchain | Host FP environment, artifact identity, driver, package and generated execution contracts |
| Metal | Capability-probed, optional CI experiment | Compile-only or skipped is never reported as GPU execution |

NVIDIA and AMD prototype branches should falsify the same structured kernel
contract before a shared production issuer is chosen. Target scheduling is not
encoded into mathematical semantics. A simple legal schedule is acceptable;
optimizing its speed is a later campaign.

## Initial bounded execution contract

Start with existing dense row-major rank-2 FP32 GEMM, retaining dynamic M/N/K,
zero extents, operand orientation and separate increasing-K multiply/add for
strict numerics. Broaden common operations only after the cross-target seam is
real. Reassociation, contraction, reduced precision and device math functions
need their own explicit numerical permissions and tests.

For the first GPU candidate, inputs are immutable host values and GPU allocations
are private candidate resources. Host/device staging must be explicit in the
selected candidate contract and inspectable evidence. This is not persistent
device residency, zero-copy, asynchronous source execution or arbitrary device
views. Transfers, launches and completion are fallible preparation; a failed
candidate must never publish its partially computed output as a semantic value.
Earlier source publications survive later failure. Resource release must not
race outstanding work. Target failure status must not silently trigger fallback.

No strict success criterion may be weakened to an arbitrary epsilon comparison.
Use explicit operation-level numerical contracts, independently computed
oracles, bit-sensitive cases where promised, and documented treatment of NaN,
infinity, signed zero, subnormals, rounding, contraction and accumulation order.

## Falsification and acceptance

- Rectangular/noncommuting inputs; both lhs/rhs carried values; aliased external
  descriptors and late reads; dynamic/empty/zero-K/overflow geometries.
- Cancellation and adversarial floating-point values, plus a no-FMA counterexample.
- Forced unavailable device/ISA, missing artifact, wrong target/entry/ABI and
  forged or changed source/artifact inputs fail closed.
- Copy/allocation/launch/completion failure does not adopt a partial value or
  corrupt the prior publication prefix; no lifetime ends before completion.
- The generated artifact actually uses the selected target, and execution runs
  on that target. CPU numerical emulation does not count as GPU validation.
- Existing frontend, semantic/structured verifier, CPU/provider, ABI, package,
  multi-source, Windows and legacy regression surfaces remain supported.
- Narrow tests first, then affected/full regression, independent adversarial
  review, hosted CI and normal PR merge. Each merged engineering boundary gets
  a compact `CURRENT_STATE.md` follow-up with the exact merge and one frontier.

## Resource and dependency discipline

Use existing documented coherent toolchains first. Do not build LLVM or Rust
from source or mutate system dependencies to make a speculative route work.
Product semantic tooling remains coherent 21.1.8 unless a separately evidenced
migration is justified. Experimental target tools must have their identity and
compatibility recorded instead of being silently mixed into the frontend.

Build directories and `TMPDIR` live on SSD-backed task directories, never the
quota-limited `/tmp` tmpfs. Start with at least 20 GiB available SSD space and
check available memory; main builds use at most two compilation jobs and heavy
links one. Agents use at most one small compilation job and coordinate heavy
work. Record failures and unsupported paths without destructive cleanup.

## Initial disposition

PROPOSED: the campaign and target recipes above. OBSERVED: the canonical state,
local x86 feature advertisement, NVIDIA sm_89 and AMD gfx1150 device enumeration.
UNSUPPORTED: new MDSLC AMD/NVIDIA/ARM/Metal execution until the implementation
and its full source/runtime contract pass their acceptance gates. No new
performance, fusion, zero-copy or BLAS-parity claim is made by this record.

## Reconciliation: one mathematical seam, distinct qualified realizations

The initial disposition above is historical. The following records what the
implementation and adversarial experiments actually established; canonical
engineering merges are identified by [CURRENT_STATE](CURRENT_STATE.md) and the
linked PRs, not inferred from a local build.

The source model did not change. Original C++ admission produces a sealed
frontend-neutral Program and an exact **untransformed** Matcore MLIR witness.
Static host orchestration retains each read, check, publication, observation and
failure frontier. Compiler-owned isolated GEMM semantics traverse verified
Tensor/Linalg/One-Shot and upstream target lowering. These generic dynamic-shape
kernels are issued when MDSLC is built and selected by the source compiler's
private candidate registry; this is **not** source-specific GPU-region compilation.
No handwritten GPU GEMM or arbitrary imported IR substitutes for that seam.

| Realization | Actual evidence | Deliberate limit |
| --- | --- | --- |
| Strict AVX | Same row-contiguous semantic/structured recipe; actual YMM multiply/add; local physical arithmetic, alias, source and ASan controls | Full CPU/OS feature closure required; not every AVX extension |
| Strict AVX2 | Same pre-LLVM recipe; actual YMM arithmetic and AVX2 register-source broadcast | No FMA permission or automatic policy change |
| Strict AVX512F | Actual ZMM multiply/add and scalar tails; physical full-width/FMA-discriminator and READ64 ASan controls | Requires OS-enabled opmask/ZMM state and exact feature closure; not all AVX512 subsets or AMX |
| Native ARM64 | Initial hosted scalar 93/93 and row 95/95; final composition 97/97 and 99/99, all no-skip; installed-source execution and complete FPCR/FPSR preservation | Strict native/generated/automatic only; no ARM provider/reassociate, SVE/SME or cross-compiled execution claim |
| NVVM `sm_89` | RTX 4060 Laptop: 76 physical cases, 17761 output comparisons; source and installed driver, real CUDA memcheck, independent mocked failures | Explicit synchronous staging; exact CUDA/toolchain tuple; no tensor-core, fusion or residency claim |
| ROCDL `gfx1150` | Radeon 890M: 76 physical cases, 17761 output comparisons; source and installed driver, normal/host-UBSan execution and independent mocked failures | Exact HIP/toolchain tuple; real host-ASan failure remains unqualified and configure-refused |
| Optional Metal research | Real hosted Apple Paravirtual Metal execution: each of two separately compiled variants had 24 mismatches out of 80 strict-f32 comparisons | Strict numerical contract falsified; no product candidate, bare-metal Apple qualification or silent relaxation |

Target details and raw claim boundaries:
[CPU ISA](agent-reports/strict-cpu-isa-candidates-v1.md),
[ARM](agent-reports/arm64-strict-integration-v1.md),
[staged GPU contract](STAGED_GPU_CANDIDATES_V1.md),
[GPU integration](agent-reports/staged-gpu-integration-v1.md),
[composition](agent-reports/multitarget-composed-v1.md), and
[preserved Metal research at 564bef0](https://github.com/onlyxItachi/MatcoreDSL/blob/564bef047e924cbc932085a06f0860a0bb1c253d/docs/mdslc/agent-reports/multitarget-metal-feasibility-v1.md).

The GPU work bounds (`M,N,K <= 65535`, at most `2^20` outputs and `2^26` scalar
products) constrain this realization, not mathematical tensor shapes or planner
crossover thresholds. Every forced target's availability and legality guards
precede result allocation and empty/zero-K shortcuts. Default selection and the
existing CPU/provider execution route remain unchanged.

### Canonical integration and preserved heads

All three engineering boundaries passed independent adversarial review and
complete qualifying hosted workflows before normal merge, in dependency order:

| Boundary | PR | Preserved pre-merge head | Canonical engineering merge |
| --- | --- | --- | --- |
| Native ARM64 | [#71](https://github.com/onlyxItachi/MatcoreDSL/pull/71) | `8157eabed52009038b5dfe584d9d7c3d52996b97` | `f69f13ac17e05da7b26f9f1d948a880b168f1265` |
| Strict CPU ISA | [#73](https://github.com/onlyxItachi/MatcoreDSL/pull/73) | `665cf3756c58196b5f58acc2f784e92f7b94b52c` | `1c8f534bba81760aaa211f13b0695254c1d93e60` |
| Composed staged GPU | [#72](https://github.com/onlyxItachi/MatcoreDSL/pull/72) | `4fc4560ce16e5c4d9be070fcb465ce2ced317dd2` | `0fe537b4246510c57d801577a8d915876e40a941` |

Operator checkpoint follow-ups #74/#75 recorded the ARM/CPU merges immediately
after each boundary. This final documentation follow-up records GPU integration
without changing its qualified production tree. Normal ancestry and the separate
research branches are preserved; no failed experiment was rewritten away.

The final frozen combined local suite passed **379/379**, zero skips. All eight
native hosted jobs, native ARM scalar/row, Windows, legacy CI and hygiene passed.
Global ASan/UBSan had **161 passes + 14 explicit eligibility skips**, not a
full no-skip sanitizer claim. Windows standalone had **32 passes + one skip**
and a separate **1/1** ASan GEMM. [Exact checkout binding, counts, skips, hashes
and installed execution](agent-reports/multitarget-composed-v1.md#qualified-canonical-checkpoint)
are retained separately from physical GPU and negative Metal evidence.

### Failure and integration counterexamples retained

- **ARM qualification:** the first real ARM run failed two x86 ABI assumptions,
  one wrong-shape refusal fixture and one unsupported legacy-API expectation.
  The repair tests the actual native ABI, supplies valid shapes before candidate
  refusal, and requires the correct unsupported legacy result. Neither
  production admission nor no-skip qualification was weakened.
- **ISA feature closure is not numerical permission:** disabling FMA hardware
  support also disabled AVX512F in the tested LLVM feature model. The surviving
  design retains required hardware facts but forbids fused arithmetic in strict
  IR and generated instructions. Wider registers do not authorize reassociation.
- **GPU completion is not inferred from an error:** potentially live device and
  host staging resources are quarantined after unproven completion, and the
  adapter is poisoned. Cleanup failure cannot publish a partial result. Caller
  GPU/TLS/FP/errno state is isolated through the documented joined-worker route.
- **Private identity collision:** independently developed CPU/GPU enum additions
  occupied the same new slots. The composition preserves CPU-first canonical
  assignments, appends GPU values consistently in all three vocabularies and
  mechanically checks them. Unmatched private artifacts may not be mixed; this
  is not a public ABI freeze.
- **A cached configuration was insufficient evidence:** GPU source tests initially
  registered an empty Python command in clean hosted builds. Local explicit
  Python configuration masked that defect. Own-scope interpreter discovery and
  a fresh unseeded build plus actual source-test execution correct it; the
  original Not Run failures remain recorded in the
  [focused correction](agent-reports/gpu-test-discovery-fix-v1.md).
- **Metal precision controls did not preserve the promised arithmetic:** static
  SPIR-V/MSL compilation and normal-value examples passed, but subnormals flushed
  on the tested paravirtual device. No relaxation of `strict_f32` or
  `reassociate_f32`, software emulation, or green-research-CI support claim was
  used to conceal that result. Dynamic descriptor lowering also encountered a
  separately recorded upstream static-stride restriction.

The compiler also refuses unavailable targets, spoofed HIP architecture state,
wrong image/DSO identities, hostile source symbol replacements, illegal shapes,
and unsupported host-sanitizer profiles. Synthetic mutation/fault tests, actual
physical execution, instrumented host execution and vendor device-memory checks
remain distinct evidence classes. Valid caller objects/lifetimes, race freedom,
conforming runtimes/allocators and trusted dynamic loading remain preconditions;
this is not a sandbox or a device-crash recovery guarantee.

### Human-readable real source control

After the local 335/335 execution pass at ARM/GPU head
`65b6e9eacc145bd514e416bced3513655ebd7db0`, the unchanged
[`two_gemm.mdsl`](../../compiler/examples/experimental/two_gemm.mdsl) compiled
through each forced GPU candidate. Both normal runs returned 0 and printed
`22 28 49 64 -> 120 156 49 64`. Both late-failure runs returned 0 after verifying
the retained first publication/observation and untouched second destination.
Its separate fresh-configuration defect remained until `4fc4560`; these executed
examples do not erase that failed gate. Source SHA-256:
`545b229311872d31180b30fa76210d96827b9100a14c8f0ea250d030d0bc475a`.
This complements the adversarial fixtures; it does not demonstrate fusion or
persistent device-local intermediates.

### Ownership and exactly one next frontier

Matcore owns mathematical values, required checks, numerical permissions and
observable/failure traces. MLIR supplies structured transformations and target
dialect conversions; LLVM and vendor tools own machine lowering. The runtime
owns checked realization resources, capability decisions and completion, while
external providers retain their authenticated numerical/execution contract.
No observations became cost-model thresholds. No performance experiment in this
campaign establishes a new planner choice, BLAS parity or optimized schedule.
Issue #15 remains partial/open and #20 remains design-only/open.

The next frontier is **one authenticated publication-to-read forwarding
derivation**. Reuse a retained immutable value only after a dominating successful
publication to the same checked resource/version, with no intervening possibly
aliasing write. Retain read/shape/failure guards and ordered effect frontiers;
do not assume that descriptor inequality proves disjointness. Apply existing
[resource contract 8](FOUNDATION_RESOURCE_DECISION_V1.md#semantic-contract-selected-for-the-adapter):
allocation-exhaustion opportunities may disappear, but required checks and the
permitted completed effect prefixes must remain. Prove that the particular
forwarding derivation satisfies that rule rather than reopening the language.
Falsifiers must include lhs/rhs noncommuting dependencies, aliased descriptors,
late reads, observed intermediates and late failure across the existing candidate
mechanisms. This is a target-independent first cross-operation optimization,
not broad fusion, a device-memory language or a performance policy.
