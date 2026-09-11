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
       +-> Vector/LLVM -> native x86 ISA variants
       +-> Vector/LLVM -> native ARM64
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
