# Closed SPIR-V issuer integration qualification

## Scope and provenance

Started from canonical main `b2597bed52de46604574f76eb33663788a5ab994`,
verified against the live remote on 2026-11-10. Main was clean and remained
untouched during implementation. Isolated branch:
`mdslc/spirv-closed-issuer-v1`. Existing unrelated worktrees were preserved.
Research PR #84 remains draft at
`5dc14db686b6b2668a2d3a47f5ef24c0e2bb4b97`; no research history was adopted.

This addresses Issue #89's compiler-private issuer boundary, not a Vulkan
source/runtime backend. The [contract](../SPIRV_CLOSED_ISSUER_V1.md),
[implementation record](spirv-closed-issuer-implementation-v1.md),
[independent review](spirv-closed-issuer-independent-review-v1.md) and
[physical harness](spirv-physical-harness-v1.md) distinguish the evidence layers.
The owner-selected performance follow-up is separately tracked by Issue #98;
additional mathematical operations are not prerequisites for measuring an
already qualified operation/realization.

## Integration and falsification

The root reused the existing strict CPU stage constructor with a checked static
specialization, connected the upstream GPU/SPIR-V pipeline, and added default-OFF
private build/inspection targets. No source registry, runtime adapter, installed
API, target default or numerical permission changed. The CLI accepts dimensions
only and publishes new no-clobber inspection directories after replay.

Independent review challenged native properties as well as ordinary attributes:
unproved `gpu.block_id` upper bounds, nontemporal memory properties, aligned
loads/stores, member relaxed precision, entry interfaces, index overflow and
decoded loop forwarding. Tests first require valid upstream MLIR where claimed;
an ill-typed mutant is not counted as a successful semantic falsification.

The actual serialized words are deserialized with pinned MLIR and structurally
checked. Its exact two-block zero-forwarding representation is admitted without
rewriting the graph. Six additional valid decoded-graph mutations challenge
seed identity, builtin-controlled seeds, K, progress, extra predecessors and
fresh-zero bypass. Artifact replay alone is not the independent graph proof.

Three independent mutants (load/store Aligned8 and member RelaxedPrecision)
were exported by the final adversarial executable: 59/59 checks passed including
three export checks. All three binaries also passed external `spirv-val
--target-env vulkan1.2`, while the closed classifier refused them. Their files
are retained under `build-release/valid-mutants-v1/` in the workspace below.

## Local environment and evidence location

Workspace: `/home/hamza-usta/mdslc-work/correctness-campaign-v1/`.
One shared `build-release` uses coherent Clang/LLVM/MLIR 21.1.8,
Release, native/bootstrap frontends, Matcore MLIR, experimental regions,
GPU issuer, SPIR-V issuer and OpenBLAS 0.3.32. NVVM/ROCDL runtime execution
options are OFF in this build. Two compile jobs maximum; no LLVM source build,
system package change or deletion of previous evidence occurred.

MLIR prefix:
`/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`.
Independent SPIRV-Tools came from the already available
`mdslc-work/multitarget-v1/builds/metal/tool-root/usr/bin`; version output
`v2026.1 unknown hash, 2026-02-06` is retained with binary hashes. Tool location
does not confer Metal qualification.

The frozen focused run at `76bbebd` passed 5/5 CTests:
148 issuer checks; 31 initial adversarial checks; 19 software-oracle controls;
36 initial synthetic receipt checks; 15 shape pairs plus 30 CLI refusals.
`build-release/focused-76bbebd.log` preserves that exact earlier checkpoint.
Subsequent decoded controls raise ordinary adversarial checks to 56; refusal
reason/property and diagnostic hardening raises synthetic receipt checks to 56.
No device dispatch belongs to these unit/synthetic counts.

External report `tests/generated_spirv/external-validation/run-5k3pwotj/report.json`
has SHA-256
`2303bdfce79f6597bb353bc9ca846693521c903f2574a7c18589956cbb1544a9`.
Every shader pair used by the physical campaign matches an externally validated
pair byte-for-byte. External validation completed before physical dispatch.
Maximum-storage/K compilation controls were not physically executed.

## Physical samples and negative targets

The process-only probe freezes LLVM APFloat nearest-even separate multiply/add
expectations before loading Vulkan. Independent integer/rational anchors cover
noncommuting rectangular indexing, last-K terms, subnormal input/product,
inexact underflow, negative values, positive-zero initialization, non-FMA,
reduction order, overflow, ties, Inf and NaN class. It retains immutable-input
and output-canary checks, three disjoint coherent allocations, both dispatches,
barriers and checked fence completion. Unsupported cases refuse, never fall back.

The first campaign at `76bbebd`:

| Evidence | Exact observed result |
| --- | --- |
| Device | AMD Radeon Graphics, RADV STRIX1, device `1002:150e`, Mesa 26.0.8-1ubuntu0.3 |
| Shapes (M,K,N) | (2,3,4), (1,1,1), (1,7,9), (9,7,1), (3,5,7), (17,31,13), (65,3,67), (65535,1,1), (1,1,65535), (3,2,5), (1,257,1), (5,1,7) |
| Physical checks | 143 cases/completions, 286 dispatches, 350098 outputs, 301116 input checks, 18304 canaries, zero mismatches |
| K<3 | The two index/base cases execute; 17 K>=3 arithmetic controls are explicitly skipped, not counted as passing |
| NVIDIA RTX 4060 Laptop | Required f32 denormal-preservation property absent; zero dispatch/completion, no fallback |
| llvmpipe | Required f32 denormal-preservation property absent; zero dispatch/completion, no hardware claim |

Frozen local summaries and SHA-256:

- `physical-amd-v1/summary.json`:
  `f0c0e2545b948b55a97aa09964becbc33739f38c94165b9a56bd76b8a051341a`.
- `refusal-nvidia-v1/summary.json`:
  `b0c731593d7b01b2a9747a5184ac461d71a17a631b8a5754240303a318f0f4f8`.
- `refusal-llvmpipe-v1/summary.json`:
  `82a33fe4b62287bcb21c41a37977dc8e56f041e01c0547608386466e9fc1a891`.

The independent reviewer revalidated all 83 command receipts, all source/catalog
hashes, empty stderr logs and the 24 physical shader identities. A low synthetic
receipt weakness was corrected: target refusal reasons must agree with actual
typed device properties, and nonempty diagnostics cannot qualify. The old real
receipts already satisfy the stronger rule. Fresh v2 campaigns at `d1d944d`
passed the same 35/24/24 commands, same physical counts and zero mismatches:

- `physical-amd-v2/summary.json`:
  `ddebc6154f076fd48bec818aae426f22f8606c40c982e32984d737e0acad401b`.
- `refusal-nvidia-v2/summary.json`:
  `6e7178f241cfb40a2ac61ebff296fdf7fdcdb63e95beda7c3b00b3b6be1a7559`.
- `refusal-llvmpipe-v2/summary.json`:
  `408b528f32a8f38f1e72e0fe3e504312275a403c8b6f432ba16da472e83fdd47`.

The independent reviewer revalidated these 83 new receipts, all 148/104/104
catalog files, source/probe identities and upstream shader-byte joins. All
diagnostic logs are empty. Source snapshots are evidence, not build-origin
attestations. These are repeated bounded controls, not broader shape coverage.

These samples do not prove all f32 inputs, all admitted shapes, other driver
versions or a production failure/cleanup protocol. All receipts retain
`source_runtime_authority:false` and `strict_contract_qualified:false`.
No performance measurement or claim is made.

## Regression and hosted gates

Reissued all 28 existing single/fused NVVM/ROCDL LLVM/MLIR/manifest files and
compared with retained PR93 artifacts under
`mdslc-work/source-pattern-v1/integrated-gpu`: byte-identical. This is artifact
preservation, not a new physical NVVM/ROCDL execution claim.

The complete 308-test Release suite is in progress; final outcomes and exact
hosted head/run identities will be appended before qualification/merge.
CI adds one explicitly checked SPIRV-Tools registration and a separate fixed
two-test issuer/adversarial sanitizer gate without diluting the retained 236.
Missing tools are not interpreted as a passed external validation gate.

Windows diagnostic transport failures discovered in unrelated hosted tests are
isolated in PR #97; they are neither hidden nor counted as SPIR-V failures.
The supported Windows compiler/runtime remains outside this optional Linux-only
issuer. No merge is authorized by this in-progress record alone.
