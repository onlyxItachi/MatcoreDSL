# GPU pair issuer-only stacked integration candidate

Status: clean isolated review candidate, not pushed or merged. No build, device
execution, source/runtime qualification or hosted CI was performed for this
restacking task. The integration owner retains publication and merge authority.

Dependency: CPU PR80 exact head
`9a4dfeff70db57f3522e66fe9e13e8602c5a167c`, not an assumption about a future merge.
Worktree: `/home/hamza-usta/mdslc-work/region-optimization-v1/gpu-pair-issuer-candidate`.
Branch: `mdslc/strict-gpu-pair-issuer-candidate-v1`.
Implementation checkpoint: `ada14a44fc4abb451c3127cd71bc2bacd98a0e3d`;
this report is a subsequent documentation-only handoff.

Only the five authorized issuer/hook commits were cherry-picked, in order and
without conflicts:

| Original commit | Candidate commit |
| --- | --- |
| `b0fe264` (shared hooks, root equivalent `3e168a1`) | `0fcd7ab913b5c4855dbfe85e910f17af8ff6bd2f` |
| `60622b06739c98dd4640596eff54982bfb98817a` | `ebfbedcb6b16fd5dce33188dea76a5f7871e7ef5` |
| `836e122` | `37e76a409e0bf8a882255856859d1897d84a8a0a` |
| `7dfcc556c935e844bb0207d97f1f7f55f7d359d9` | `4518d764eb4b73d702e4a08afb56c2d9df51c8f2` |
| `21aad61e21fbc3e86687fd35e3290a77a7d0c401` | `ada14a44fc4abb451c3127cd71bc2bacd98a0e3d` |

## Read-only coherence and authority audit

The candidate's entire `compiler/` Git tree is
`dcdfdca1e7e6ed705ceaaa7b8d97caf4ba8734c1`, exactly equal to the independently
accepted isolated issuer `21aad61` compiler tree. This is a byte-identity check,
not a new build result. CPU dependency `9a4dfef` has compiler tree
`d53a28d99ec12d5204c3263accc287ca492bc40c`; its changes since original issuer base
`6e25df7` are only the hosted native workflow's timeout setting.

The complete dependency-to-candidate diff is additive: 15 issuer/test/report
files plus the two narrow shared CMake hooks. No runtime, public include,
frontend, codegen, source driver, CPU issuer or original single-GEMM GPU issuer
file changes. Those protected paths were compared directly to the exact CPU
dependency; the diff was empty.

Existing optional GPU target/toolchain gates surround the new issuer. With all
GPU options OFF the existing early return remains; issuer-only mode does not
require SDKs or generate physical images. Enabled targets reuse the existing
pinned vendor/toolchain checks and append a distinct checked embedded pair image
and dependency. These are build artifacts, not dispatch authority. The new pair
compile definitions have no runtime consumer in this branch. Original fill and
single-GEMM images, source registry and their invocation paths remain unchanged.

The private CLI still accepts only output prefix plus the closed target spelling;
the constructor has no external IR, Transform, image or callback ingress. Its
diagnostic predicates/manifest cannot create a source plan. The isolated test
launcher is not linked into production dispatch.

The unchanged `ExperimentalRegionEmitter.cpp` explicitly refuses
`strict-fused-pair` unless the candidate is `generated-strict` on checked Linux
x64. The unchanged `SessionAbiV2::gemmStrictFusedPair` similarly rejects other
candidates before its CPU-only pair leaf. Thus `generated-nvvm`/`generated-rocdl`
do not acquire combined-pair source execution merely because this build embeds
the image. No GPU adapter, combined error/poison/quarantine law, source-binding
extension or root-owned RuntimeTests was imported.

## Evidence and remaining handoff

This assembly ran `git diff --check` against the dependency and the repository
hygiene check: both PASS. All historical failures and physical/math boundaries
remain in the [component record](gpu-fused-pair-issuer-v1.md) and its linked WIP
snapshot. Independent review accepted original `21aad61` only as a bounded
issuer/image component; the identical compiler tree is preserved here without
rerunning or relabeling that evidence.

The root's composed build/source/runtime/package/full qualification is ongoing
elsewhere. It must establish its own exact checkpoint and cannot be inferred
from this issuer-only restack. No performance, automatic dispatch, public API,
residency, arbitrary GPU, rank-N or Issue15/20 closure claim is added. Keep this
candidate unpushed until the root confirms composition and authorizes publishing.
