# Connected strict GPU pair: merged qualification checkpoint

Status: **bounded source/runtime connection normally merged and independently accepted**.
[PR #86](https://github.com/onlyxItachi/MatcoreDSL/pull/86) merged as
`523f8637bb135b7b5e6b4f08f563a2c93a2ebfee` from exact head
`fb70f7d98356607974389ebd2a9a6e4e96fc94dc` into
`13283c440fc4fdad42606bb723dfcedde4f07644`.
Actual merge tree: `f2078702cf8d2ff9287fae62c079821de1f2573e`;
compiler tree: `89d039099120b5ce88fab0b13e8175962ae96f98`;
AGENTS blob: `b72e0a6e519a7c17eeba1ba10f6f982c97d54849`.
This documentation checkpoint records that engineering merge, not a future docs SHA.

## Exact source and authority boundary

The merged route extends existing explicit `--optimization strict-fused-pair` to
forced `generated-nvvm`/`generated-rocdl` on qualified Linux x64 sm_89/gfx1150.
Only adjacent pure strict lhs `C=A*B; E=C*D` with dominating immutable Read inputs
and private single-use C qualify. Original source/header/host identities, Program
and paired witness stay unchanged. Imported/serialized IR, plans, manifests,
image paths and manual linking acquire no source authority. Native/default,
provider, original single-GEMM and independent forwarding routes remain unchanged.

The [exact candidate contract](../GPU_STRICT_FUSED_PAIR_V1.md)
declares a separately owned combined realization, not reuse of CPU's no-recoverable-error argument.
f1 retains the original A/B/profile/contraction/capability/full-C/FP guard bundle
and retires guards only, not a launch or C Value. f2 retains D/second-profile/
full-E/original bounds and checks combined `M*N*(K+P)<=2^18` before preparing the
shared invocation. Actual combined failures belong to f2; this cannot relabel an
actual sequential producer failure. Old output and earlier publication/owning
observation survive failure. Unresolved completion retains possibly live host/
device resources and poisons the original and combined routes' shared domain.

The one-thread/one-block kernel uses adapter-owned, checked `min(4,M)*N` panel storage,
preserves increasing separate f32 reductions and the first result's f32 boundary.
Empty E/N=0 bypasses follow both original guards; K=0 with nonempty E/N>0 executes
the consumer, including `+0*Inf/NaN`. This serial correctness recipe is not a
performance, residency, general fusion or universal hardware result.

## Completed local and component evidence

The [owner's independently inspected local receipt](https://github.com/onlyxItachi/MatcoreDSL/pull/86#issuecomment-5983601325)
records clean tested source `9f7f9af80d96c8f38d4f66097a5c8779fb185464`:
**571/571 unique tests passed, zero failures/skips/disabled, 1760.01s**.
It used `env -u PYTHONOPTIMIZE`, serial CTest, JUnit and no exclusions;
legacy Python assertions remained active. Full compiler tree
`89d039099120b5ce88fab0b13e8175962ae96f98` and AGENTS match final `fb70f7d` and the actual merge.
Workflow identity differs by the reviewed Release budget/comment correction;
commands, flags and scope are unchanged. The full run was at `9f7f9af`, not at `fb70f7d`.

| Distinct scope | Receipt and limitation |
|---|---|
| Real Session/API mocks | 151 focused CTests, including the independent 449-check real-Session test; injected driver/API faults and host ASan/UBSan are not physical GPU execution. C++ allocation/thread failure reasoning is code inspection, not injected proof. |
| Actual connected source | Both GPU targets, single-source and multi-source, none/pair modes; original 22-case arithmetic, distinct call linkage, interposition/unavailability refusals and the work-cap f2/effect-prefix discriminator. |
| Genuine relocated package | Producer source/build deleted before installed execution on both GPU targets and both source/program routes. Both package tests and deletion safety passed within the full571 run. |
| Isolated issuer/images | Normally merged PR85 from `950e5ca`, compiler-identical to physically qualified `21aad61`; component and hosted evidence remain in the [issuer checkpoint](gpu-pair-issuer-merged-checkpoint-v1.md), not re-counted as connected source tests. |
| Physical/instrumentation | Normal NVIDIA/AMD and CUDA device Memcheck positive/expected-negative scopes are separate. Real CUDA host-ASan failed prelaunch at cuInit2; HIP/global host-ASan stays unqualified. AMD has no device-sanitizer claim. |

Full local raw evidence remains outside the repository at
`/home/hamza-usta/mdslc-work/region-optimization-v1/evidence`.
`gpu-pair-9f7-full.log` SHA256:
`01b59521ddab24f26e13b0bc794e3db25f66372c595c4d2550d3fbd7b958c47b`;
`gpu-pair-9f7-full.xml` SHA256:
`ff54a191006468adea6e3b40d6ec4626b40ecb8d68acb1f3541e5e9c4155365d`.
The receipt records the independently parsed 571 unique run nodes, package
identities and unchanged installed-driver digest. No raw log/binary is tracked.

## Exact-head hosted and final acceptance

The [final integration review](https://github.com/onlyxItachi/MatcoreDSL/pull/86#issuecomment-5984197447)
records **all 23 final `fb70f7d` hosted checks SUCCESS**, independent Astra acceptance
and separate Sol prospective-tree acceptance, followed by the normal merge above.
Native [PR run37229724826](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37229724826)
and [push run37229722549](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37229722549)
cover Release provider/MLIR configurations, row-contiguous, Debug, ASan+UBSan and TSan;
the full rollup also includes Windows, both ARM64 schedules, legacy CI and hygiene.
These are 23 checks across events, not 23 unique configurations or physical devices.

Independently inspected native job receipts, all at final `fb70f7d`:

| Job | Registered | Passed | Skipped | Failed |
|---|---:|---:|---:|---:|
| PR Release, OpenBLAS disabled, scalar (111516672322) | 264 | 264 | 0 | 0 |
| PR Release, OpenBLAS disabled, row-contiguous (111516672373) | 265 | 265 | 0 | 0 |
| Push Release, OpenBLAS disabled, row-contiguous (111516665675) | 265 | 251 | 14 | 0 |
| Push Release, OpenBLAS required (111516665693) | 267 | 267 | 0 | 0 |
| PR Release, OpenBLAS required (111516672522) | 267 | 253 | 14 | 0 |
| PR Debug (111516672321) | 262 | 262 | 0 | 0 |
| Push Debug (111516665619) | 262 | 262 | 0 | 0 |
| PR ASan+UBSan (111516672324) | 200 | 186 | 14 | 0 |
| Push ASan+UBSan (111516665639) | 200 | 186 | 14 | 0 |

Each listed 14-skip set is existing packed/generated AVX512 capability gating.
`generated_gpu_fused_pair.frontiers` actually passed, including SDK-free real-
Session/mocked-device host sanitizers, not physical vendor-driver sanitization.
The listed PR disabled-provider Release scalar/row-contiguous jobs also completed
the installed production-without-tests proof; this step is intentionally conditional
OFF in OpenBLAS-required jobs. Earlier job counts are not substituted here.

Saved external rollup `evidence/pr86-exact-fb70-green.json` SHA256:
`b855a4da3b0697d10a940c4b134a656bbbf973a84fb175efcaaeabc2f97eef48`.
It is a premerge exact-head success snapshot; its historical OPEN/draft fields
do not supersede the actual normal merge identity recorded above.

## Retained reviews and losing controls

The [independent design/component report](https://github.com/onlyxItachi/MatcoreDSL/blob/954a8e6b6fea232db53146a64ef824f3b51e6739/docs/mdslc/agent-reports/gpu-fused-pair-independent-review-v1.md)
retains its historical pending snapshot; the later local receipt and final review
complete their respective scopes without rewriting that report. Its separate mocked,
machine-image and actual-source inspections are not interchangeable receipts.
The [PR85 final review](https://github.com/onlyxItachi/MatcoreDSL/pull/85#issuecomment-5983552509)
accepts only the predecessor issuer boundary.

Retain the earlier Python-O auxiliary run, fixture-main SIGILL/explicit-return
correction, SDK-free Threads CMake failure, stale real host-ASan registration
failure and all image-gate/weak-oracle losing controls. None is retroactively
relabeled as assertion-enabled regression or physical arithmetic passage.
The [Release CI budget report](gpu-pair-release-ci-budget-v1.md)
retains two clean `9f7f9af` push jobs canceled by the 45-minute deadline, not manually.
Their zero-failure CTest summaries do not make the incomplete jobs passing.
The independently reviewed 45-to-60-minute change preserves every command,
matrix entry and test scope; it supplies time, not a relaxed acceptance rule.

## Branch reconciliation

The actual merge preserves the tested compiler/AGENTS and the five main-only
checkpoint documents inspected in the final integration-tree review. Boundaries
remain separately qualified; a later connection does not rewrite earlier receipts.

| Boundary | Exact premerge head | Normal engineering merge | Evidence |
|---|---|---|---|
| CPU source PR80 | `9a4dfeff70db57f3522e66fe9e13e8602c5a167c` | `c478e49d56719caaa898ff517a09bc0379d31aa6` | [CPU contract](../STRICT_FUSED_PAIR_SOURCE_V1.md), [source report](strict-fused-pair-source-v1.md), [PR87 checkpoint](https://github.com/onlyxItachi/MatcoreDSL/pull/87); local406 belongs to compiler-identical `6e25df7`, not final `9a4dfeff`. |
| GPU issuer PR85 | `950e5cabcaf42b0f22b09160fa140a0015a85729` | `bb5cc1dd80f79ac2dc09259b6d85a04c14f39a34` | [Issuer checkpoint](gpu-pair-issuer-merged-checkpoint-v1.md); physically qualified component `21aad61` is compiler-identical, not itself the merge. |
| GPU source/runtime PR86 | `fb70f7d98356607974389ebd2a9a6e4e96fc94dc` | `523f8637bb135b7b5e6b4f08f563a2c93a2ebfee` | Local571 at compiler/AGENTS-identical `9f7f9af`, exact-head hosted and final review above. |

This completes only the bounded NVVM/ROCDL source-connection lane of
[issue #83](https://github.com/onlyxItachi/MatcoreDSL/issues/83), not broader accelerator authority.

## Exactly one subsequent boundary

[Issue #89](https://github.com/onlyxItachi/MatcoreDSL/issues/89) scopes a closed
parameterized bounded fully-static positive rank-2 SPIR-V shader issuer, with
the fixed 2x3x4 specimen as control, not family qualification. It grants no source/
runtime authority. [Research PR84](https://github.com/onlyxItachi/MatcoreDSL/pull/84)
remains draft; the [d214 audit](https://github.com/onlyxItachi/MatcoreDSL/blob/d214b017cad93ede7445148988f7aad36bd4573f/docs/mdslc/agent-reports/spirv-source-frontier-audit-v1.md)
maps whole-graph/ABI/strict-FP gates and later Vulkan ICD/lifetime/failure/artifact
obligations. The separate [LLVM-SPIR-V lowering control](https://github.com/onlyxItachi/MatcoreDSL/blob/25f696e7c4102bdf3b385fe760a9175249ab5975/docs/mdslc/agent-reports/llvm-spirv-lowering-control-v1.md)
retains five compile-only Kernel/OpenCL variants, byte-identically reproduced by
the integration owner. It is neither Vulkan nor device arithmetic proof. MLIR
SPIR-V, LLVM's SPIR-V backend and SPIRV-LLVM-Translator are distinct routes, not
interchangeable qualification. No NVVM/ROCDL authority transfers to them.

No performance/parity, broad fusion, automatic selection, residency, zero-copy,
asynchrony, arbitrary target or public API/ABI stability claim follows.
[Issues #15](https://github.com/onlyxItachi/MatcoreDSL/issues/15),
[#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) and
[#77](https://github.com/onlyxItachi/MatcoreDSL/issues/77) remain open.

This checkpoint preparation performs docs/read-only evidence work only; no
source code, compiler, SDK or device workload is changed or rerun.
