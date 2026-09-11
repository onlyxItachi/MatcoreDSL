# Connected staged-GPU integration evidence v1

Canonical starting point: `9c2149a25e12f5192c168f3e047097505263fc1a`.
Integration branch: `mdslc/generated-gpu-correctness-v1`.
This is implementation/review/local execution evidence, not a merge or release.
See the [contract](../STAGED_GPU_CANDIDATES_V1.md) and
[campaign decomposition](../MULTITARGET_CORRECTNESS_CAMPAIGN_V1.md).

## Focused history and independent review

- AMD research `c48ba635` / adapter `93882bbe` were integrated as `acc5c74` /
  `6b9240a`; NVIDIA research `f73cab9` / adapter `ff3d4e33` as `cd778e3` /
  `41f567c`. Shared issuer `c5657a6` became `c73516e`.
- `7116bd7` connects the shared verified issuer, real toolchain/image build,
  private DSO, runtime guards, source selector, host symbol ownership and source
  execution tests. Independent NVIDIA-lane review read all 25 changed files and
  supporting paths: no blocking issue; replayed qualification **20/20**, symbol
  ownership **98/98**, and both real GPU-enabled hidden-device refusal tests.
  The optional security-report wrapper failed its metadata finalization; this
  evidence is an engineering review, **not a finalized security audit**.
- `a3b547f811c23374a69f14fd93040494a5d3470e` corrects two internal manual-archive
  test link closures, makes an incompatibility diagnostic truthful for shape/
  capability failures, and adds real installed source, hidden-device refusal and
  machine-image checks. Independent AMD-lane review found no blocking issue,
  specifically checking OFF/issuer-only configuration, exact dependency paths,
  source authority and the distinction between image and DSO identity.

Prior NVIDIA failure-path review found that context-restoration failure could
affect caller state and pending transfer buffers could outlive stack storage.
The surviving adapters use isolated worker threads, heap-owned staging, checked
completion and resource quarantine/poisoning. Fault tests preserve those negative
results; no context-error assumption or premature free was waived.

## Actual validation

Linux x86-64 / AMD Ryzen AI 9 HX 370; RTX 4060 Laptop `sm_89` and Radeon 890M
`gfx1150`. Coherent Clang/LLVM/MLIR 21.1.8, CUDA ptxas 13.3.73, actual NVIDIA
driver DSO 615.71.09, HIP 7.2.70201, OpenBLAS 0.3.32 enabled for coexistence.
Build jobs bounded to two with cross-lane serialized links. Temporary validation
uses an ext4 SSD directory outside the enclosing home Git worktree; no toolchain
source build, system installation or user-work cleanup occurred.

| Scope | Exact result |
| --- | --- |
| Initial real source + physical adapter CTest | **4/4 passed**, 75.49 s |
| Initial issuer/guard/registry + normal and ASan/UBSan driver mocks | **140/140 passed** |
| Focused follow-up at `a3b547f` | **8/8 passed**, 155.86 s: both built/installed source routes, both object contracts, installed archive and generated-source consumers |
| Full clean GPU-enabled standalone suite at `a3b547f` | **330/330 passed**, no skips, 602.38 s |
| Separate SDK-independent issuer-only build | 36 build actions plus 61 driver build actions completed; **2/2** issuer/qualification tests passed |
| GPU-compiled-out source refusal in issuer-only build | **2/2 passed**, 6.21 s; not GPU execution evidence |
| Physical CUDA adapter | **76 cases / 17,761 output comparisons**, zero failures |
| Physical HIP adapter | **76 cases / 17,761 output comparisons**, zero failures |
| Source storage/effect oracle per enabled target | Existing **22 lhs + 22 rhs** alias/late-read/failure cases, unchanged mathematical oracle; built and installed driver runs passed |
| CUDA/HIP driver fault models | **34 modes each**, normal and ASan+UBSan; distinct from real-driver instrumentation |
| Machine artifacts | Two images per target; **9 NVIDIA / 7 AMD** synthetic target/arithmetic validator falsifiers rejected |

The first full attempt had **322/326 passes**: two real manual-static-archive
test dependency omissions, plus two expected cleanliness-gate refusals caused
by the integration owner editing before the suite ended. The gates were not
weakened. Both dependency closures were fixed; clean configured `a3b547f` then
passed the entire 330-test suite in one run.

Physical tests compare exact f32 bits except NaN payload (class only), plus
caller FP/errno/context/device preservation, output guards, input aliasing,
rectangular/empty shapes, FMA and reduction-order witnesses, subnormal input/
output and IEEE special values. An actual target is required: no CPU substitute,
emulation or missing-device skip is counted as execution.

Separate historical adapter evidence remains bounded: real NVIDIA memcheck
reported zero errors/leaks; real HIP normal and host-UBSan execution passed.
Real HIP+ASan failed with an unresolved alternate-signal-stack teardown error;
its failure is retained in the AMD report. Mock-driver ASan/UBSan passes do not
prove the real HIP library is sanitizer-clean. The full suite includes real
devices in a Release process, not an ASan-instrumented HIP runtime.

## Exact local artifacts at the clean code checkpoint

Build root: `/home/hamza-usta/mdslc-work/multitarget-v1/builds/gpu-integration`.
Logs: sibling `evidence/gpu-full-ctest-02.log`, `gpu-followup-focused.log`,
`gpu-issuer-only-{configure,build,driver-build,refusal}.log`. The driver source
tests retain command/exit/output JSON and driver SHA under
`tests/generated_gpu/gpu-*-source-*`; object tests retain image/command JSON.

| Artifact | SHA-256 at `a3b547f` |
| --- | --- |
| `bin/mdslc-region` | `3448e932a1a8c9e9bcf5619ff783c639296f6edb53ca5ba6d848427cf71d3e87` |
| Private candidate DSO | `de1ac14826bef9cf390d23364efefabdb6d5f489f78d03729656cfd3f630bc36` |
| NVVM fill image | `2142a5eb35b130f6775e637409c48f511fbc5fa08c88cf6a647f79e0dfed8b01` |
| NVVM GEMM image | `5c7dc7c3b12a7372d11b1290b83216d31fd61dd074c2da2f5b42f842042c2fe4` |
| ROCDL fill image | `2bdced3fa6c69d8546700c6c58a22799d8f29c0e80912c50c8bb362f5fce5621` |
| ROCDL GEMM image | `2201f598ec1d0db1d14456baee98393a67f1eb0f6752f8ae8d96b3215e21ab8c` |

## Claim boundary and reconciliation

This proves a real source-authenticated, synchronous, staged strict GEMM GPU
candidate under a bounded target/toolchain/shape contract, with observable
failure-prefix preservation. It does not prove whole-region GPU transformation,
fusion, residency, zero-copy, async execution, fastest-choice dispatch, general
accelerator support or performance. Issue #15 remains partial/open and #20
design-only/open (live checked during this integration).

Hosted checks and eventual canonical merge must be recorded at their actual
heads; local GPU hardware evidence must not be relabeled hosted execution.
The separate ARM and strict-CPU-ISA branches require normal dependency merging
and an independently reviewed combined regression before their claims compose.
