# GPU pair issuer: canonical qualification checkpoint

Status: **normally merged isolated issuer/image boundary; no combined GPU source authority**.
[PR #85](https://github.com/onlyxItachi/MatcoreDSL/pull/85) merged as
`bb5cc1dd80f79ac2dc09259b6d85a04c14f39a34`, parents `f5345c5` and exact premerge
head `950e5cabcaf42b0f22b09160fa140a0015a85729`.
The merge, premerge head and physically tested isolated issuer
`21aad61e21fbc3e86687fd35e3290a77a7d0c401` share exact `compiler/` Git tree
`dcdfdca1e7e6ed705ceaaa7b8d97caf4ba8734c1`.
This identity preserves the tested compiler; it is not a new canonical physical run.

The [integration-owner final review](https://github.com/onlyxItachi/MatcoreDSL/pull/85#issuecomment-5983552509)
accepts this exact head following independent adversarial component/composition
review. The [component report](gpu-fused-pair-issuer-v1.md),
[restacking audit](gpu-fused-pair-issuer-candidate-v1.md) and
[interrupted snapshot](gpu-fused-pair-issuer-wip-v1.md) retain their historical
results, failures and pending states; this checkpoint does not rewrite them.

## Evidence scopes

| Scope | Recorded outcome |
|---|---|
| Issuer | 147 checks, including 48 rejected valid-IR corruptions; CLI refusal checks passed. No imported IR/image/Transform/callback authority. |
| Actual image drift gates | 17 NVVM and 16 ROCDL captured-inspector corruptions rejected; not execution of forged images. |
| Normal physical, each sm_89/gfx1150 device | 285 launches plus 264 arithmetic host bypasses; 148377 strict outputs, 2586 canaries and 407354 immutable-input checks passed. |
| CUDA device instrumentation | Compute Sanitizer normal: zero errors. Deliberate undercapacity negative: expected exit86/two errors, synchronization719, no output or reclamation of possibly live resources. |
| Real host-ASan | CUDA experiment failed before launch at cuInit status2 and remains unqualified; HIP/global host-ASan remains unqualified. API mocks and device Memcheck do not substitute for this scope. |
| Hosted exact head950 | All 23 checks SUCCESS; hosted software checks are not physical GPU qualification. |

Counts, commands, exact tool/device/image identities, external raw-log hashes,
strict NaN-class/non-NaN-bit oracle, zero cases and failed attempts are in the
component report. The serial one-thread/one-block kernel implements strict lhs
`C=A*B; E=C*D` with caller-owned `min(4,M)*N` panel. Full logical C checks and
documented dimension/work bounds remain caller obligations, not leaf guards.
No HPC schedule, broad hardware, performance,
residency, automatic dispatch or general binary-verifier claim follows.

## Hosted qualification, separate from physical evidence

PR native [run37224783544](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37224783544)
and push [run37224707120](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37224707120)
completed successfully on exact head950.
Required-OpenBLAS Release [job111502084777](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37224783544/job/111502084777):
262 registered, 248 passed, zero failed, 14 preexisting AVX512 capability skips.
`generated_gpu_fused_pair.issuer` and `.cli` passed in both native Release runs.
Debug [job111502084295](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/37224783544/job/111502084295):
257 registered, 243 passed, zero failed, the same 14 existing AVX512 skips.
Debug's configuration did not enable the optional GPU issuer; its successful
regression scope is not an issuer or physical GPU run. Skipped ISA execution is
not qualified by these green jobs.

## Authority retained and exactly one next boundary

This merge changes no source/runtime dispatch: GPU `strict-fused-pair` remains
refused. CPU source PR80 and its unchanged truthful 406-distinct-test/hosted-skip
evidence remain documented in [CURRENT_STATE](../CURRENT_STATE.md) and the
[CPU source contract](../STRICT_FUSED_PAIR_SOURCE_V1.md).

Next is separately qualifying/admitting [draft PR #86](https://github.com/onlyxItachi/MatcoreDSL/pull/86),
the combined GPU source/runtime law. At this checkpoint its head is
`954a8e6b6fea232db53146a64ef824f3b51e6739`, base main; its own exact-head hosted,
complete assertion-enabled local, package and independent qualification remain
pending. The immutable Program/witness, both original guards/full C extent,
private output/workspace, fallible launch/completion and shared poison/quarantine
law require that separate acceptance. CPU guard retirement does not authorize it.
Issues [#15](https://github.com/onlyxItachi/MatcoreDSL/issues/15) and
[#20](https://github.com/onlyxItachi/MatcoreDSL/issues/20) retain their existing boundaries.

This documentation task performed read-only Git/hosted-log verification and
lightweight diff/hygiene checks only: no compiler, device workload or source-code change.
