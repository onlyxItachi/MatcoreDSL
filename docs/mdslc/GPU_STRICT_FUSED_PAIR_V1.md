# Bounded staged GPU strict pair v1

This specifies the bounded combined realization, not a general GPU-fusion or
performance guarantee. The exact canonical engineering checkpoint is recorded
in [CURRENT_STATE.md](CURRENT_STATE.md). Integration, full-regression receipts
and hosted merge gates are tracked in [PR #86](https://github.com/onlyxItachi/MatcoreDSL/pull/86),
separately from the [isolated issuer PR #85](https://github.com/onlyxItachi/MatcoreDSL/pull/85).

## Meaning and authority

The original authenticated source plan for adjacent strict `C=A*B; E=C*D`
remains authoritative. A/B/D must be dominating immutable reads and C must have
exactly one retained use. The unchanged original Program and paired witness are
replayed; rhs carry, late reads, observable C, live shape uses and arbitrary
transformed/imported IR acquire no new authority. Default execution is unchanged.

The separately issued combined GPU realization does not inherit the CPU leaf's
no-recoverable-error property. It follows the [reviewed combined realization
law](agent-reports/accelerator-fusion-authority-review-v1.md), grounded in resource
contract 8: implementation failures can vary when a physical invocation is
legitimately removed. This is not permission to relabel a failure from an actual
producer invocation that already ran.

- f1 retains A/B validity, profile, shape, original capability discovery, pair
  image availability, FULL logical C extent, original GPU work bounds and FP
  entry/restore. It retires guards only: no launch and no C Value are reported.
- f2 retains D/profile/shape/discovery, pair image availability, FULL logical E
  extent and its original GPU bounds, followed by the separate combined work
  qualification. Only then may private E and the shared realization be prepared.
- Actual combined setup/transfer/launch/completion/cleanup failure belongs to f2.
  Earlier publications and owning observations survive; no partial E escapes.
  Failure remains sticky and the old output handle is retained, including when
  that handle aliases one of the immutable input handles.

## Physical realization

The closed issuer reuses the CPU pair's actual upstream Transform/Linalg and
One-Shot row-panel derivation, not handwritten GPU arithmetic. Standard serial
loop conversion and GPU outlining produce one kernel with one thread and one
owner of `min(4,M)*N` private global panel storage. The five rank-2 descriptors
carry A/B/D/E/panel in the checked 35-field kernel ABI. Each reduction preserves
increasing order, separate f32 multiply/add, positive-zero seeds and the
producer's intermediate f32 store/load rounding boundary. No performance claim.

Targets remain the exact qualified sm_89/NVVM and gfx1150/ROCDL recipes and
toolchains, not arbitrary NVIDIA/AMD devices. Original per-GEMM candidate bounds
remain mandatory. The combined serial recipe additionally requires
`M*N*(K+P) <= 2^18`; this is a private qualification bound, never semantic IR or
a planner/provider crossover. Each dimension is bounded before multiplication.

Private staged A/B/D/E/panel resources are owned by fresh joined CUDA/HIP workers.
Original single-GEMM control flow is retained. Its poison domain and completion
helpers are shared with the combined route: uncertain completion cannot leave a
usable sibling route. Every possibly live device resource and host staging
buffer remains reachable on quarantine. There is no CPU fallback, retry,
caller-context mutation, source residency or zero-copy claim.

Empty E and N=0 may bypass the physical leaf only after all original guards and
the pair-image/target checks. K=0 with N>0 and nonempty E still invokes the pair:
the consumer must evaluate `+0 * D`, including Inf and NaN. A hypothetical first
launch failure is not invented for a launch that the combined realization omits.

## Evidence and limitations

The production issuer at `21aad61e21fbc3e86687fd35e3290a77a7d0c401`
(compiler-identical isolated integration `950e5cabcaf42b0f22b09160fa140a0015a85729`)
has separate [image qualification](agent-reports/gpu-fused-pair-issuer-v1.md):
147 issuer checks, 48 valid-IR refusals, 17 NVVM/16 ROCDL machine-gate negatives,
and actual execution on both qualified devices. Each device performed 285
launches plus 264 explicitly counted host bypasses, checking 148377 strict
outputs, 2586 canaries and 407354 immutable inputs. Bypasses are not GPU launches.
CUDA Compute Sanitizer passed its normal run and detected the deliberately
under-sized panel in its exact negative control; AMD oracle/canary evidence is
not device-sanitizer evidence.

Completed focused connected qualification uses production files preserved
unchanged through `cb2bb730693aedd1d469e73e2fd943fb955b4f3d` and
`9f7f9af80d96c8f38d4f66097a5c8779fb185464`: real-Session/mocked-device and
normal/ASan API-fault tests (151 CTest cases), plus four actual NVVM/ROCDL
source/program tests. The source tests check the original 22-case oracle in both modes,
actual combined-call linkage, host interposition refusals and unavailable-target
failure. The private work-cap fixture distinguishes legal unoptimized execution
from combined f2 refusal while preserving the prior effect/observation prefix,
original source identity, final-output sentinel and caller FP/errno state.
At clean checkpoint `9f7f9af`, the relocated installation ran both GPU targets and both source/program routes
after deleting its temporary producer source/build trees; its unchanged driver
SHA-256 was `0d14d7a716e8ddf191b319a6358e935aad4bc956d033e672b57d801dcba5956f`.
These scopes and retained counterexamples have a separate
[independent review](agent-reports/gpu-fused-pair-independent-review-v1.md).

Real vendor-driver host-ASan remains unqualified: the CUDA attempt failed at
cuInit status 2 before any image load/launch, while real HIP/global host-ASan
remains configure-refused. Neither outcome is a passing sanitizer result.
The passing API mocks and CUDA device instrumentation have different scopes.
An auxiliary broad run with Python `-O` is not assertion-enabled regression
qualification; the final full-regression gate must run with PYTHONOPTIMIZE unset.

The independent real-Session/mocked-device test and API fault harnesses address
the ordered guard/completion law. See [runtime qualification commands and
outcomes](agent-reports/gpu-fused-pair-runtime-v1.md). No physical driver can be
substituted by those mocks in a product build. No generated region-wide fusion,
parallel schedule, automatic target selection, external-provider fusion,
resident/asynchronous values, new dtype/rank or public ABI stability is claimed.
