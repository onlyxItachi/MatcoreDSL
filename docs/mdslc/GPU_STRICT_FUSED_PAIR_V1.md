# Bounded staged GPU strict pair v1

Engineering candidate, not yet a qualified canonical source route. This record
must be updated with composed/physical/package/hosted evidence before merge.

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

Initial isolated GPU research at `40cac19fbe0157ab9ed92536612e090aa4385581` passed
physical independent arithmetic on both available devices. CUDA memory checking
and its malformed-scratch detection are separate from AMD oracle/canary checks.
These results do not alone establish source/runtime/package authority.

The independent real-Session/mocked-device test and API fault harnesses address
the ordered guard/completion law. See [runtime qualification commands and
outcomes](agent-reports/gpu-fused-pair-runtime-v1.md). No physical driver can be
substituted by those mocks in a product build. No generated region-wide fusion,
parallel schedule, automatic target selection, external-provider fusion,
resident/asynchronous values, new dtype/rank or public ABI stability is claimed.
