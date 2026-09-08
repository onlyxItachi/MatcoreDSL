# Generated GEMM execution/performance controls

Research harness, not an installed API, production selector, new source authority,
or BLAS-parity benchmark. It compiles ordinary public `.mdsl` source with a supplied
real driver and separately links supplied generated primitive objects with Clang.
No compiler/runtime build is started and no existing build tree is modified.

## Layers and numerical checks

- `region-...`: dynamic read A, read B, GEMM, publish C, complete. Timing includes
  per-invocation snapshots, owned result allocation/initialization, numerical and
  completion checks, publication and Result/session cleanup. There is no observe
  operation. Host A/B/C vector allocation, filling and the oracle are outside timing.
- `primitive-...`: three preallocated identity-layout rank-2 memrefs passed to the
  supplied symbol. Includes primitive output initialization if the primitive emits
  it, but no adapter, snapshots, publication, allocation or FP-state management.
  This direct experimental link does not inherit the driver's execution authority.
- Native-strict is the actual forced public-region policy, not a copied loop given
  that name. Existing-native and OpenBLAS receive explicitly reassociated source.
  Public Result does not expose actual candidate identity; requested policy is
  recorded without inventing identity telemetry. Empty cases may bypass GEMM.

24 strict cases cover square 1..512, rectangles, skinny/tall/tails, dynamic shapes,
zero M/N/K/all-zero with null empty buffers, and scalar plus 7x13 tail forms of
cancellation, separate-product rounding, gradual underflow and signed zero.
The 16 ordinary cases use small dyadics: every output is independently checked
both against a scalar double oracle and a separate volatile f32 increasing-K
multiply/add oracle, with exact agreement required. The eight adversarial cases
require strict f32 bits but are excluded from reassociated/provider lanes.
All output bits, red zones and immutable inputs are checked. Every lane must
detect an intentionally corrupted output. Public-region lanes also check exact
early shape/read and late access/publication failure frontiers. Separate strict
existing-native/OpenBLAS binaries must reject at GEMM before publication.

## Reproduction

Run `python3 compiler/experiments/hpc_execution_v1/run.py --help` for arguments.
Supply an exact driver SHA and its real build-source commit, baseline primitive
object and a new empty output directory under ignored `benchmark_reports/`.
`--provider /configured/exact/libopenblas.so` opts into provider execution; omission
is recorded as not requested/not tested, never as a successful OFF build.
`--extra-primitive LABEL OBJECT ENTRY` repeats for research objects using the private
three-memref C wrapper ABI. `--extra-driver LABEL DRIVER SHA256 SOURCE_ROOT` captures
the extra source HEAD and uncommitted compiler diff and compares generated/native
source routes. This supports a source-equivalent historical baseline without
relabeling its binaries as a fresh current-head build.

The runner freezes driver, compiler, candidate/runtime/provider and primitive
hashes, records fixture hashes and complete commands, and checks ELF64 x86-64
object/executable kinds, primitive entry definitions and direct region dependencies.
It fails on any changed frozen input. `evidence.json` includes failures, not just
the successful summary. Runtime dynamic loading remains trusted.

## Optional timing protocol

Timing requires `--timing --quiet-window-note TEXT`; coordinate other agents first.
`--cpu N` pins executions to a permitted CPU, but does not reserve its SMT sibling
or isolate desktop/system work. No governor, boost, thermal or affinity system
configuration is changed. CPU topology/policy, process checks and load averages
are recorded. Active compiler/build processes cause timing refusal/interruption.

Default: six representative timed shapes, deterministic randomized lane order
over three rounds, two warmup calls, calibrated batches targeting at least 30ms,
three measured batches per round. Thus nine batch-mean samples per lane/shape;
median/min/max are reported. Inputs/outputs are reused and warm; this is not a
cold-cache, multithread scaling, NUMA, training, end-to-end application or packing
amortization study. Very small calls retain timer-loop/status-check overhead.
The optional corpus/runs can be changed explicitly and are recorded in argv.

Compare like numerical permissions and like layers. Region minus primitive time
is not a causal measurement of copying alone: allocation, initialization, FP
scopes, linkage, alignment and dispatch also differ. A fast standalone research
object is not an integrated product optimization. A few shapes cannot establish
provider parity, a universal schedule, a planner cost model or a performance floor.
