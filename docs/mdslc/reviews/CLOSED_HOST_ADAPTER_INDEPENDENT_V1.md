# Independent bounded-host adapter review

**Verdict: ACCEPT the reviewed runtime adapter within its bounded Linux x86-64
synchronous normal-return contract.** This is not a claim about generated source
connection, complete integration, packaging, or hosted CI.

Base canonical SHA: `fd64850a0c8cb7d2c0801a64dd94d515dd714130`.
Integration implementation commit: `b33a1dd7621b9d60c2940c783340b9a8c8c93a7c`.
Reviewed source SHA-256:

- `closed_host_v1.h`: `d3e2dc92a86038995f2a82b5d9ddbab1e66d132d99dbb363836275a7d6a8e424`
- `closed_host_v1.cpp`: `5119906fe266d853d37589007308762fe0f4cc31510684d12ce16a0d263ee883`

## Independent evidence

Before integration, the independently authored
[adversarial test](../../../compiler/tests/closed_host/closed_host_independent_test.cpp)
passed **216 checks**, and the
[production allocation test](../../../compiler/tests/closed_host/closed_host_allocation_test.cpp)
passed **131 checks**, both with zero failures under Clang 21 ASan/UBSan.
The latter replaces global `new` with a narrowly armed failure counter and
exercises actual production allocation failure, not the test injection branch.

Both compiled with `-std=c++20 -Wall -Wextra -Wpedantic -Werror -O1 -g
-frounding-math -ftrapping-math -ffp-contract=off -fsanitize=address,undefined
-fno-omit-frame-pointer`. Only the adversarial test and its adapter variant used
`MDSLC_CLOSED_HOST_TESTING=1`. Runtime settings were empty `DEBUGINFOD_URLS`,
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

Three authority controls rejected private value forgery, production candidate
injection, and a test-macro-forged client linked against the production object.
Production `nm` output contained no test-control exports. The durable
[authority script](../../../compiler/tests/closed_host/check_closed_host_authority.cmake)
also requires a successful production link before crediting a failed link.

Tests cover partially overlapping views, old values, late reads, owning
observations, repeated destinations, footprint sentinels, rectangular rhs-carried
GEMMs, ordered second-shape failure, injected and actual OOM, candidate prefix
mutation/exception/reentry/FP tampering, invalid extents/access/numerics/frontiers,
completion, zero-K initialization, exact caller FP restoration, and gradual
underflow despite caller FTZ/DAZ. The allocation sweep verifies no allocation
inside publication and no later allocations after sticky failure.

## Findings corrected before acceptance

- Production warnings-as-errors initially rejected four unused test-only fields;
  scoped `maybe_unused` annotations fixed this without weakening warnings.
- Test-hook reconfiguration initially worked after execution started; it now
  rejects non-pristine sessions and retains sticky reentry failure.
- Planned observation only indicated readiness; it now records an immutable
  snapshot. Returned observation handles own their lifetime independently of
  vector growth.
- Thread confinement and trusted allocator boundaries are explicit. Synchronous
  reentry detection is not a lock or an arbitrary callback sandbox.

## Limits of acceptance

Valid live host float storage, honest capacity/access and no conflicting external
access remain caller preconditions; range checks cannot authenticate pointers.
Strong publication excludes concurrent multiword atomicity, fatal failures and
device/file/network transfers. Irrecoverable complete-FP restoration failure
terminates instead of claiming recovery. Arbitrarily interposed allocator hooks
that modify host/FP state are outside this bounded adapter contract.

Only the new strict scalar candidate is production-callable; test callbacks grant
no production authority. Source authentication and generated-region connection
remain separate gates. Snapshot realization does not require every mathematical
value to materialize in future implementations. Source-generator acceptance must
retain lazy guards, unsigned 64-bit shape control, a pristine session, exact
resource bindings and sealed real-host source pairing.
