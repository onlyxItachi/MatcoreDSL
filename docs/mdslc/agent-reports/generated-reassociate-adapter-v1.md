# Forced generated-reassociate adapter: bounded implementation evidence

Base: canonical `c04215e787832b35bbd6a85a51d9285b053181bc`.
Worktree: `/home/hamza-usta/MatcoreDSL-wt-generated-reassociate-adapter-v1`,
branch `mdslc/generated-reassociate-adapter-v1`. This report records the adapter
lane before integration with the independently owned issuer/CMake change.
There is **no new-candidate source-execution or combined full-gate claim yet**.

## Contract and ownership

- Append private candidate/implementation enums without changing previous
  enumerator values or record layout. Add `generated-reassociate` to the emitter
  and CLI policy mapping; default/automatic/generated-strict remain unchanged.
- Each executed GEMM retains value/profile/shape checks, then candidate legality.
  The new arm requires `reassociate_f32`, a compiled linked leaf, and direct
  platform-v2 AVX2/FMA hardware plus OS-enabled XMM/YMM state. Unknown/malformed
  relevant domains fail closed. Discovery preserves errno and never runs for
  an unchosen or numerically refused candidate. No packed selftest supplies proof.
- Output allocation, complete FP scope, empty/K0 handling, invocation, control
  validation, FP restoration and final owning-Value replacement keep their
  existing order. Only positive M/N/K enter the new leaf. Only fresh C may carry
  the issuer's output-data noalias fact; input/input aliasing remains legal.
- No whole-region profile preflight: the first reassociated GEMM may publish and
  be observed before a later strict GEMM fails. Earlier effects/old values remain.
  The stronger normal-return publication guarantee is not weakened to generic
  dialect partial-write semantics.
- Borrower's interface is
  `_mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1`, three existing private
  rank-2 memref descriptors, guarded by `MDSLC_CLOSED_HOST_GENERATED_REASSOCIATE`.
  Borrower owns issuer and `compiler/lib/regions/CMakeLists.txt`; direct discovery
  must be present in both installed archive and isolated DSO, without embedding
  a second provider-policy adapter. This lane does not edit those files.

## Actual focused checks before integration

| Check | Actual result |
| --- | --- |
| Synthetic capability/adapter executable, Clang 21.1.8 O2 | 144 checks, zero failures |
| Same actual adapter compiled O1 ASan+UBSan | 144 checks, zero failures |
| Expanded native-only candidate matrix, O2 | 61,171 checks, zero failures |
| New source fixture on existing generated-strict driver route | 1,804 checks, zero failures |
| Same source fixture on existing-native route | 1,863 checks, zero failures |
| Syntax checks and `git diff --check` | Passed |

The synthetic target compiles `closed_host_v1.cpp` and
`generated_reassociate_guard_test.cpp` with `MDSLC_CLOSED_HOST_TESTING` and
`MDSLC_CLOSED_HOST_GENERATED_REASSOCIATE`. A test-linked discovery deliberately
sets errno to EINVAL; selected calls preserve caller EDOM. Unchosen/strict-refused
calls must not discover or invoke. Sixteen pure predicate negatives include
unknown architecture/version, missing/unknown AVX2/FMA hardware or OS evidence,
unknown/missing XCR0 state, malformed domains and attempted inheritance of
unrelated implementation/selftest proof. Empty M/N and K0 run the same legality
gate before allocation. The fake leaf verifies distinct C while input handles
may coincide. It is explicitly **not** the generated numerical implementation.

Normal artifact: `/tmp/mdslc-reassociate-adapter-guard.uEZQkl/guard`, SHA256
`cb8b240c6c277ee54667b7d5c4e2764c5907ac5d06dc1aa4c454b75676871aac`.
Sanitized artifact: `/tmp/mdslc-reassociate-adapter-guard-asan.os4InT/guard`, SHA256
`e0c467d3c6b8605b94f4e18bc57100cf617740f6ef0630651f7b2d50dd812a98`.
The sanitized run sets `DEBUGINFOD_URLS` empty,
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`
and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.
Native-only artifact: `/tmp/mdslc-reassociate-native-only.QQuuCm/candidates`, SHA256
`c686864fa20c372b7aa348dfd9b190f3e1ba029b820e50a1bb6595297a654414`.
All three harnesses retain strict host compilation flags
`-fno-fast-math -ffp-contract=off -frounding-math`.

## Source fixture and first rejected forms

`compiler/tests/closed_driver/generated_reassociate.mdsl` SHA256:
`bb57a495fb507121f573cfdd97e67c7eba01364ee34a007bd6e6dd39cbe6cfd4`.
The baseline smoke driver was the root-owned existing Release executable at
`/home/hamza-usta/MatcoreDSL-wt-source-visible-math-libraries-v1/build-library-release/bin/mdslc-region`,
observed SHA256 `d755534f5fd47a80d42ac7851209c6734021a1b0fb55f7a82cc499e2d892ad82`.
This was a read-only fixture/compatibility smoke, not a clean combined-head
validation; no H1 build or source was changed by this lane.

The strict route's 1,804 checks exercise the source's permitted reassociate
profile using strict arithmetic, **not FMA**. The existing-native route's 1,863
checks also exercise the new fixture's later strict-profile refusal. That failure
is at frontier 8, completed frontier 6, completed effect frontier 5, original
source line 29 column 15, with one publication and one owning observation.
Dynamic shape failure at the same operation dominates its candidate check.
The fixture includes square/rectangular/tail/zero cases, noncommuting lhs/rhs
carries, alias-sensitive late reads, destination sentinels, untouched mutable
inputs, preserved caller errno/MXCSR/x87 controls+status and observations surviving
Result destruction. A full/tail FMA discriminator awaits the real new candidate.
Legacy/provider sensitive values use permitted K2-family membership, not an
arbitrary reassociated value as a universal oracle.

Initial fixture attempts were correctly rejected for inherited FP pragma
attributes, a missing canonical marker/noexcept declaration, and host inline
assembly. Only the test was corrected: marker/noexcept, host-only FP pragmas,
and Linux/glibc `fenv_t` fields instead of inline assembly. Production admission
was not relaxed. This fixture is intentionally in the existing Linux x86-64
region lane, not evidence for portable `fenv_t` layout or generated Windows.

Independent read-only peer review accepts the bounded adapter/API delta: appended
enums, strict-first refusal, direct errno-preserving hardware/OS discovery,
unchanged default and ownership/FP order. This is not full connected acceptance.

## Required combined acceptance

Root will combine the focused commits before actual issued-leaf source execution,
normal and sanitized candidate matrices, independent storage/failure/FP controls,
full regressions and fresh installed SDK/archive/DSO tests. New source cases are
registered per existing candidate, with OpenBLAS conditional on actual build
availability. Public Result is still status/effect evidence, not implementation
identity. No timing, crossover, automatic selection, family-wide portability,
whole-region transformation or BLAS-parity claim is introduced.
