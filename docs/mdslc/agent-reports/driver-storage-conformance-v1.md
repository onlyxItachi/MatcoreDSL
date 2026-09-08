# Independent public-driver mathematical/storage conformance

## Outcome

On 2026-09-08, the frozen Linux x86-64 Release `mdslc-region` compiled and
executed ordinary public experimental region programs for both noncommuting
rectangular carry directions. All 22 cases per direction passed under requested
`native-strict`, `generated-strict`, and `automatic` policies. The additional
`existing-native` strict-policy rejection tests passed. A separately emitted
generated-policy ELF relocatable object was inspected, linked with ordinary
`clang++-21`, and executed successfully. Two unsupported source forms rejected
with original locations and no output object. No correctness counterexample was
found within this bounded matrix.

There were 43,315 assertion evaluations with zero conformance failures across
198 case executions, including repeated policies and the separate object-link
execution. These are checks, not distinct operations or performance samples.

## Scope and artifact identity

The independent fixture worktree is
`/home/hamza-usta/MatcoreDSL-wt-driver-storage-conformance-v1`, branch
`test/experimental-storage-conformance-v1`, based on
`103dfd2168785a09ca8c039e21e74c672c52f251`.
The integration worktree contained uncommitted implementation changes; this
report binds the actual tested artifacts by hash, not by pretending its base
HEAD completely describes those changes. No root worktree files were edited,
no compiler build was started, and no branch was merged.

Tested driver:

```text
/home/hamza-usta/MatcoreDSL-wt-private-candidate-dso-v1/build-isolated-release/bin/mdslc-region
SHA256 5159adc3f960c90c551ba6e790068c64e107f8f8b2f59eb294bf5d606547b8e6
```

The shared artifacts in the adjacent `lib` directory were:

| Artifact | SHA256 |
| --- | --- |
| `libmatcore_closed_candidates_isolated_v1.so` | `00c4f784c614d0a6f55b3baed1ce08382ff5a5622c9e62cf041d98bc6aaaccbd` |
| `libmatcore_runtime.so` | `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014` |

The runner checked these hashes before each subprocess and confirmed the driver
hash again on completion. The freeze was then released to the integration owner.
Later changes to the integration build are outside this evidence.

The complete command/return-code/stdout/stderr record, fixture hashes, library
hashes, and final status are in the local generated artifact
`/tmp/mdslc-storage-conformance-4nz09v8g/evidence.json`. Temporary artifacts are
not committed. Durable reproducer sources and runner are under
`compiler/tests/driver_storage_conformance/`; the runner has no CMake integration
and does not build the compiler.

## Numerical and alias oracles

The first multiplication is `A(2x4) * B(4x3) = C(2x3)`. The lhs-carry second
operation is `C(2x3) * D(3x2) = E(2x2)`; the rhs-carry second operation is
`D(2x2) * C(2x3) = E(2x3)`. Commuting these operands changes or invalidates the
shape, so square commuting examples cannot conceal an operand-order bug.

Every arena element starts as the small integer `((i*7+3)%13)-6`. Independent
host code snapshots the first reads, multiplies with a double-precision scalar
oracle, publishes the expected first result, then snapshots D from that updated
arena. It computes the second product and applies only the publications that
should retire. Bitwise float comparisons check all 192 arena elements, including
unrelated sentinel storage. These values and all intermediate integer sums are
exactly representable in f32; positive-zero reductions are also checked.

The disjoint first inputs, in row-major order, are:

```text
A = [-3, 4, -2, 5; -1, 6, 0, -6]
B = [-4, 3, -3; 4, -2, 5; -1, 6, 0; -6, 1, -5]
C = [0, -24, 4; 64, -21, 63]
```

Representative exact second-result oracles:

| D storage relationship to C | Lhs-carry E, row-major | Rhs-carry E, row-major |
| --- | --- | --- |
| Disjoint | `[104,-36,-531,211]` | `[64,123,39,128,78,106]` |
| Same pointer, differently shaped descriptor | `[-180,-1284,-1407,1089]` | `[-1536,504,-1512,4096,-1440,4048]` |
| D begins two floats into C | `[496,-1492,571,3088]` | `[4096,-1440,4048,4032,-819,3885]` |
| A/B/C/D/E all have the same pointer | `[-112,898,-1288,2836]` | `[394,-108,404,4001,-1338,3970]` |

In the all-same-pointer case the first C is
`[4,6,8;63,-22,62]`, because the first input snapshots share the initial arena.
The disjoint products are also hard-coded as a cross-check on the host oracle.
The exact/partial late-alias cases explicitly check that a wrongly hoisted
pre-publication D read would produce a different E.

All Storage descriptors initially use `Access::read_write`; none is treated as
a no-alias proof. Seven layouts cover disjoint storage, exact/partial late-read
aliasing, C overwriting A, C overwriting B, overlapping C/E outputs, and all five
resources sharing a pointer. Three additional alias-plus-failure cases ensure
that a later destination may legally change through the first publication while
the failing second publication itself performs no writes.

## Shape, failure-prefix, and ownership oracles

The 22 cases include independently zero m, n, k, and p, plus all-zero dimensions.
Empty buffers use null pointers and zero capacity. Lhs `n=0` and rhs `m=0`
exercise a nonempty second result with zero reduction dimension; `k=0` exercises
zero reduction in the first product. Successful empty publications and
observations still retire their ordered effects.

The issued operation order is read A=1, read B=2, first GEMM=3, publish C=4,
observe C=5, late read D=6, second GEMM=7, publish E=8, observe E=9, complete=10.
These numbers are checked through Result diagnostics, never provided to source
intrinsics as execution authority.

| Case | Error | Failed / completed / effect frontier | Publications / observations | Source line |
| --- | --- | --- | --- | --- |
| Success | `ok` | `0 / 10 / 9` | `2 / 2` | none |
| First read dimension mismatch | `shape_mismatch` | `1 / 0 / 0` | `0 / 0` | 9 |
| First publication read-only | `access_denied` | `4 / 3 / 0` | `0 / 0` | 12 |
| Late D descriptor dimension mismatch | `shape_mismatch` | `6 / 5 / 5` | `1 / 1` | 14 |
| Late D insufficient capacity | `insufficient_capacity` | `6 / 5 / 5` | `1 / 1` | 14 |
| Valid D descriptor, dynamic contraction mismatch | `shape_mismatch` | `7 / 6 / 5` | `1 / 1` | 15 |
| Second publication dimension mismatch | `shape_mismatch` | `8 / 7 / 5` | `1 / 1` | 16 |
| Second publication read-only | `access_denied` | `8 / 7 / 5` | `1 / 1` | 16 |

Every failure checks the original `lhs.mdsl` or `rhs.mdsl` filename and nonzero
column. All successful and retained-prefix observations are used after Result
destruction and after replacing the entire external arena with `9999.0f`.
Observation copies/moves survive releasing the original handle. Moving a Result
preserves status and leaves its source empty; out-of-range observations are
invalid.

## Selection, object link, and unsupported forms

Each of `native-strict`, `generated-strict`, and `automatic` produced 4,885
passing assertion evaluations for lhs and 4,903 for rhs. `existing-native`
produced 4,533 passing rejection/prefix assertions per direction: strict-f32
GEMMs fail with `candidate_incompatible`, frontier `3 / 2 / 0`, zero
publications/observations, source line 11. This includes empty math; an earlier
invalid first read still fails at frontier 1. The initial exploratory harness
incorrectly expected this policy to execute strict math; its expectation was
corrected and retained as an explicit negative test, not a compiler bug.

`readelf -d` on every linked executable recorded dependencies on the isolated
candidate DSO and runtime DSO, with the frozen build library directory in its
RUNPATH. `readelf -h` identified the separately emitted object as ELF64 x86-64
REL. The runner passed that object and the two shared libraries to ordinary
`/usr/bin/clang++-21`, and the resulting executable passed all 22 lhs cases
(4,885 assertions). No compiler-internal execution API was used by the fixture.

Two useful unsupported forms fail closed with no object:

- `nested_read.mdsl:7:23`: inline `read(A,m,k)` as a GEMM argument rejects;
  operation arguments must first be immutable named bindings.
- `host_effect.mdsl:11:3`: `++host_counter` inside the region rejects as an
  out-of-vocabulary `UnaryOperator`.

The driver exposes requested candidate policy, not chosen-implementation
telemetry through public Result, so automatic's internal implementation is not
claimed from these tests alone. The evidence is limited to this real Linux CPU
execution, these exact artifacts, and small integer-valued strict numerical
cases. It is not a benchmark, sanitizer run, comprehensive floating-point
edge-case proof, installed consumer validation, cross-platform proof, private
DSO ownership audit, or generated GPU claim.
