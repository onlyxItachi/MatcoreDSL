# Public-driver conformance mode supplement

This supplements, rather than rewrites, the historical
`driver-storage-conformance-v1.md` evidence. No compiler/runtime implementation
was changed or built by this lane.

## Runner changes

- `--sanitized` requires the driver banner to identify its ASan+UBSan profile
  and adds `-fsanitize=address,undefined` to the separately performed ordinary
  final link. It cannot make an uninstrumented compiler installation sanitized.
- Every executable now mechanically requires `NEEDED` records for
  `libmatcore_closed_candidates_isolated_v1.so` and `libmatcore_runtime.so.0`.
  The separately emitted object must report ELF64, x86-64, and `REL` metadata.
  Readelf runs in the C locale; these validation checks are recorded separately
  from successful subprocess execution.
- The runner makes temporary `lhs.mdsl` and `rhs.mdsl` copies with the same host
  header bytes. Only their two per-GEMM profiles, on lines 11 and 15, change from
  `strict_f32` to `reassociate_f32`. Their 22-case source/storage/failure oracles
  and original filename/line checks remain unchanged.
- Reassociated sources execute under requested `existing-native` policy in
  every run. `--has-openblas` additionally runs both under requested `openblas`
  policy. The strict-f32 `existing-native` rejection lane remains separate.
  Without the flag, the report says `not_requested_not_tested`: absence of a
  request is not evidence that the installation was built provider-OFF.
- The driver and both shared-library hashes are checked before every command
  and again at final completion. Failures in structural assertions now leave
  an explicit failed evidence record rather than just captured readelf output.

## Executed Release provider-ON matrix

On 2026-09-08, the integration worktree was independently verified clean at
`849082e6c9e8f7e40ffe90e75babc2705ee8acc1`. The parent supplied a fresh build
freeze after its rebuild. This lane executed:

```sh
python3 compiler/tests/driver_storage_conformance/run.py \
  /home/hamza-usta/MatcoreDSL-wt-private-candidate-dso-v1/build-isolated-release/bin/mdslc-region \
  --expected-sha256 57a34cc895c4e42cc3ccf9b0da0de6269d017d25f6bd118fc5999267852d7568 \
  --has-openblas
```

The candidate DSO hash remained
`00c4f784c614d0a6f55b3baed1ce08382ff5a5622c9e62cf041d98bc6aaaccbd`,
and the runtime DSO hash remained
`df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014`.
The generated evidence is
`/tmp/mdslc-storage-conformance-5wkzhutb/evidence.json`.

| Requested policy / source numerics | Lhs assertions | Rhs assertions | Outcome |
| --- | --- | --- | --- |
| native-strict / strict_f32 | 4,885 | 4,903 | pass |
| generated-strict / strict_f32 | 4,885 | 4,903 | pass |
| automatic / strict_f32 | 4,885 | 4,903 | pass |
| existing-native / strict_f32 | 4,533 | 4,533 | expected checked rejection passes |
| existing-native / reassociate_f32 | 4,885 | 4,903 | pass |
| openblas / reassociate_f32 | 4,885 | 4,903 | pass |

The separately emitted generated-strict lhs object linked with ordinary
`clang++-21` and passed another 4,885 assertions. Overall: 62,891 assertion
evaluations, 286 case executions, zero conformance failures. All 14 mechanical
ELF dependency/header validations passed, and both unsupported source forms
still rejected with original locations and no object. The record contains 44
subprocesses, with exactly the two intentional source-rejection failures.

Five standalone runner unit tests pass without compiler execution or a build:
explicit sanitizer link-flag selection, required dependency records, rejection
of names outside NEEDED records, wrong ELF class/type/machine negative controls,
and the exact source-copy/header/operation-line invariants. They also caught
and corrected an initial multiline ELF-header parser error before the real
Release matrix ran.

The final factoring of the ordinary-link command into a unit-tested helper
preserves the exact Release command recorded above; it does not add an
unrecorded sanitizer execution claim.

## Explicit limits

This supplement records an actual provider-enabled Release run with the
OpenBLAS policy requested, not a provider-OFF build/run. Provider-OFF behavior
must be established with an actual provider-OFF installation; merely omitting
`--has-openblas` is not that proof. ASan+UBSan command construction is unit-tested,
but this supplement does not yet record an actual sanitized driver matrix.

Public Result still does not identify the selected implementation, so the
numerical and prefix oracles do not independently authenticate implementation
identity. No benchmark, throughput, broad floating-point edge-case, GPU,
cross-platform, installed-package, or private-DSO ownership claim is made.
