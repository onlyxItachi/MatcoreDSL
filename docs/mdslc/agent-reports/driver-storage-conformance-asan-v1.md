# Public-driver ASan+UBSan provider-OFF results

This is a separate execution supplement to the historical Release reports
`driver-storage-conformance-v1.md` and `driver-storage-conformance-modes-v1.md`.
It supplies the actual sanitized execution evidence that was explicitly absent
when the runner-mode supplement was written. No compiler/runtime implementation
was changed or built by this independent lane.

## Exact invocation and build identity

On 2026-09-08, the integration owner supplied a frozen whole-component sanitizer
build of implementation `849082e6c9e8f7e40ffe90e75babc2705ee8acc1`. The runner
and fixtures were at test commit `d4fb1cf219e47bc7ed0589b6f46bf47646063657`.
The independent lane executed:

```sh
env DEBUGINFOD_URLS= \
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  python3 compiler/tests/driver_storage_conformance/run.py \
  /home/hamza-usta/MatcoreDSL-wt-private-candidate-dso-v1/build-isolated-asan/bin/mdslc-region \
  --expected-sha256 69688d9c687aaad3c5e1305f258423e544620722fe32599c7b574409efec82a1 \
  --sanitized
```

Read-only inspection of this build's `CMakeCache.txt` confirmed:

```text
CMAKE_BUILD_TYPE:STRING=Debug
CMAKE_CXX_FLAGS:STRING=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
MDSLC_ENABLE_OPENBLAS:BOOL=OFF
MDSLC_REQUIRE_OPENBLAS:BOOL=OFF
```

Thus provider-OFF is a verified build setting, not an inference from omitting
`--has-openblas`. The runner truthfully records that the OpenBLAS policy lane
itself was not requested or tested in this run.

| Tested artifact | SHA256 |
| --- | --- |
| `bin/mdslc-region` | `69688d9c687aaad3c5e1305f258423e544620722fe32599c7b574409efec82a1` |
| `lib/libmatcore_closed_candidates_isolated_v1.so` | `6b3402ebf504460f9a363ac491761238b5d02ee44e05715d9bff68c502d6d179` |
| `lib/libmatcore_runtime.so` | `3971ee0de8164f7198a9d4740afed4b0ff88a98c331a567877f4f829a01508c0` |

All three initial hashes were independently matched to the supplied freeze and
checked again by the runner before every subprocess and at final completion.
The driver reported its coherent Clang/MLIR 21.1.8 ASan+UBSan profile. The freeze
was released only after the complete matrix passed.

## Observed results

The complete generated record is
`/tmp/mdslc-storage-conformance-dv3lkawi/evidence.json`.

| Requested policy / source numerics | Lhs assertions | Rhs assertions | Outcome |
| --- | --- | --- | --- |
| native-strict / strict_f32 | 4,885 | 4,903 | pass |
| generated-strict / strict_f32 | 4,885 | 4,903 | pass |
| automatic / strict_f32 | 4,885 | 4,903 | pass |
| existing-native / strict_f32 | 4,533 | 4,533 | expected checked rejection passes |
| existing-native / reassociate_f32 | 4,885 | 4,903 | pass |

Each cell covers the same 22 source/storage/shape/failure-prefix cases as the
Release suite. The independently emitted generated-strict lhs object linked
with ordinary `/usr/bin/clang++-21`, explicitly including
`-fsanitize=address,undefined`, and its executable passed another 4,885 assertions.

Total: 53,103 assertion evaluations over 242 case executions, zero conformance
failures. All 12 mechanical shared-dependency/object-header checks passed.
Both unsupported sources still rejected with their original locations and no
object. The record has 38 subprocesses; only the two intended source rejections
returned nonzero. All successful subprocesses had empty stderr; no ASan, UBSan,
or leak diagnostic was observed under the stated halt-on-error environment.

After execution, read-only `nm --undefined-only` inspection of
`/tmp/mdslc-storage-conformance-dv3lkawi/lhs-generated-strict.o` found actual
instrumentation references including `__asan_init`, `__asan_report_load4`,
`__asan_report_store4`, `__ubsan_handle_pointer_overflow`, and
`__ubsan_handle_type_mismatch_v1`. This is artifact-level instrumentation
evidence in addition to the banner and cache, not a claim that every potential
fault or operation was exercised. No new deliberate sanitizer-fault positive
control was added in this lane.

Public Result still does not expose actual candidate identity. This evidence
establishes the requested-policy numerical/storage/prefix matrix on the exact
sanitized provider-OFF build; it does not independently authenticate automatic
selection, test unavailable OpenBLAS selection, or establish exhaustive memory
safety. No performance, GPU, cross-platform, installed-package, or private-DSO
ownership claim follows.
