# Private synchronous closed-host adapter tests

These tests cover `compiler/lib/runtime/closed_host_v1.{h,cpp}`. This is an
ordered operation API for a future authenticated generated wrapper, not an AST
interpreter or a source-to-generated-executable claim. The production variant
has only its new strict scalar GEMM candidate. Existing CPU/provider execution
and inspection contracts are unchanged.

## Bounded contract

The Linux x64 adapter snapshots reads into immutable owned values, allows
external view overlap, captures actual immutable observation records, and
publishes complete results with a strong normal-return guarantee. Earlier
publications survive later failure. Successful publication does no allocation,
callback, floating-point computation or recoverable validation after its first
write. Its eager snapshots are an initial realization, not required semantics.

Inputs must describe live initialized ordinary float objects with truthful
capacity/access and no concurrent conflicting access. Sessions are
thread-confined; the active-call guard detects synchronous reentry, not races.
The linked allocator/deallocator is trusted to preserve FP state and introduce
no arbitrary host effects, including on allocation failure or result destruction.
Allocator attempt counts and instrumentation are not stable mathematical trace
events. No sandbox, crash recovery, concurrent atomicity, arbitrary export,
device storage or no-copy guarantee is made.

GEMM runs under an isolated nearest-even, masked-exception, gradual-underflow
environment and restores full caller FP control/status before returning. A
post-candidate control-state mutation rejects the candidate. An inability to
restore a captured supported-host environment is an unrecoverable malfunction
and terminates; it cannot honestly return a recovery status. Unsupported
platforms fail closed. The scalar candidate preserves increasing-K separate
f32 multiply/add rounding; it is also a legal implementation of the broader
reassociation-permitted profile. No provider performance or compatibility claim
follows.

## Tests and initial evidence

- `closed_host_v1_test.cpp`: 137 checks covering rectangular noncommuting GEMMs,
  both carry orientations, old values and late reads across partial-overlap
  writes, stable observation records, sticky errors and effect prefixes,
  injected partial writes/throws/reentry/FP corruption, all initial snapshot and
  observation allocation points, capacity/access/overflow rejection, zero-size
  geometry, strict reduction/FMA counterexamples, and caller FP restoration.
- `closed_host_production_test.cpp`: actual non-injected scalar adapter smoke,
  including publication, observation, completion and opaque handle properties.

Both executed in Release and ASan+UBSan builds. They compile independently of
Clang/MLIR libraries. Compile the adapter translation unit with
`-frounding-math -ftrapping-math -ffp-contract=off`; fast-math compilation is
explicitly rejected. Build the injection test against a **separate** adapter
object with `MDSLC_CLOSED_HOST_TESTING=1`; never link that object into production.
Production has no `configureForTesting` or `allocationAttemptsForTesting` export.
The test configuration is accepted only before the first Session operation.

Example from the repository root, placing binaries in an existing build tree:

```sh
clang++-21 -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  -frounding-math -ftrapping-math -ffp-contract=off \
  -DMDSLC_CLOSED_HOST_TESTING=1 -Icompiler/lib/runtime \
  compiler/lib/runtime/closed_host_v1.cpp \
  compiler/tests/closed_host/closed_host_v1_test.cpp \
  -o build/closed-host-tests
build/closed-host-tests
```

For sanitizer validation replace `-O2` with `-O1 -g`, add
`-fsanitize=address,undefined -fno-omit-frame-pointer`, and run with
`DEBUGINFOD_URLS=`, `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

Independent review additionally executed 216 adversarial checks
and 131 real replaceable-global-new allocation-failure checks under ASan/UBSan.
These now live in `closed_host_independent_test.cpp` and
`closed_host_allocation_test.cpp`, with a separate production-authority CTest.
Production injection failed compilation, a macro-forged injection client failed
linking against the production object, and private Value forging failed
compilation. Their exact reviewed identities and limits are in the
[independent review](../../../docs/mdslc/reviews/CLOSED_HOST_ADAPTER_INDEPENDENT_V1.md).

The review found two draft defects before acceptance: production
warnings-as-errors rejected inactive test-hook fields, and test configuration
was accepted after execution began. Scoped field annotations and a pristine-only
configuration guard fixed them. No warnings or sanitizer scopes were disabled.
