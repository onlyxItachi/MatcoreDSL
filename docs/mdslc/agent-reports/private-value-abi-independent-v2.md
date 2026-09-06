# Independent private Value ownership / ABI review

Verdict: **ACCEPT within the existing bounded synchronous host contract**.
Reviewer did not author the opaque private-Value implementation. This review
covers the implementation delta on top of composed checkpoint
`de0911eebc788b2d05266efadb9d3902ebc2b7c7`, not new source-language admission or
the independent installed driver/compiler integration.

Reviewed exact source identities (SHA-256):

```text
compiler/lib/runtime/closed_host_v1.cpp
102d88f0d0430dfca16394b549df8824c2fe54d5beed44900f9c1d51ab6d3cf6
compiler/lib/runtime/closed_host_v1.h
d8f3d0312c4cd71e869f32e350d0aa3baa5bbc139b7bdef948099ad6b7a3db7f
compiler/include/matcore/region.h
2cbff60e4c5685d1388604df9642126e31296e067bd471e6eae9a70a26c2a39b
compiler/include/matcore/detail/region_storage.h
8803a9b31de0e72ffb1045acfede828473b08f9f7292e4fefd27e1dc9b1c0927
```

## Source reasoning

- Private `ValueAbiV2` contains one opaque pointer; all owning operations are
  out of line. Independent immutable handles retain atomically; final release
  uses acquire/release ordering. Nothing makes concurrent mutation of the
  **same handle**, a Session, or external storage legal.
- Nonempty snapshots/results retain the two owned allocation points (storage
  owner, elements). A temporary owning Value retains fresh result storage until
  candidate/FP/status checks succeed. Assigning a result that aliases both input
  handles occurs only after their last mathematical read, preserving separately
  retained old-value identity. Failure does not replace the caller's Value.
- Observation records remain internal and own immutable Values. Public
  observations retain the opaque block only after retirement; Session-level
  observations return independent Values, not vector-element pointers. Vector
  growth and eventual block destruction cannot invalidate an outstanding Value.
- The runtime-only `ObservationAbiV2` and `ValueStorageAbiV2` names avoid retaining
  incompatible template/COMDAT ownership implementations under old type names.
  `SessionAbiV2` and `ValueAbiV2` have new C++ linkage identities, not merely a
  new marker call that an older weak inline constructor could discard.
- `matcore_closed_host_private_value_abi_v2` is a private revision dependency,
  **not** source/candidate authority, a cryptographic artifact identity, or a
  promise to accept arbitrary independently compiled ABI-compatible code.

## Independent mechanical evidence

Local environment: Linux x86-64, Clang 21.1.8. The production native-only adapter
archive was linked directly; no test callback/injection macro was enabled in
the new independent harness.

`compiler/tests/closed_host/private_value_independent_test.cpp` was authored
independently and retained from
`/tmp/mdslc-private-value-independent.Z9qlo4/value_review.cpp`; SHA-256:
`dffc920e7ee082e4d42ee4bc4b62b5adedd3f1325f683b538a37236cccec1af8`.

- **85 assertions and 32,000 ownership cycles, zero failures**, in both a
  strict-warning Release link and ASan+UBSan (`-O1 -g -fsanitize=address,undefined
  -fno-omit-frame-pointer`). Runtime archives are respectively
  `build-value-release/lib/libmatcore_closed_host_v1.a` and
  `build-value-asan/lib/libmatcore_closed_host_v1.a` in the isolated Value worktree.
- Actual global-new failures at ordinals 1 through 12 across
  `publish -> observe -> same-handle GEMM -> publish -> observe -> complete`
  preserve the exact first-failure/completed-effect prefixes, prior writes,
  prior observations and old independent Values. The allocation-failure
  injector itself performs no arbitrary host/FP effects.
- A Value extracted from the first observation survives 33 subsequent
  observation entries, vector growth, Session destruction and block release.
- Eight host threads independently copy/move/read/release retained Values;
  the original outside owner is dropped before joining. No shared mutable
  Session or same-handle race is used as a purported synchronization proof.
- Independently reran six existing focused tests: retirement, mixed public
  result ownership, production allocation failure, private Value ownership,
  mixed-library-configuration private Value, and actual revision-link controls.
  **6/6 Release passed in 0.66 s; 6/6 ASan+UBSan passed in 1.18 s.**
- Revision controls execute a matching real archive, then rename only its
  revision symbol in a separate archive and require the exact unresolved-v2
  diagnostic. The stale constructor adversary is genuinely emitted as a weak
  old `Session` constructor, inserted first/last at `-O0`/`-O2`; every incompatible
  link rejects, and each matching-runtime mixed-object control executes.

Sanitizer execution used `DEBUGINFOD_URLS=`,
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`,
`UBSAN_OPTIONS=halt_on_error=1`. No sanitizer diagnostic occurred.

## Boundaries retained

Valid exclusively accessed host memory and a conforming trusted runtime
allocator/deallocator remain preconditions. Atomic reference counts protect
independent immutable ownership, not arbitrary pointer writes, `const_cast`,
crash rollback, asynchronous publication or device/file effects. Overflow of a
reference count terminates; practical capacity impossibility is not modeled as
a recoverable source frontier. This review does not claim a stable ABI, source
authentication from link success, Windows execution, accelerator support,
copy elimination or performance improvement.

The previous marker-only COMDAT counterexample is retained by the implementation
lane; the reviewed versioned-class design is its correction, not a claim that
the earlier marker-only design was sufficient.
