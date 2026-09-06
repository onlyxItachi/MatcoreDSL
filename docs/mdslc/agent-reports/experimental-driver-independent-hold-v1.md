# Independent driver review: HOLD

Recorded by the integration owner from the independent review lane on
2026-09-07. This is a preserved negative experiment, not an acceptance report.

## Observed counterexamples

The initial Release driver binary SHA-256
`be96321332dd20d8a2dea34008f7b53f97b44f86ba261691564814b221392440`
allowed Clang to select a substitute `ld` through inherited `PATH`. The substitute
returned success after writing ASCII text, and the driver published that text.
An ordinary compilation/execution control succeeded first.

At corrected implementation `0d880002402f3d04de0f96a77749dbcd1419a90a`, the
independent review confirmed that the same PATH adversary instead produced a
real ELF executable with correct arithmetic. The driver now pins the configured
linker and verifies emitted artifact kind. This correction alone is insufficient.

The original host can replace the private archive's weak
`ObservationAbiV2` destructor. A conforming allocation-failure sweep invokes the
forged destructor during observation cleanup. The assembler-label version exits
71; an ordinary private namespace/class definition with no assembler label exits
72. The successful, non-failing computation control does not invoke the forged
destructor. Whole-archive linkage therefore does not prove ownership of weak
implementation symbols. These are cross-TU ownership/ODR forgery adversaries,
not claims that the forged programs obey the runtime's private C++ ABI contract.

The permanent fixtures in `compiler/tests/closed_driver/independent_*` preserve
the PATH control, actual generated arithmetic/owning observation, and both
private-symbol spellings. Their initial harness requires refusal of both
executable and object publication and intentionally fails on the current driver.
They are not yet registered as a passing production test.

## Direction under investigation

Isolate compiler-issued helper definitions before LLVM host linking; isolate
private candidate runtime implementation from host weak symbol selection.
Rejecting assembler-label syntax alone was falsified by the named class variant.
A private DSO alone is also insufficient if weak inline helper definitions remain
replaceable at the earlier LLVM link. Exact export ownership checks must remain
for public/private boundary symbols. Hidden implementation names may coexist in
host code only if they cannot replace Matcore's implementation.

If intrinsic encapsulation wins, change the hostile fixture's acceptance from
mandatory rejection to either reasoned rejection or real isolated execution
with correct failure prefixes, no forged exit, and sanitizer coverage. Do not
change it to mere compilation success.

## External review interruption

Both independent reviewer agents hit the service usage limit before the new
encapsulation design could be implemented and independently accepted. No driver
acceptance or merge was issued. Throwaway reproducers additionally remain under
`/tmp/mdslc-driver-independent.TjYD0d`; durable source fixtures are authoritative
for future reproduction, not that temporary location.
