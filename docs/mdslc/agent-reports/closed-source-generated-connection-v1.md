# Authenticated source / generated CPU connection: independent test checkpoint

## Verdict

**PROVEN WITHIN A BOUNDED CONTRACT:** the same real-host-admitted private source
programs execute through either the strict-native control or the compile-trusted
MLIR-generated strict GEMM candidate. These are actual object compilation, final
linking and executable runs, not successful IR verification alone.

This test-only change adds no frontend, emitter, kernel, registry or runtime
implementation. It composes their already independently reviewed boundaries:

```text
physical private .mdsl source -> real-host Clang admission seal
  -> rebuilt/source-paired semantic graph -> generated C++ orchestration
  -> Session guards + isolated private output + fixed candidate selection
  -> issued MLIR/LLVM GEMM object -> value issuance -> ordered host effects
```

Branch origin: source checkpoint
`85d871fc6bb2bc762de1bb5030eb7a2c46854996`. Required reviewed dependencies are
generated oracle/issuer `ef28808` / `f256954`, candidate registry
`78135023dfdd99bd2dd8c3323c2137f6041d64ac`, owning provider correction
`437bdcc1d15373634f6cacd5439566ea6cd3d689`, and independent candidate oracle
`e85b28c73f07b19098a164de9c0f29ebc2389ffd`.
The isolated test branch cherry-picked those dependencies as `c43272f`,
`0b533a2`, `31cc0cd`, `50a8a0e`, and `c395ca2`; it skipped the duplicate host
adapter commit. Integration should take the focused test commit after its
dependencies, not duplicate the dependency implementations.

## What the tests establish

`frontend.closed_source_execution_v1` retains separate ordinary compilation of
the native adapter. `frontend.closed_source_generated_execution_v1` compiles the
same source-derived orchestration and links the production registry archive plus
the ordinary runtime DSO. The registry archive already owns the generated object;
the test does not link a duplicate copy or introduce a candidate callback.

Every nonempty positive mathematical specimen forces `generated_strict` in the
generated lane and checks actual implementation identity, one actual thread,
invocation and successful value issuance. The corresponding native lane checks
the native identity. Original counterexamples remain: rectangular lhs/rhs carries,
old immutable values versus late aliased reads, overlapping output reuse,
observations, late/dead-result failures, unsigned branches with untaken invalid
resources, zero extents, helpers, provenance tampering, insufficient/moved-from
authority, exact strict non-FMA arithmetic and caller FP restoration. Zero-K
realization is explicitly reported as local initialization, not fake kernel
execution. See the [original source review](../reviews/CLOSED_SOURCE_EXECUTION_INDEPENDENT_V1.md).

A new admitted source publishes and observes an old value before reaching a
strict GEMM under forced OpenBLAS selection. It must fail incompatibly at frontier
6, retain completed frontier 5 and effect frontier 3, preserve the earlier output
and observation, leave the later destination untouched, and perform no provider
probe or GEMM. The generated executable also invokes the ordinary legacy C ABI
with a forced reference candidate and verifies its output and actual variant.

These results demonstrate composition: the source graph retains ordered semantic
checks/effects; the adapter consumes runtime shape/storage obligations before
isolated computation; the fixed primitive supplies the mathematical implementation.
No editable inspection certificate gains generic execution authority.

## Exact validation

Linux x86-64, Clang/LLVM/MLIR 21.1.8. Release links local OpenBLAS 0.3.32;
Debug ASan/UBSan disables OpenBLAS. No performance measurement is claimed.

| Configuration | Native control | Forced generated | CTest |
| --- | --- | --- | --- |
| Release, OpenBLAS ON | 220 compiler checks + 51 child assertions | 221 + 52 | 2/2 PASS, 13.15 s |
| Debug ASan/UBSan, OpenBLAS OFF | 220 + 51 | 222 + 52 | 2/2 PASS, 16.80 s |
| Independent reviewer, Release | same counts, zero failures | same counts, zero failures | 2/2 PASS, 13.02 s |
| Independent reviewer, Debug ASan/UBSan | same counts, zero failures | same counts, zero failures | 2/2 PASS, 16.68 s |

The extra generated check executes the established heap-out-of-bounds negative
control and requires an AddressSanitizer failure inside the generated symbol.
In the sanitized configuration, another check byte-compares the source-linked
`strict-normal.o` with the negative control's `strict-asan.o`. The reviewer also
independently extracted the object from the actual production registry archive:
all three had SHA-256
`7c7a6986a264cde615c804badc296cac1c681c6756334148ee3dc6ee84137892`.
The generated entry wrapper was present once.

ASan flags instrument the actual LLVM kernel through its issuer and the separate
child C++ orchestration compilation; the registry/runtime are instrumented in the
sanitized build. UBSan covers the C++ components, not arbitrary raw LLVM IR merely
because UBSan is linked. Leak detection, halt-on-error, strict string checks and
initialization-order checking were enabled for positive runs.

The first successful generated Release run took 96.36 seconds because its
negative-control symbolizer consulted the inherited debug-info service. Matching
the existing kernel control's local-only environment (`DEBUGINFOD_URLS` empty,
first-error termination) reduced the final generated run to 6.37 seconds. This
was test orchestration, not a mathematical/runtime fix; no failure was waived.

Independent CPU-candidate reviewer: **ACCEPT** at test-source SHA-256
`76e3fe81619ffc23d3c2b1653fc5d27998e1a8d09003325e4bb3facce59deed0`.
Test CMake SHA-256:
`432d0199a9b5022e7bfe63e60995a67750aa070be9ea7ba5fdd51450f7e5a165`.
`git diff --check` passed. No hosted CI or full clean integration suite is claimed
by this isolated test checkpoint.

Build `matcore_closed_source_execution_tests`, then run:

```sh
ctest --test-dir <build> --output-on-failure -V \
  -R '^frontend.closed_source(_generated)?_execution_v1$' -j1
```

## Deliberate limits and integration handoff

This remains a private source-execution specimen, not an installed/public frontend
or whole-host-TU replacement. No source-issued provider mathematics is claimed by
the incompatible-provider fixture; actual provider execution has its separate
registry evidence. No fusion, target scheduling, buffer reuse, accelerator,
zero-copy, performance or BLAS-parity claim follows. Valid host-storage, allocator,
thread confinement, fixed trusted candidate and bounded FP preconditions remain.

The integration owner must add `frontend.closed_source_generated_execution_v1`
to the exact hosted source/sanitizer test scopes and update their expected counts,
run coherent full regressions, and retain the prior dependency reviews. The next
usable-product boundary is a deliberately designed frontend/tooling surface over
this now-connected private contract, not another execution-authority flag on an
inspection representation.
