# Closed source execution: independent review v1

## Verdict and scope

**ACCEPT, proven within a bounded contract.** Authenticated physical `.mdsl`
source can produce ordinary compiled C++ orchestration of the synchronous
`closed_host_v1` adapter. The emitted program executes the adapter's strict
native scalar mathematics. This review does **not** establish an MLIR-generated
mathematical kernel, installed frontend, arbitrary candidate registration,
accelerator support, performance improvement, or hosted-CI success.

Reviewer authored adversarial tests, not the emitter or runtime implementation.
Review base: `b33a1dd7621b9d60c2940c783340b9a8c8c93a7c`, descended from canonical
`fd64850a0c8cb7d2c0801a64dd94d515dd714130`. Source files were uncommitted during
review; the following SHA-256 identities bind this verdict precisely.

| File | SHA-256 |
| --- | --- |
| `compiler/lib/codegen/ClosedHostEmitter.cpp` | `47cc150212c864b7deea55eeb6aea0f6ed296c338b0744aa6006ff8e90964837` |
| `compiler/lib/codegen/ClosedHostEmitter.h` | `362854ad81c3db349e7fe40bdb9752855fe86d954988448708bb0824256574c1` |
| `compiler/lib/frontend/ClosedRegionAdmission.cpp` | `773e1843b308ad2569e17242976b08a2d0230dbba2b5a79addfc0426b400cee9` |
| `compiler/lib/frontend/ClosedRegionAdmission.h` | `a4953a3e96ca4ddd46a0702bb1200049301008159cf9f2cc27a9a3a8e491781a` |
| `compiler/lib/runtime/closed_host_v1.cpp` | `5119906fe266d853d37589007308762fe0f4cc31510684d12ce16a0d263ee883` |
| `compiler/tests/closed_source/closed_source_execution_test.cpp` | `ffa90dcc90c684be8a4b47566c1d0fa28f4a31cb16cc9e7c1d554dcf795bea2c` |
| `compiler/tests/closed_source/CMakeLists.txt` | `42640aab981b4ee8a2f926575888b9810ebe319fb624f730e589fa09fd818817` |

## Mechanically defended boundary

The emitter requires a real-host immutable admission seal, rebuilds the semantic
witness and pairs it with replayed frozen source/context before emission.
Editable MLIR and hermetic inspection admission do not confer this authority.
Emitted C++ contains statically enumerated adapter calls and shape branches, not
an AST/IR interpreter. By-value resource/shape arguments retain original
parameter order. Validation stays at each executed semantic frontier; neither a
late failure nor an invalid untaken branch can suppress an earlier publication.

Independent real-executable fixtures cover:

- rectangular, noncommuting lhs- and rhs-carried two-GEMM computations;
- immutable old values, late reads, physical aliases, partially overlapping
  output reuse, immutable earlier observations and later visible writes;
- second-read shape failure after publication/observation; unused GEMM results
  whose failing shape checks must still occur;
- unsigned high-bit shape comparisons, untaken invalid resources, frontier gaps,
  zero-K initialization, zero-footprint null views and Session reuse rejection;
- admitted template helpers, value-shape queries and explicit numerical profiles;
- strict separate multiply/add discrimination against FMA and restoration of
  caller rounding mode and exception flags;
- two selected regions from one TU, historical source seals, changed source or
  compile-context identity, environment poisoning, witness tampering, absent or
  moved-from authority, and forbidden host-call/throw/volatile admission.

## Counterexamples that changed the implementation

1. **Same-TU symbol collision:** source and host-context hashes alone do not
   identify a selected region. Two admitted entry points collided. The emitter
   now hashes the complete paired semantic module; both entries co-link and
   execute distinct publications in the regression.
2. **Host identity format:** real admission returns `sha256:<digest>`, not a bare
   digest. The initial emitter rejected every real seal. Prefix validation and
   normalization now permit the positive physical-source path.
3. **Moved-from seal:** default move leaves an empty shared payload; the initial
   host-context query dereferenced it. Queries now report absent authority,
   pairing/emission reject, and payload accessors diagnose invalid use. The
   moved-to seal remains usable.

These were unmerged prototype defects, not claims that canonical behavior had
already provided source execution.

## Exact validation

Observed on Linux x86-64, Ubuntu Clang **21.1.8 (6ubuntu1)** and exact MLIR
**21.1.8**, OpenBLAS disabled for this test lane:

- Release `frontend.closed_source_execution_v1`: **219 compiler/admission checks
  and 37 child-executable assertions, zero failures**; CTest 1/1, 6.25 seconds.
- Debug with `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`: the
  same **219 + 37**, zero failures; CTest 1/1, 8.08 seconds. Leak detection,
  halt-on-error, strict string checks, initialization-order checking and UBSan
  stack traces were enabled. Sanitizer flags were explicitly passed into the
  separate child compilation, including production runtime source.
- Affected Release regression group: **5/5 passed**, 20.60 seconds:
  `mlir.closed_region_semantics` (71 checks),
  `frontend.closed_region_admission` (506),
  `frontend.closed_region_ordinary_rejected`,
  `frontend.closed_region_host_context` (179), and
  `frontend.closed_source_execution_v1` (219 + 37).
- `git diff --check`: passed.

Each source test performs real-host admission, deterministic repeated emission,
ordinary object compilation, final linking with production `closed_host_v1.cpp`,
and actual executable launch. Child compilation uses `-ffp-contract=off`,
`-frounding-math`, and `-ftrapping-math`; no test candidate hook is enabled.

Reproduction: build `matcore_closed_source_execution_tests` in the corresponding
native-frontend/MLIR configuration, then run
`ctest --test-dir <build> --output-on-failure -V -R '^frontend.closed_source_execution_v1$' -j1`.

## Remaining limitations

This is a private source-to-executable specimen, not whole-host-TU replacement or
a packaged user language. The compiler, emitted source, compiler/linker and
runtime remain trusted; editable generated C++ is not an importable authority
certificate. Runtime host-object/lifetime/capacity, thread-confinement and trusted
allocator/FP preconditions remain necessary. Publication is a bounded
synchronous-host normal-return guarantee, not whole-region rollback or universal
device/export atomicity. Copies are this implementation's conservative
realization, not a semantic materialization requirement. A different candidate
registry or mathematical kernel requires its own guarded, isolated-output and
numerical validation before this verdict can extend to it.
