# Matcore MLIR map/domain independent review

Date: 2026-08-11

## Verdict

Accepted for the internal Milestone C composition-v1 boundary. No unresolved
high- or medium-severity finding remains.

This verdict covers the original map/domain implementation in `7a31f22`, the
closed composition-envelope and provenance fixes in `d2698fb`, the final
canonical-index/source-authentication fixes in `3f29b1c`, and the evidence
record in `c3f0c7e`. It is independent from the ordinary-C++ GEMM recognizer
review and does not authorize map/sine CPU lowering.

## Canonical domain review

The four version-1 domain forms are closed and mutually constrained:

- `all` is the only whole-tensor spelling;
- static slices are zero-based, half-open, nonempty, in bounds, and reject the
  full unit-step tensor spelling;
- static indices are rank-exact, signless i64, nonnegative, in bounds, unique,
  and strictly ascending in lexicographic order;
- predicate masks are rank-compatible unencoded i1 tensors whose dynamic shape
  equality remains a required precondition rather than an optimizer fact.

The final index fix validates coordinate structure before comparing decoded
coordinates. Reversed and interior-permuted lists reject rather than being
silently normalized, while the accepted ordered representation remains
byte-stable through MLIR parse/print. A unique in-bounds set whose cardinality
equals the complete static tensor rejects in favor of `domain(all)`.

## Source-authentication review

The operation verifier validates only the closed syntax of source-backed
provenance. The context-free production composition verifier now rejects every
`source_authenticated` map or nested sine, so self-asserted MLIR attributes are
not permission.

The trusted overload requires a caller-owned `AuthenticatedSourceSnapshotV1`.
The review verified that it:

- exactly matches `mdsl.source_file` identity;
- requires the declared byte length to equal the supplied byte view;
- accepts only canonical lowercase `sha256:<64 hex>` and independently
  recomputes that digest over the supplied bytes;
- bounds every nonempty half-open range by the authenticated byte length;
- derives one-based byte-oriented line/column from each range begin;
- requires each operation's `FileLineColLoc` and provenance to match that
  derived source position; and
- independently checks every source-authenticated map and every nested
  source-authenticated `mdsl.sin`.

The real C++ fixture is 573 bytes with SHA-256
`6ec02671ca476d510ecef9fd862bc15044aa1a185207d21e54f9e467e412def4`.
Independent byte scanning found the complete GEMM call at `[412,461)` on line
16, column 3 and the sine expression at `[543,569)` on line 18, column 34.
These positions agree with the source-backed test module.

Adversarial tests reject a missing trusted context, wrong source identity,
well-formed but incorrect digest, mismatched byte length, out-of-bounds range,
jointly forged provenance/location line, and a forged digest on the nested
sine. The valid authenticated module remains deterministic after parse/print.

The trusted context authenticates provenance against bytes; it deliberately
does not parse C++ or rediscover the mathematical operation. The trusted
frontend/producer remains responsible for semantic recognition before creating
source-authenticated operations. No production caller of this overload exists
yet, so that producer boundary remains an integration gate rather than an
implicit permission granted to arbitrary MLIR text.

## Composition and effect review

- The composition envelope is a strict allowlist: public defined single-block
  semantic roots, exactly one destination-aware GEMM, at least one functional
  map, and one returned live semantic result.
- Calls, helper declarations/functions, unknown top-level/body operations,
  unknown nested effects, synthetic provenance, unencoded tensor violations,
  site/source drift, and dead map results reject.
- Generic SSA use-def carries GEMM-to-map and map-to-map dependencies without
  inventing a total program order.
- Map is functional, preserves its tensor contract, reads input (and predicate
  when present), and carries no writes/read-write effects.
- The GEMM destination write remains observable even when its SSA result feeds
  a pure map.
- The sine numerical profile remains a verified semantic obligation, not a
  claim that a correctly-rounded lowering is implemented.

## Reproduced validation

Toolchain: Clang/LLVM/MLIR 21.1.8. The reviewed build uses the isolated,
coherent MLIR 21.1.8 package. The target build was limited to Ninja `-j2` and
CTest was serialized.

```text
cmake --build /home/hamza-usta/.cache/mdslc-semantic-mlir-build \
  --target matcore_mlir_semantics_tests matcore_mlir_map_domain_tests -- -j2
ninja: no work to do

/home/hamza-usta/.cache/mdslc-semantic-mlir-build/bin/matcore_mlir_map_domain_tests
Matcore MLIR map/domain: 319 checks, 0 failures

ctest --test-dir /home/hamza-usta/.cache/mdslc-semantic-mlir-build \
  -R '^mlir\.semantic\.(core|map-domain)$' --output-on-failure -j1
2/2 passed

clang++-21 -x c++ -std=c++20 -fsyntax-only -I compiler/include \
  compiler/tests/mlir/map_sin_source_fixture.mdsl
passed
```

`git diff --check` passed for the implementation, focused tests, composition
contract, and evidence diff through `c3f0c7e`.

## Review boundary

This acceptance is limited to the internal Matcore MLIR composition model,
its verifier, and focused goldens/tests. Dynamic slice/indices, general scalar
operation catalogs, source recognition, source-backed producer attestation,
map/sine lowering, package exposure, whole-repository tests, sanitizers, and
Windows validation remain outside this lane and must pass in their owning
integration milestones.
