# Matcore MLIR map/domain lane report

Date: 2026-08-11

Implementation commits: `7a31f22`, `d2698fb`

Scope: Milestone C semantic composition only

## Ownership and result

This lane changed only the assigned Matcore dialect implementation, its narrow
CMake test registration, focused MLIR tests/goldens, and the versioned
composition contract. It did not change the frontend, runtime, public headers,
planner, bridge schema, roadmap, or strict Milestone B bridge verifier. The
Milestone B document received only a bridge-versus-composition wording
correction and a link to the separate composition contract.

The dialect now contains:

- functional, pure `mdsl.map` over one ranked tensor SSA value;
- scalar, pure `mdsl.sin` with a closed numerical contract;
- mandatory `mdsl.yield` termination;
- exact version-1 domain dictionaries for `all`, `slice`, `indices`, and
  predicate `mask`;
- canonical `GEMM -> SIN(all)` and static partial-slice golden modules;
- a separately versioned `matcore-mlir-composition-v1` module envelope.

## Semantic invariants

- Map v1 accepts only identical rank-two F32 input/result tensor types. This
  narrow bound prevents the rank-two row-major stride rule from being
  incorrectly generalized.
- `all` is the unique whole-tensor form. A full static slice and a statically
  complete indices set are rejected.
- Static slice and index coordinates are verifier-bounded and exact. Dynamic
  slice/indices are not yet supported.
- A mask is a rank-compatible i1 tensor. Static dimension conflicts reject;
  dynamic dimension equality remains the explicit
  `shape_equality = "required_precondition"` obligation and is not treated as
  proof.
- Partial domains use `outside_domain = "preserve_input"`; the all domain uses
  `not_applicable`.
- Map produces a new functional value, forbids in-place execution, reports no
  memory writes, and includes the mask read when present.
- GEMM, map input/result, and predicate-mask tensor values reject every
  non-null `RankedTensorType` encoding because v1 claims an explicit unencoded
  host row-major contract.
- Tensor shape/stride/layout/memory/alignment fields propagate exactly from a
  producing GEMM or map. Dynamic symbols may not be guessed or silently
  renamed.
- The isolated region has one F32 argument, a nonempty linear `mdsl.sin` SSA
  chain, and one matching `mdsl.yield`. Captures, side-effecting/non-Matcore
  operations, dead branches, and malformed block/terminator structures reject.
- Map and sine carry exact FileLineCol-backed, versioned provenance with a
  closed authenticity discriminant: source-authenticated, derived from a
  producer, or synthetic test fixture. Derived goldens never forge a source
  expression. Source-authenticated fixtures require an exact source range and
  SHA-256 snapshot identity. Production composition rejects synthetic
  provenance, and provenance anchors remain tied to the producing semantic
  site.
- Sine v1 requires `correctly_rounded_f32`, non-approximate math, nearest-even
  rounding, gradual subnormals, preserved signed zero, quiet NaN for infinity,
  and an explicit quiet-NaN/payload-unspecified contract. These are semantic
  obligations, not a claim that a lowering exists.
- The existing destination-aware GEMM remains effectful and non-trivially-dead
  when its result feeds a pure map.
- The composition envelope admits only public, defined, single-block semantic
  roots containing exactly one GEMM, at least one map, and one return. Root
  argument order, site/location binding, live SSA chains, and final return are
  exact. Calls, helper functions/declarations, unknown top-level/body
  operations, unknown effects, and falsely labelled GEMM-only modules reject.

The exact domain interpretation is documented in
`docs/mdslc/MATCORE_MLIR_COMPOSITION_V1.md`: slice intervals are zero-based and
half-open, index coordinates are zero-based and unique, and a mask `true`
element is active while `false` preserves the input. Dynamic mask equality is
a required future runtime guard, never static proof.

## Focused validation

Audited tuple:

```text
Clang/LLVM/MLIR 21.1.8
MLIR_DIR=/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir
Release, Ninja, -j2
```

Commands:

```text
cmake --build /home/hamza-usta/.cache/mdslc-semantic-mlir-build \
  --target matcore_mlir_semantics_tests matcore_mlir_map_domain_tests -- -j2

ctest --test-dir /home/hamza-usta/.cache/mdslc-semantic-mlir-build \
  --output-on-failure -R '^mlir\.semantic\.(core|map-domain)$' -j1
```

Results:

```text
Matcore MLIR map/domain: 280 checks, 0 failures
mlir.semantic.core: passed
mlir.semantic.map-domain: passed
2/2 focused CTest tests passed
git diff --check: passed for the lane diff
```

The tests cover byte-deterministic parse/print goldens, generic SSA use-def,
all four domain kinds, a dynamic mask guard obligation, an independent partial
slice evaluator proving byte-preservation of all 52 inactive elements, exact
tensor-contract propagation, effects, source/derived/synthetic provenance,
real-fixture source hashing/ranges, numerical fields, rank/type/encoding
bounds, malformed domains, full-domain canonicality, block count/arguments,
missing/wrong/multiple terminators, region capture, unsupported nested and
module operations, calls, helpers, site/source/location drift, dead semantic
results, and the observable GEMM destination write.

## Independent-review resolutions

The first independent review reproduced five material gaps. Commit `d2698fb`
resolved them as follows:

- added the versioned composition module envelope rather than treating generic
  MLIR validity as Matcore semantic permission;
- bound derived and source-authenticated map/sine provenance to module source,
  root site, and exact FileLineCol locations;
- rejected unmodelled ranked-tensor encodings throughout the host row-major
  surface;
- documented exact slice/index/mask semantics and corrected Milestone B's
  GEMM-only statement to refer only to the strict capture bridge;
- rejected a statically complete indices domain in favor of canonical `all`.

The follow-up provenance review found that the original synthetic sine golden
incorrectly claimed a source expression at a file containing only the GEMM.
The canonical goldens now use `derived_from_producer`; a separate valid C++
fixture exercises the source-authenticated shape without forging source text.

## Deliberate limitations

- No public `map` or `sin` C++ API was added.
- No map/sine CPU lowering or fusion was added.
- No execution claim is made for correctly-rounded sine.
- Slice and indices are static-only in v1; mask is the current dynamic-domain
  form and still requires a dominating runtime equality guard before lowering.
- The Matcore IR v1 JSON capture/bridge remains GEMM-only and unchanged.
- The composition-v1 envelope is internal and deliberately narrow; it is not a
  public ABI and does not accept general MLIR functions or calls.
- Whole-repository, package, sanitizer, and CPU execution gates belong to the
  integration owner after concurrent frontend/runtime lanes are committed.
