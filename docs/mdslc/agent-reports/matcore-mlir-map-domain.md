# Matcore MLIR map/domain lane report

Date: 2026-08-11

Implementation commit: `7a31f22`

Scope: Milestone C semantic composition only

## Ownership and result

This lane changed only the assigned Matcore dialect implementation, its narrow
CMake test registration, and focused MLIR tests/goldens. It did not change the
frontend, runtime, public headers, planner, bridge schema, roadmap, or accepted
Milestone B contract document.

The dialect now contains:

- functional, pure `mdsl.map` over one ranked tensor SSA value;
- scalar, pure `mdsl.sin` with a closed numerical contract;
- mandatory `mdsl.yield` termination;
- exact version-1 domain dictionaries for `all`, `slice`, `indices`, and
  predicate `mask`;
- canonical `GEMM -> SIN(all)` and static partial-slice golden modules.

## Semantic invariants

- Map v1 accepts only identical rank-two F32 input/result tensor types. This
  narrow bound prevents the rank-two row-major stride rule from being
  incorrectly generalized.
- `all` is the unique whole-tensor form. A full static slice is rejected.
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
- Tensor shape/stride/layout/memory/alignment fields propagate exactly from a
  producing GEMM or map. Dynamic symbols may not be guessed or silently
  renamed.
- The isolated region has one F32 argument, a nonempty linear `mdsl.sin` SSA
  chain, and one matching `mdsl.yield`. Captures, side-effecting/non-Matcore
  operations, dead branches, and malformed block/terminator structures reject.
- Map and sine carry exact FileLineCol-backed, versioned source provenance.
  Provenance anchors remain tied to the producing semantic site.
- Sine v1 requires `correctly_rounded_f32`, non-approximate math, nearest-even
  rounding, gradual subnormals, preserved signed zero, quiet NaN for infinity,
  and an explicit quiet-NaN/payload-unspecified contract. These are semantic
  obligations, not a claim that a lowering exists.
- The existing destination-aware GEMM remains effectful and non-trivially-dead
  when its result feeds a pure map.

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
Matcore MLIR map/domain: 179 checks, 0 failures
mlir.semantic.core: passed
mlir.semantic.map-domain: passed
2/2 focused CTest tests passed
git diff --check: passed for the lane diff
```

The tests cover byte-deterministic parse/print goldens, generic SSA use-def,
all four domain kinds, a dynamic mask guard obligation, an independent partial
slice evaluator proving byte-preservation of all 52 inactive elements, exact
tensor-contract propagation, effects, source provenance, numerical fields,
rank/type bounds, malformed domains, block count/arguments, missing/wrong/
multiple terminators, region capture, unsupported nested operations, and the
observable GEMM destination write.

## Deliberate limitations

- No public `map` or `sin` C++ API was added.
- No map/sine CPU lowering or fusion was added.
- No execution claim is made for correctly-rounded sine.
- Slice and indices are static-only in v1; mask is the current dynamic-domain
  form and still requires a dominating runtime equality guard before lowering.
- The Matcore IR v1 JSON capture/bridge remains GEMM-only and unchanged.
- Whole-repository, package, sanitizer, and CPU execution gates belong to the
  integration owner after concurrent frontend/runtime lanes are committed.
