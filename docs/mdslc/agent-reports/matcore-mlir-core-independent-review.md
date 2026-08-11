# Matcore MLIR core independent review

- Reviewer lane: independent adversarial Milestone B review
- Reviewed implementation: `e0dee79be7060a2618ff7ed08c88ed80122a66dd`
- Reviewed fixes: `339ff7bc8884d77c09482e738c99c21a85ce96cb` and
  `a66ade8eaa940ca3f4e32266f34da78488504bc5`
- Review date: 2026-08-11
- Scope: ADR-0009, `compiler/lib/mlir/`, `matcore-mlir`, its CMake boundary,
  MLIR goldens/tests, and the v1 capture-to-semantic bridge

## Verdict

Accepted for Milestone B. The reviewed head has no unresolved high- or
medium-severity finding in the declared Matcore MLIR core scope.

This verdict covers the internal `mdsl.gemm` semantic operation, exact Matcore
IR v1 bridge envelope, deterministic inspection tool, and the fail-closed
representation boundaries tested here. It does not claim map/domain semantics,
ordinary-C++ recognition, CPU lowering, or machine-code emission; those remain
Milestones C, D, and E.

The final provenance-sensitive whole-suite run remains an integration-owner
gate after every report/status/documentation commit is present. This review did
not use a stale pre-documentation full-suite result as final acceptance
evidence.

## Adversarial findings and disposition

### Resolved high: incomplete numerical and FP-environment contract

The original operation omitted infinity behavior even though ADR-0009 forbids
`no-infs` assumptions. The bridge context, emitted numerical dictionary,
dialect verifier, golden, and negative tests now carry and require
`infinity = "ieee_no_no_infs_assumption"`. The numerical dictionary also
requires the declared accumulation, reassociation, contraction, reduction,
NaN, signed-zero, approximation, rounding, exception, subnormal, and in-place
fields as a closed contract rather than optimizer defaults.

### Resolved high: semantic sites were removable by standard SymbolDCE

The original bridge emitted unreferenced private `func.func` symbols. A normal
symbol-DCE pass could erase every captured semantic operation despite the
destination write nested inside each function. Semantic site functions are now
public internal liveness roots, and a real MLIR 21 `SymbolDCE` regression test
proves that each `mdsl.gemm` survives. The comments correctly state that this is
not a promised native ABI.

### Resolved medium: explicit-declaration coupling blocked a common semantic op

`mdsl.gemm` no longer requires every origin to forge the canonical
`matcore::mdsl::gemm` declaration. The verifier has exact, versioned variants:

- explicit calls require the canonical callee, explicit provenance,
  `explicit-gemm-f32-v1`, `target=cpu`, and `fallback=error`;
- source-proven recovered loops omit a canonical callee, require authenticated
  recovered provenance, carry the relaxed source-derived numerical profile,
  and remain guard-required;
- recognized but rewrite-rejected loops omit a canonical callee, use strict
  increasing-K C++ semantics, preserve ordinary C++ fallback, and are
  explicitly analysis-only.

The verifier rejects crossed permission/profile combinations. The strict
recovered form uses forbidden reassociation, within-statement contraction,
increasing-K reduction order, strict NaN handling, and signed-zero preservation;
it therefore no longer inherits the relaxed explicit-eDSL contract. Textual
parse/core verification accepts this form, while the executable explicit-v1
bridge envelope rejects both recovered forms.

### Resolved medium: provenance and module identity could disagree

The exact v1 bridge verifier now requires each operation provenance file to
equal module `mdsl.source_file`. Explicit and recovered provenance dictionaries
have separate exact schemas and origin-correlated kinds. Recovered provenance
requires versioned source/proof ranges, a compilation identity, and a canonical
SHA-256 source-snapshot identity; explicit provenance retains exact call and
argument ranges.

### Resolved medium: malformed integer attributes could wrap or assert

The dialect and bridge-envelope verifiers check the exact integer width before
calling `getInt()`. File-line/column values are bounded before conversion to
MLIR's unsigned location fields. I128 mutation tests now reject malformed
alignment and module-version metadata without a Debug assertion.

### Accepted destination, effect, and precondition model

The operation is destination-style, ties its sole result to the explicit output
operand, reports reads on lhs/rhs and an observable write on output through
`MemoryEffectOpInterface`, and is not trivially dead. The textual effect record
is verified against that exact contract. Output/lhs and output/rhs no-alias
relations and alignment are explicitly labeled `required_precondition`, not
proven optimizer facts. The bridge retains operation-local dynamic M/K/N
relationships without unifying identical symbol spellings across source sites.

### Accepted IR and package boundary

Matcore IR v1 remains the sole deterministic JSON capture DTO. The new
representation is the intended MLIR optimizer layer, not a second undocumented
JSON schema. MLIR support is opt-in, requires exact LLVM/Clang/MLIR 21.1.8 CMake
packages, and leaves the default non-MLIR build available. The internal static
semantic library is not installed/exported; only the leaf inspection tool is
installed, so CMake consumers acquire no public MLIR headers or targets.

## Independent validation

The final focused review used an isolated build directory and low parallelism:

```text
cmake --build /home/hamza-usta/.tmp/mdslc-mlir-review.Uhqg2L -- -j2
ctest --test-dir /home/hamza-usta/.tmp/mdslc-mlir-review.Uhqg2L \
  -R '^mlir\.' --output-on-failure -j1
/home/hamza-usta/.tmp/mdslc-mlir-review.Uhqg2L/bin/matcore_mlir_semantics_tests
```

Results:

- final focused rebuild: passed;
- `mlir.semantic.core` and `mlir.semantic.cli`: 2/2 passed;
- semantic core adversarial checks: 204/204 passed;
- deterministic explicit-v1 print/golden/parse/print contract: passed;
- strict and relaxed recovered-form core verification: passed;
- explicit/recovered origin and permission cross-mutation rejection: passed;
- standard MLIR 21 SymbolDCE preservation: passed;
- malformed width/location/source-consistency rejection: passed.

The final installed leaf-tool smoke used a fresh prefix:

```text
cmake --install /home/hamza-usta/.tmp/mdslc-mlir-review.Uhqg2L \
  --prefix /home/hamza-usta/.tmp/mdslc-mlir-final-install.kSLUwD
<prefix>/bin/matcore-mlir \
  --input compiler/tests/ir/gemm_capture.v1.golden.json \
  --numerical-profile explicit-gemm-f32-v1 \
  --execution-intent generic \
  --output <prefix>/out.mlir
cmp <prefix>/out.mlir compiler/tests/mlir/gemm_capture.semantic.golden.mlir
```

The installed output matched the reviewed golden. The installed tool and CMake
package contained neither the source checkout path nor the user-owned MLIR
development prefix. Its dynamic dependency surface named system
`libLLVM.so.21.1` and standard C++/C runtime libraries; it had no installed
RPATH/RUNPATH.

`git diff --check` and commit-level whitespace checks passed at the accepted
implementation head.

## Residual low-severity and later-gate items

1. Only `generic` execution intent is executable in the v1 bridge today.
   `inference` and `training` are enumerated but deliberately reject rather
   than borrowing generic semantics. This is the correct fail-closed Milestone
   B behavior; later intent support requires its own verified contract.
2. Recovered source-snapshot and compilation identities are structurally
   authenticated in this IR. Binding them to actual source bytes and effective
   Clang FP options is the trusted producer's Milestone D responsibility; the
   dialect cannot rediscover a source file from a digest alone.
3. `recognized_rewrite_rejected` is analysis-only and the current executable
   bridge excludes it. Every future lowering must continue to reject that state
   rather than treating a core-verifiable operation as execution permission.
4. Public semantic function visibility is an internal SymbolDCE-rooting
   mechanism. Before Milestone E emits machine artifacts, the compiler must
   deliberately choose collision-safe symbols and emitted visibility rather
   than leaking `__matcore_semantic_<site>` as a stable ABI.
5. This milestone intentionally supports the current rank-two row-major host
   F32 GEMM surface only. Wider dtype/layout/memory-space support must add exact
   dialect contracts; it must not relax this verifier opportunistically.

## Acceptance statement

Milestone B's semantic-core implementation is reviewable and accepted. The
initial rejection findings were reproduced, fixed with focused regressions, and
independently rechecked. No high or medium finding remains open in the reviewed
scope.
