# Two-GEMM RHS mirror: native admission

Base: `8a825350ae8e3f24e55bc8d050375a0b2c0da3dd` (canonical PR #31 merge).
Branch: `agent/two-gemm-rhs-frontend-v1`.
Scope: native region admission and its existing focused test only. No DTO,
public API, runtime, execution capture, or source-rewrite changes.

## Change and preserved boundary

The first output's resolved descriptor identity may match either input of the
second GEMM. The predicate is inclusive, not exclusive: the already-admitted
`C=A*B; E=C*C` case remains admitted. Operand order and both source bindings stay
unchanged in sealed native evidence; the structured handoff can derive its
existing lhs-first edge, or the RHS-only mirror, from those same identities.
There is no additional caller-supplied role or authority field.

Descriptor equality is not physical pointer/noalias proof. A copied descriptor
or same-spelled declaration in another namespace cannot manufacture a carried
dependence. Existing direct-call authentication, nonoverlapping pair selection,
adjacency and host/control barriers, distinct output bindings, and proven
output/input descriptor self-alias rejection are unchanged.

This frontend commit alone does not establish an RHS structured handoff. It
must compose with the independently reviewed role-aware MLIR derivation and
paired verification. Neither change authorizes execution of a derived region
or discharges the per-call guard ledger.

## Falsifiers

The existing test now covers direct RHS consumption without swapping operands,
immutable RHS evidence despite mutable-result tampering, transparent reference
chains, qualified declarations, and a direct try-body pair. The `C*C` control
asserts both input bindings still identify the first output.

Negative cases reject independent calls, a copied descriptor, same-spelled
declarations in different namespaces, observer/mutation/empty-statement
barriers, repeated output bindings, and output/input aliasing through resolved
references. RHS call-derived, cast-derived and static references remain outside
the transparent automatic-reference proof. Existing lhs admission,
authentication, dependency-snapshot, control/scope, and ordinary capture tests
remain in the same test target.

## Validation

From this isolated worktree, using coherent Clang/LLVM 21.1.8 and with the
integration owner's serialized build slot:

```sh
cmake -S compiler -B build-rhs-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_MATCORE_MLIR=OFF \
  -DMDSLC_ENABLE_OPENBLAS=OFF \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang
cmake --build build-rhs-native \
  --target matcore_two_gemm_region_frontend_tests -- -j2
ctest --test-dir build-rhs-native --output-on-failure -V \
  -R '^frontend\.native\.two_gemm_region$' -j1
git diff --check
```

Outcome: configure and the 13-step focused build succeeded; **44 extractions
passed**, **1/1 CTest passed**, and whitespace validation passed. The test's
explicit checks remain active under Release/NDEBUG. No MLIR integration,
generated-region execution, full regression suite, other platform, or
performance claim follows from this focused native result.
