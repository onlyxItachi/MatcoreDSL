# Milestone G bounded contract independent review

Date: 2026-08-11

Verdict: accepted for the bounded pre-freeze scope. No unresolved high- or
medium-severity finding remains.

## Reviewed change set

The review covered:

- `357fd5af330d4c9d87f7970ac319c7e44043e037` — bounded packed-B,
  returned-string, and additive-evolution contracts;
- `64897a779da827a5026958990aeb0a8fded071cf` — focused ABI, packed-B, and
  borrowed-string tests;
- `717a5ce82f0bdc93b50b14b2060239da3d9ad593` — hosted Clang/LLVM/MLIR 21.1.8
  matrix and link-boundary checks;
- `517ec97cb8fdb7c2a24a3387cfb614df61921100` — semantic-MLIR artifact
  hygiene; and
- `ffa295687abb17d52427241e9c306bd2bf339ac8` — corrected Windows-native
  unavailable-path invocation.

This was an independent source, contract, test, ABI, and workflow review. No
implementation file was edited by the reviewer.

## Adversarial conclusions

### Packed-B v1

The public wording matches the existing runtime boundary:

- provenance binds source and packed addresses, shape, storage extent,
  packed-element count, and packing constants;
- provenance does not authenticate source or packed contents;
- execution requires exact RHS address identity and rejects relocated packed
  storage with stale provenance;
- tensor, descriptor, packed-storage, and workspace overlap is rejected before
  execution;
- the v1 prepacked entry points route only the forced native packed AVX2/FMA
  candidate, whose planner legality requires one requested thread; and
- no execution-context entry point accepts the v1 packed descriptor.

The serial/cross-context limits and manual invalidation rule are therefore
honest. Same-address byte mutation remains deliberately undetectable; explicit
repacking is the only supported refresh operation. The contract does not claim
content identity, immutable ownership, concurrency, or a global packed-weight
cache.

### Returned C strings

Every exported pointer population path reviewed resolves to one of:

- a runtime string literal;
- a planner `std::string_view` backed by a static registry or string literal;
- a platform-validation rejection literal; or
- the function-local static OpenBLAS provider record, whose configuration and
  core-name pointers remain provider-owned.

No exported report/status path returns a temporary `std::string`, stack buffer,
or per-call plan storage. The pointers are NUL-terminated borrowed values and
the documented dynamic-library unload boundary is accurate. Human-readable
messages and reasons remain diagnostic text; status codes, enums, versioned
fields, and documented stable variant IDs are the machine-readable contract.
The tests compare repeated diagnostic text only to exercise deterministic
storage/readability, not to declare sentence wording an ABI.

### Evolution and ABI

The policy explicitly says that it is not a public API, ABI, backend-contract,
operation-set, or support-window freeze. It requires additive `_vN` symbols or
records for growth, preserves existing layouts and numeric values, keeps fixed
candidate arrays bounded, and defers general transformed-operand ownership,
structured diagnostic iteration, execution intent, and support-duration
decisions.

The reviewed public-header changes are comments only. No record member,
parameter, calling convention, enum value, or exported declaration changed.
The ABI test continues to pin LP64 sizes and offsets and now also pins the
relevant version/count constants and pointer field types.

### Hosted matrix, link boundary, and artifact hygiene

The Linux hosted matrix covers:

1. OpenBLAS required with Matcore MLIR enabled;
2. OpenBLAS explicitly disabled with Matcore MLIR enabled; and
3. OpenBLAS explicitly disabled with Matcore MLIR disabled.

MLIR-enabled jobs install and authenticate the matching 21.1.8 tools and CMake
package, configure with an explicit `MLIR_DIR`, build at parallelism two, run
the full registered CTest set, and reject an aggregate shared `libMLIR`
dependency in both `matcore-extract` and `matcore-mlir`. The MLIR-disabled
compatibility job exercises the explicit unavailable path. Repository hygiene
runs independently on every push and pull request and now rejects a force-added
`*.semantic.mlir` artifact while allowing intentional goldens with the
`.semantic.golden.mlir` suffix.

One medium finding was reproduced during review: the first platform-neutral
unavailable test registration passed GNU `.o`/driver spellings to the Windows
clang-cl parser, so Windows would reject the output form before reaching the
intended MLIR-unavailable diagnostic. Commit `ffa2956` closes it by selecting
validated clang-cl spellings and a `.lib` output on Windows while retaining the
GNU invocation on non-Windows hosts. Static rereview against
`ParseClangClCpuInvocation` confirmed `/TP`, `/std:c++20`, `/EHsc`, `/MD`,
`/c`, separated `-o`, and `.lib` are accepted.

## Independent focused validation

A fresh out-of-tree Release build at the final reviewed head used Clang 21.1.8,
the coherent native Clang frontend, OpenBLAS 0.3.32 from pkg-config, Matcore
MLIR disabled, low process priority, and build parallelism two:

```sh
nice -n 10 cmake -S compiler -B "$MDSLC_G_REVIEW_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_MATCORE_MLIR=OFF \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang

nice -n 10 cmake --build "$MDSLC_G_REVIEW_BUILD" --target \
  matcore_runtime_cpu_test \
  matcore_workspace_runtime_test \
  matcore_runtime_abi_compat_test -- -j2

nice -n 10 ctest --test-dir "$MDSLC_G_REVIEW_BUILD" \
  --output-on-failure -j1 -R \
  '^(runtime\.cpu\.workspace_v1|runtime\.c_abi\.compatibility_v1|runtime\.cpu\.gemm_v0)$'
```

Result: all three targets built; focused CTest passed 3/3 with zero failures.

The following non-build checks also passed:

```sh
bash tests/check_repository_hygiene.sh
bash -n tests/check_repository_hygiene.sh
git diff --check
git ls-files '*.semantic.mlir'
```

The Windows harness parsed successfully as Python source and was checked
against the actual Windows driver grammar. The implementing lane separately
reported the registered MLIR-disabled unavailable CTest passing 1/1 after the
fix.

## Evidence boundary

This review does not claim a fresh Windows runner execution, a hosted GitHub
Actions result, a dynamic-library unload experiment, or the full repository
test suite. Those remain integration/hosted gates. It accepts the reviewed
contracts, focused runtime behavior, workflow definition, and artifact-hygiene
policy with zero unresolved high or medium finding.
