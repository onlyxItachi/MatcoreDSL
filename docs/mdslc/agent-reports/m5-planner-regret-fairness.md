# Milestone 5 planner-regret fairness report

## Ownership and baseline

This lane owned only:

- `compiler/tools/matcore-bench/benchmark.cpp`;
- focused benchmark tests under `compiler/tests/benchmark/`;
- this report.

The isolated branch `mdslc/m5-regret-fairness` began at integration commit
`74d36e8956a629d66c74606b12d67ba9f31fca30`. No planner, runtime, kernel,
package, or legacy source was changed.

## Finding and resolution

Planner-regret previously reused the primary auto-selected timing, then timed
each alternative once in forward registry order. The selected candidate
therefore had a different measurement history, and forward-only ordering could
turn transient frequency, thermal, or scheduler drift into false regret.

The corrected contract is:

1. Preflight every registry candidate once and preserve registry order in the
   output.
2. Measure every legal complete-call candidate in stable forward registry
   order.
3. Measure the same candidates in exact reverse registry order.
4. Define each reported candidate median as the equal-weight arithmetic
   midpoint of its forward-pass median and reverse-pass median.
5. Derive the selected median from its candidate's balanced measurement, never
   from the earlier primary result.
6. Reject the regret result if either pass has invalid timing or failed
   correctness. Execution failures reject the run immediately with the pass
   and variant in the diagnostic.

The existing JSON schema fields remain sufficient: `median_seconds` now holds
the balanced midpoint, while `measurement_reason` and the enclosing regret
`reason` state the exact aggregation and ordering contract. No raw benchmark
artifact was added to Git.

## Focused verification

Configured a fresh Release build with Clang/LLVM 21.1.8 and required OpenBLAS
0.3.32 pthread:

```sh
cmake -S compiler -B /tmp/matcore-m5-regret-fairness-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=ON \
  -DMDSLC_REQUIRE_OPENBLAS=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang
cmake --build /tmp/matcore-m5-regret-fairness-release \
  --target matcore_benchmark_core_test matcore-bench -- -j2
ctest --test-dir /tmp/matcore-m5-regret-fairness-release \
  --output-on-failure \
  -R '^benchmark\.cpu\.(contract|cli_json)$' -j1
```

Result: **2/2 tests passed**.

The deterministic recording-runner regression verifies the exact plan order
after preflight, stable output order, selected-timing derivation, explicit
midpoint wording, and reverse-pass failure rejection. A real guarded 64 cubed
run also completed with valid correctness and emitted a balanced selected
median distinct from its primary timing; that raw JSON remained under `/tmp`.

`clang-format-21` was not installed, so a formatter dry run was unavailable.
Both changed translation units compiled under the project's warning-as-error
Release configuration, and `git diff --check` passed.
