# Milestone 7 parity-harness lane

Ownership was limited to:

- `compiler/tests/performance/run_native_blas_parity.py`
- `compiler/tests/performance/run_native_blas_parity_tests.py`
- the native-parity CTest registration in
  `compiler/tools/matcore-bench/CMakeLists.txt`

No Milestone 6 runner or summarizer was changed.

## Result

The new Linux x86-64 runner makes the reviewed
`native-blas-parity-methodology-v1.md` matrix executable without weakening the
Milestone 6 evidence authority. It freezes the calibration and holdout shapes,
the serial and parallel candidate sets, the `1`, `4`, and caller-supplied
physical-core thread strata, repeated-input and authenticated prepacked-B
cases, and bounded built-in planner-regret diagnostics.

Every retained multi-thread parity case uses the same
`allow-smt`/`affinity=none` placement for native and OpenBLAS candidates. Every
retained forced result must use exactly the requested thread count and exact
stable variant ID. A clamped result or substituted implementation is a failure,
not evidence.

The runner authenticates:

- a completely clean Git source worktree;
- runner bytes against the tracked blob at exact `HEAD`;
- benchmark binary, runner, plan, raw-file, and source-commit identities;
- exact benchmark configuration and Linux x86-64 topology;
- ordered samples and reconstructed median/p95/minimum/GFLOP/s;
- final timed output and independent correctness replay;
- prepacked-B preparation and amortization arithmetic;
- balanced forward/reverse planner-regret candidate measurements.

Raw output is rejected inside the source repository or any other Git worktree.
Stable-forward and stable-reverse plans are exact inverses. Interrupted runs
persist an atomic manifest after each case, and resume reuse is accepted only
after full identity and raw-digest authentication. Explicit hardware/provider
legality failures use exact diagnostics; arbitrary failures and any failed
process that leaves a raw artifact remain hard failures.

## Validation

Fresh Release configuration used Clang/LLVM 21.1.8 and OpenBLAS 0.3.32.

```text
python3 -m py_compile \
  compiler/tests/performance/run_native_blas_parity.py \
  compiler/tests/performance/run_native_blas_parity_tests.py
ruff check \
  compiler/tests/performance/run_native_blas_parity.py \
  compiler/tests/performance/run_native_blas_parity_tests.py
cmake -S compiler -B <fresh> -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build <fresh> --target matcore-bench -- -j2
ctest --test-dir <fresh> \
  -R '^benchmark.cpu.native_blas_parity_runner_contract$' \
  --output-on-failure -j1
ctest --test-dir <fresh> \
  -R '^benchmark\.cpu\.(deep_audit_runner_contract|deep_audit_summary_contract|cli_json|provenance_incremental)$' \
  --output-on-failure -j1
git diff --check
```

Results:

- native BLAS parity runner contract: `1/1` passed;
- existing benchmark/deep-audit contracts: `4/4` passed;
- Python compilation: passed;
- Ruff: passed;
- diff check: passed.

The contract test exercises the complete dry-run matrix and its reverse, one
real guarded planner-regret case, authenticated resume reuse, plan and raw
tamper rejection, forced-substitution rejection, final-output-authentication
rejection, exact legality classification, non-empty-output rejection, and
source/other-Git-worktree path rejection.

This lane does not claim parity measurements. It supplies the authenticated
measurement boundary that the integration owner must run in both orders after
the candidate kernels and planner rules are finalized.
