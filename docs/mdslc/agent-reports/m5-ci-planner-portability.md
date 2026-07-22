# Milestone 5 hosted planner-resource portability fix

Date: 2026-07-22

Base: `b07cb775fa1b7bf2263427a68206605af922680a`

Test correction: `ff039563eda18f90e07804fc07430fd2cfde3ffa`

## Failure and root cause

GitHub Actions run `29952201174`, job `89032387866`, failed only
`runtime.cpu.planner_v3_resources` in the OpenBLAS-disabled matrix.  The runner
correctly skipped `runtime.cpu.packed_avx512`, proving AVX-512 was not usable.

The failing test assumed seven worker submissions whenever OpenBLAS was
disabled.  The validation flow actually makes five unconditional serial
submissions (reference, tiled, compiler-vectorized, serial packed AVX2, and
serial packed AVX-512), then adds an OpenBLAS submission only when linked and
one parallel submission for each runtime-usable packed ISA.  Parallel packed
execution intentionally returns `isa_unavailable` before `run_tasks` when its
ISA is unavailable.  Therefore the hosted AVX2-only runner correctly made six
submissions, while the AVX2+AVX-512 development host made seven.

## Resolution

The test now derives the expected count as:

```text
5 + openblas_linked + avx2_runtime_usable + avx512_runtime_usable
```

It also independently authenticates the exact evidence bits for
compiler-vectorized, OpenBLAS, serial AVX2, serial AVX-512, parallel AVX2, and
parallel AVX-512 against their legal process-local gates.  No production code,
runtime gate, fallback behavior, or provider behavior changed.

## Validation

Fresh external Release configuration used Clang/Clang++ 21.1.8, native and
bootstrap frontends enabled, OpenBLAS explicitly disabled, Ninja `-j2`, and
CTest `-j1`:

```sh
cmake -S compiler \
  -B /tmp/matcore-m5-ci-planner-portability-build -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=OFF \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-m5-ci-planner-portability-build -- -j2
ctest --test-dir /tmp/matcore-m5-ci-planner-portability-build \
  --output-on-failure -j1
```

Results at the clean test-correction commit:

- focused `runtime.cpu.planner_v3_resources`: **1/1 passed**;
- complete OpenBLAS-disabled Release suite: **42/42 passed** in 65.13 s;
- `git diff --check`: passed;
- repository hygiene: passed.

An earlier whole-suite run made while the test edit was deliberately
uncommitted passed the planner-resource test and 39 other tests; only the two
benchmark provenance guards rejected the dirty worktree, as designed.  The
reported 42/42 result above is the clean-commit rerun after provenance was
refreshed.

## Verdict

The failure was a host-capability-dependent test expectation.  The correction
preserves fail-closed ISA dispatch and makes the validation portable across
AVX2-only and AVX2+AVX-512 hosts.  No unresolved product finding remains in
this lane.
