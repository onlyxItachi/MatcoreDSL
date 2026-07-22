# Milestone 4 compute-only and allocation review fixes

Date: 2026-07-22

Base: `8038da0c6bd74846edaaca7be087518266311272`

Scope: independent-review findings M4-R2 and M4-R3 only. No planner,
production runtime, OpenBLAS adapter, or legacy implementation was changed.

## M4-R2: honest compute-only diagnostic

`matcore-bench --exclude-packing` now has one explicit implementation:
`cpu.native-packed.avx2-fma.f32.v1`. Its preparation phase packs complete A
and B full-K micro-panels into caller-visible benchmark workspace before any
warmup, probe, or measured interval. The timed execution walks those panels
and invokes the exact production 4x16 AVX2/FMA microkernel, including dispatch,
edge handling, and output stores. It performs no packing or allocation.

The result is deliberately a microkernel diagnostic, not an end-to-end native
GEMM result. JSON and human output report:

- `packing_mode=exclude-packing`;
- `timing_scope=packed-compute-only: ...`;
- `complete_implementation_comparison=false`;
- `comparison=diagnostic-only` in human output.

Consequently this mode is not eligible for planner-regret or complete
OpenBLAS comparisons. Reference, tiled, compiler-vectorized, and OpenBLAS
requests reject `--exclude-packing` because none exposes a separable
benchmark-managed packing phase. OpenBLAS provider-internal packing remains
opaque and inside its complete call. `--include-packing` still times transient
A+B packing and compute. `--prepack-b` still prepares B outside timing while
timing transient A packing and compute.

Focused correctness covers 1x1x1, 2x3x2, 5x17x19, 33x35x37, and
127x129x131. Every supported run uses the independent double-precision oracle.

## M4-R3: one-shot allocation and peak accounting

`--include-allocation` no longer constructs unused reusable output or
workspace buffers outside the interval. Each invocation destroys the previous
one-shot output/workspace pair before planning and allocating the next pair,
preventing a temporary double-live replacement allocation. The final output
remains alive for correctness verification.

The pre-allocation bound now accounts for every simultaneously live
`AlignedBuffer` as:

`payload + 2 * requested_alignment + alignof(max_align_t)`

This is applied independently to lhs, rhs, output, workspace, and prepacked-B
storage when present. The cold-cache vector payload is also included. A unit
boundary proves that the exact calculated peak passes and one byte below it
fails. Packing-excluded and prepacked-B modes require reusable workspace, so
their preparation cannot be accidentally combined with one-shot allocation.

## Validation evidence

Release configure/build used Clang 21.1.8 with native frontend and OpenBLAS
enabled:

```text
cmake -S compiler -B /tmp/matcore-m4-compute-fix-build -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-m4-compute-fix-build -- -j2
ctest --test-dir /tmp/matcore-m4-compute-fix-build \
  --output-on-failure -j1
```

Result: 27/27 tests passed. The focused benchmark contract and CLI/JSON tests
passed in 0.14 seconds.

The ASan/UBSan Debug build used
`-fsanitize=address,undefined -fno-omit-frame-pointer` with leak detection and
stack traces enabled. `benchmark.cpu.contract` and `benchmark.cpu.cli_json`
passed 2/2, including the packed tiny/rectangular/tail diagnostic and exact
memory-cap tests. No sanitizer diagnostic was emitted.

Direct validation on the review host for 33x35x37 reported a valid
`diagnostic-only` interval, `workspace_bytes=12480`, and independent-oracle
correctness. Transient-packing and prepacked-B executions for the same tail
shape also passed. The diagnostic throughput is intentionally not used as an
end-to-end or OpenBLAS performance claim.

`git diff --check` passed. No generated benchmark JSON or build artifact is
included in this change.
