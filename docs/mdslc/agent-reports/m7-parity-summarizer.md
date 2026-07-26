# Milestone 7 parity-summarizer lane

## Ownership

This isolated lane changed only:

- `compiler/tests/performance/summarize_native_blas_parity.py`;
- `compiler/tests/performance/summarize_native_blas_parity_tests.py`;
- this report.

It did not modify the active Milestone 7 implementation worktree, production
planner/runtime code, the benchmark runner, or CMake registration.

## Result

The new summarizer treats the two complete stable-order benchmark bundles as
untrusted input and fails closed before emitting a sanitized result. It
authenticates:

- manifest schema v3, benchmark schema v6, exact suite/timing/seed/memory
  contract, and complete case accounting;
- the benchmark executable SHA-256, local tracked runner SHA-256, runner Git
  blob, and the runner blob at the declared source commit;
- the independently reconstructed manifest-v3 case matrix, corrected
  partition chronology, per-shape
  task-capacity projection, stable order, command vector, plan SHA-256, raw
  filename, and raw SHA-256;
- clean Linux x86-64 source provenance, one homogeneous physical topology and
  build/provider environment, exact requested/actual forced thread counts, and
  manifest-v3 parallel task geometry;
- ordered timing samples, aggregate timer-floor coverage, minimum/median/p95,
  GFLOP/s, correctness tolerance, authenticated final output, prepacked-B
  preparation/amortization, and balanced planner-regret arithmetic;
- exact forward/reverse semantic-cell coverage, rejection status, selected
  implementation, actual thread count, placement, checksum, and planner-regret
  presence.

Only paired cells are aggregated. The deterministic sanitized JSON/Markdown
reports contain no raw paths. They report fastest-native/OpenBLAS ratios per
declared family and exact thread count, the full comparison matrix, native
scaling, prepacked-B evidence, planner regret, missing comparisons, and each
Milestone 7 acceptance threshold. `--require-pass` turns a valid non-passing
performance result into exit status 1, while malformed or incomplete evidence
uses exit status 2. Measured bounded regret is acceptance-enabled, so adverse
values fail the bounded summary. Full-envelope regret coverage and
capacity-limited original thread ceilings remain separate manual Milestone 7
gates.

The ratio gate is intentionally not satisfied by an automatic plan that chose
OpenBLAS. Native parity uses only packed or persistent-parallel MDSLC
variants. Multi-thread ratios require exact requested/actual equality and the
same unbound placement as OpenBLAS. The scaling diagnostic clearly retains its
different compact-one-thread versus unbound-multi-thread placement boundary.

## Adversarial coverage

The focused contract test synthesizes the entire manifest-v3 matrix in both
directions and verifies a deterministic bounded-evidence result. It separately
checks a favorable passing result, an adverse-performance failure, and rejects:

- changed raw bytes with an unchanged digest;
- changed timing fields with an attacker-updated raw digest;
- selective case omission with an attacker-updated plan digest;
- forward/reverse automatic cells with unequal actual thread counts;
- a forged runner digest;
- a changed benchmark executable.

The synthetic fixture also exercises capacity-limited exact thread strata,
parallel task metadata, prepacked-B arithmetic, planner regret, family/thread
ratios, four-thread scaling, and every performance verdict gate.

## Validation

```text
python3 -m py_compile \
  compiler/tests/performance/summarize_native_blas_parity.py \
  compiler/tests/performance/summarize_native_blas_parity_tests.py
ruff check \
  compiler/tests/performance/summarize_native_blas_parity.py \
  compiler/tests/performance/summarize_native_blas_parity_tests.py
python3 \
  compiler/tests/performance/summarize_native_blas_parity_tests.py \
  --summarizer \
  compiler/tests/performance/summarize_native_blas_parity.py
git diff --check
```

Results:

- Python compilation: passed;
- Ruff: passed;
- complete synthetic forward/reverse matrix: passed;
- deterministic Markdown/JSON rerender: passed;
- six adversarial rejection classes: passed;
- diff check: passed.

## Integration handoff

The integration owner should register the test next to the existing parity
runner contract after the manifest-v3 runner changes land:

```cmake
add_test(
  NAME benchmark.cpu.native_blas_parity_summary_contract
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/performance/summarize_native_blas_parity_tests.py"
    --summarizer
      "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/performance/summarize_native_blas_parity.py"
)
```

The final raw sweep must be generated from a clean commit that already contains
the manifest-v3 runner. Run the summarizer while the authenticated benchmark
binary remains available; then commit only its sanitized report, never either
raw bundle.
