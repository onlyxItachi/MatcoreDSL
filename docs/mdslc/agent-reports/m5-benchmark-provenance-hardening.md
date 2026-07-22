# Milestone 5 benchmark provenance hardening

## Ownership and defect

This lane changed only the `matcore-bench` build-time provenance path, benchmark
JSON contract, and focused tests. It did not change planner costs, runtime
execution, or GEMM kernels.

The previous `MATCORE_BENCH_SOURCE_COMMIT` definition was computed during CMake
configuration. A later Git commit followed by an incremental build therefore
left the executable reporting the old commit until CMake happened to
reconfigure.

## Implementation

`matcore_benchmark_provenance_v4` is an always-run build target. Its CMake
script authenticates that the configured source root is the exact Git
top-level directory, obtains the full 40- or 64-hex `HEAD`, and inspects only
tracked staged/unstaged changes. It writes a generated header through
`cmake -E copy_if_different`. Consequently, a changed commit or dirty state
recompiles the benchmark core, while an unchanged build does not touch the
header or object.

No source path is compiled into the executable. A non-Git source archive
records `unknown` provenance by default. Reproducible archive builders may set
both cache variables explicitly:

```text
MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE=<exact 40- or 64-hex object ID>
MDSLC_BENCH_SOURCE_STATE_OVERRIDE=<clean|dirty|unknown>
```

The override origin remains visible as `explicit-override`; it is not presented
as a live Git worktree observation.

## JSON v4 migration

Strict schema v3 is preserved byte-for-byte because it rejects additional
properties. The default report is now schema v4 and adds these required
environment fields:

- `source_worktree_dirty`: boolean tracked-worktree status;
- `source_provenance_state`: `clean`, `dirty`, or `unknown`;
- `source_provenance_origin`: `git-worktree`, `explicit-override`, or
  `unavailable`.

`source_commit` remains the exact object ID when known. `--guard` now fails
closed for dirty, unknown, malformed, or internally inconsistent source
provenance and emits an actionable diagnostic. JSON is still written before
the guarded nonzero exit so rejected evidence remains inspectable.

## Focused validation

- Release benchmark core compiled with Clang 21.1.8.
- `benchmark.cpu.contract` passed.
- `benchmark.cpu.provenance_incremental` passed across clean, unchanged,
  tracked-dirty, new-commit, archive-unknown, and explicit-override states,
  including paths containing spaces.
- A repeated build left both generated-header and benchmark-object timestamps
  unchanged.
- A dirty live-worktree benchmark emitted schema v4 evidence and `--guard`
  returned 1 with the tracked-dirty diagnostic.
- A separate explicit-clean archive-style build passed
  `benchmark.cpu.contract`, `benchmark.cpu.cli_json`, and
  `benchmark.cpu.provenance_incremental` (3/3).
- The v4 schema parsed as JSON and differs from strict v3 only by version and
  the three provenance fields.

The integration owner must still run the clean live-Git incremental proof after
this focused commit exists; that proof is what verifies the embedded SHA moves
without reconfiguration in the real build tree.
