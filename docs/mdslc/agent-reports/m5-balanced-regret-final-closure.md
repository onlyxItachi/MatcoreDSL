# Milestone 5 balanced-regret final closure review

## Scope and reviewed state

This independent review covered integration commit
`5f634aef2a0b47cd033df77c40d709456603b405`. It re-audited the prior balanced
planner-regret findings, the benchmark-caller isolation added by `c0b4409`, and
the build-time benchmark provenance contract added by `5f634ae`. Production
sources were read only in this lane.

## Verdict

**No high or medium finding remains in the reviewed benchmark methodology,
caller-placement, or source-provenance scope.**

The prior N1 finding is closed: equal-cardinality untimed correctness replay is
now placed after the forward timed block and before the reverse timed block.
The benchmark caller is isolated from bound workers when a spare CPU exists,
and the no-spare and truly unbound cases make no false isolation claim. Schema
v4 authenticates an exact build-time source commit and tracked-worktree state;
`--guard` rejects dirty, unknown, malformed, or inconsistent provenance.

This verdict does not turn a two-pass measurement into a universal unbiased
estimator. Warmup, probe, and aggregate-repetition calibration remain necessary
preconditioning before each timed pass, and the two passes calibrate
independently. Nonlinear frequency, thermal, provider, and cache effects remain
measurement limitations. The report exposes both pass medians, validation
cardinalities and placements so these effects are not hidden.

## N1 closure: mirrored validation placement

The recursive regret path now has these fail-closed properties:

- comparable candidates run in stable registry order and then exact reverse
  registry order;
- forward candidate runs use `after-timing` untimed validation and reverse
  candidate runs use `before-timing` validation;
- each untimed phase executes the same cardinality as its corresponding timed
  phase and checks every execution with the independent double-precision
  oracle;
- the final timed output is authenticated immediately after the timed block in
  both passes;
- the forced plan fingerprint includes validation placement as well as stable
  ID, timing scope, planner version, threads, workspace, packing and affinity;
- any plan drift, invalid timing, execution error or wrong output rejects the
  complete regret result;
- schema v3 and v4 require the forward and reverse placement fields and require
  their values to be `after-timing` and `before-timing`, respectively.

The recording-runner regression checks the complete forward/reverse plan order,
injects an intermediate corrupt forward replay and an intermediate corrupt
reverse replay, and proves that both mirrored positions reject before a later
correct output can overwrite the fault. Focused Release and ASan/UBSan runs of
`benchmark.cpu.contract` and `benchmark.cpu.cli_json` passed.

## Benchmark caller isolation

For a bound-worker plan, the runner deterministically prefers the
highest-numbered spare logical CPU on the selected NUMA node and on a physical
core unused by workers. It applies current-thread affinity, reserves all online
siblings of a dedicated caller core, removes the reservation from subsequent
worker topology, and replans. Later bound plans reauthenticate the caller mask.
Application or reauthentication failure is sticky and fails bound planning
closed. Mutable caller/context state is serialized by `context_mutex_`.

If no spare logical CPU exists, execution remains legal but diagnostics state
both `no spare logical CPU` and
`caller_scheduler_affinity_applied=false`. A fresh
`--allow-smt --affinity none` run takes the unbound-context path before caller
isolation and does not pin the caller. This is important for provider-managed
OpenBLAS execution.

Five independent schema-v4, guarded `256x256x256`, two-thread compact-affinity
regret runs at the reviewed commit all reported workers `[0,1]`, caller CPU
`23`, a dedicated caller core, and reserved siblings `[11,23]`. Every legal
complete-call candidate was plan-authenticated, correct, timing-valid and
carried the mirrored validation fields. The observed regret values were:

- 1.0000
- 1.0000
- 1.0000
- 1.5434
- 1.1345

The median was **1.0000** and the maximum was **1.5434**. No prior
scheduler-quantum-scale outlier recurred. This five-run sample is a focused
closure probe, not the final full shape-matrix calibration.

A separate schema-v4 OpenBLAS run with
`--allow-smt --affinity none --threads 2` retained
`worker_affinity_applied=false`, reported two provider threads, and explicitly
said caller affinity was not requested. A one-CPU `taskset` run reported the
no-spare condition without claiming caller isolation. Both were correct and
timing-valid.

Raw closure JSON remains outside Git under:

```text
/home/hamza-usta/archives/MatcoreDSL-M5-perf-20260722/final-v4-independent-review/
```

## Build-time provenance v4

The prior configure-time commit definition is gone. An always-run CMake target
now executes `GenerateBenchmarkProvenance.cmake` before the benchmark core. The
script verifies the exact Git top-level directory, reads the full 40- or
64-hex `HEAD`, checks tracked staged and unstaged changes, and writes a generated
header through `copy_if_different`. No source checkout path is embedded in the
binary.

Schema v4 adds required, typed environment fields:

- `source_worktree_dirty`;
- `source_provenance_state` (`clean`, `dirty`, or `unknown`);
- `source_provenance_origin` (`git-worktree`, `explicit-override`, or
  `unavailable`).

An isolated-clone incremental proof configured once and then exercised three
states:

1. The clean build embedded
   `5f634aef2a0b47cd033df77c40d709456603b405` and passed `--guard`.
2. A no-op build left both generated-header and benchmark-object mtimes
   unchanged.
3. An empty commit advanced the clone to
   `fbea1190ca5b45c79c84c2fb20174a745d609b7a`; rebuilding without CMake
   reconfiguration refreshed the header, recompiled the benchmark core and
   emitted that exact new commit.
4. A staged tracked mode change rebuilt with `source_worktree_dirty=true` and
   state `dirty`; JSON remained inspectable, while `--guard` returned 1 with
   `tracked source worktree was dirty when matcore-bench was built`.

The dedicated incremental test also passed clean, unchanged, tracked-dirty,
new-commit, archive-unknown and explicit-override cases, including paths with
spaces. Unknown archive provenance fails guarded certification unless the
builder deliberately supplies both exact-commit and state overrides; the
origin remains visible as `explicit-override`.

Strict historical schemas are preserved byte-for-byte across the v4 change:

- v2 SHA-256:
  `1f7f7b34fa1a8c92958528cb00eca397513e80e372fe83b64b472ab59195e1a9`;
- v3 SHA-256:
  `ffeeb54fb68b1a76c2b5c70fc7eba62c6b99493dd5b44b73149d93680517120e`.

## Evidence boundary

The root integration owner's earlier raw schema-v3 matrix under
`v3-final-*` correctly embeds
`e149d64bf840b326b23e37b64536b713072dae2f`; it is exact evidence for that
checkpoint. It is not final-tip evidence after `5f634ae`. Final milestone
calibration and published summaries must therefore be rerun through schema v4
and pass guarded clean-source provenance. This review performed representative
v4 reruns at `5f634ae`; it did not substitute them for the owner's complete
declared shape and thread matrix.

## Commands and results

Focused Release validation:

```text
cmake --build /tmp/matcore-m5-caller-tip-release \
  --target matcore-bench matcore_benchmark_core_test -- -j2
ctest --test-dir /tmp/matcore-m5-caller-tip-release \
  -R '^benchmark\.cpu\.(contract|cli_json|provenance_incremental)$' \
  --output-on-failure -j1
3/3 passed
```

Fresh focused ASan/UBSan validation:

```text
cmake -S compiler -B /tmp/matcore-m5-final-closure-asan -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF \
  -DMDSLC_ENABLE_OPENBLAS=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build /tmp/matcore-m5-final-closure-asan \
  --target matcore-bench matcore_benchmark_core_test -- -j2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir /tmp/matcore-m5-final-closure-asan \
  -R '^benchmark\.cpu\.(contract|cli_json|provenance_incremental)$' \
  --output-on-failure -j1
3/3 passed
```

Additional checks:

- direct clean guarded schema-v4 emission at the exact reviewed SHA: passed;
- configure-once commit-transition provenance proof: passed;
- direct dirty-build guarded rejection: passed;
- five guarded compact-affinity balanced-regret runs: passed;
- fresh unbound two-thread OpenBLAS run: passed;
- one-CPU explicit no-spare diagnostic: passed;
- v2/v3 unchanged checks and `git diff --check`: passed.

## Remaining low-severity limitations

- Forward/reverse balancing reduces systematic registry-order drift but does
  not prove immunity to nonlinear thermal, frequency, cache or provider-state
  effects. Use adequate warmups and repetitions and report pass disagreement.
- Independently calibrated forward/reverse aggregate counts may differ; their
  exact validation counts are emitted rather than hidden.
- A no-spare run is valid correctness evidence but must not be used as a
  caller-isolated performance claim.
- Build provenance observes Git state when the build-time generator runs; the
  source worktree must not be mutated concurrently with compilation.
- The complete final calibration matrix still belongs to the integration
  owner and must use guarded schema-v4 output from the final source commit.

## Acceptance recommendation

Accept the balanced-regret methodology, caller-placement hardening and
provenance v4 changes. The original M1-M3 and N1 review findings are closed.
Use only guarded schema-v4 reruns from the final milestone commit for final
aggregate regret or performance claims.
