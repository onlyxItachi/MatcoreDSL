# Legacy Archaeology Agent Report

## Assignment

- Role: repository and backend archaeology specialist.
- Write ownership: `docs/mdslc/LEGACY_REUSE_MAP.md` and this report only.
- Baseline inspected: `351075e4d8af1880330b7c0474d701ca76776dfa`.
- Implementation branch: `mdslc/legacy-reuse`.

## Instructions read

The audit re-read all top-level files under `rules/` and the architecture-review
agent role. The key constraints retained in the reuse map are explicit target
routes, no silent CPU fallback, preservation of the Python API, target-specific
backend isolation, and clear labeling of implemented, scaffolded, and future
behavior.

## Evidence inspected

- `context.md` and root `CMakeLists.txt`;
- `include/matcore/kernel_ir.h` and related runtime/target/plan headers;
- `src/bindings.cpp`, `src/mlir_engine.cpp`, `src/jit_runner.cpp`,
  `src/executor.cpp`, and `src/cache_manager.cpp`;
- `src/runtime_capabilities.cpp` and `src/target_registry.cpp`;
- CPU, NVVM, AMD, and common lowering pipelines;
- RegionV1 verifier and emitter;
- the repository's Python tests, GPU scripts, and benchmark layout.

## Conclusions

- No existing named production component should be linked unchanged into the
  standalone frontend or C ABI runtime.
- The root build is a Python/nanobind/MLIR extension build and must remain
  independent from `compiler/`.
- Target parsing, runtime capability probing, structural verification, device
  ownership tokens, cache versioning, and backend pass ordering are useful
  substrate only behind explicit new interfaces.
- Legacy host-to-GPU staging performs hidden allocation/copy behavior and must
  not cross the standalone no-hidden-copy boundary.
- The reuse map defines a one-way, loss-checked bridge from verified JSON IR v0
  through planning into future existing-MLIR reuse. JIT orchestration remains
  outside that bridge.

## Validation performed

- Confirmed the worktree branch and baseline before editing.
- Confirmed both owned output paths were absent before creation.
- Reviewed the rendered Markdown structure and checked for all requested audit
  topics: Python/nanobind, JIT assumptions, hardcoded paths, ABI-like
  structures, device ownership, capabilities, verifiers, pass ordering, cache
  versioning, v0 exclusions, and the future bridge.
- No production build or runtime test was needed because this commit changes
  documentation only.

## Handoff

The integration owner should cherry-pick the focused documentation commit from
`mdslc/legacy-reuse`. No merge was performed by this agent.
