# Milestone E CPU MLIR lowering independent review

- Date: 2026-08-11
- Reviewer lane: independent; no implementation files edited
- Primary implementation: `7171a1efffbd8d590818a3a27a74792a6ecc6afb`
- CPU lowering dependency: `c914b4a`
- Executable-authority fix: `230e14278cb9d63af031b105238e4f57a3c210a2`
- Evidence-report head inspected: `1ee35a67433e151e83eb1b10725319c7eb37fdc9`
- Design contract: `docs/mdslc/CPU_MLIR_LOWERING_V1.md`
- Verdict: accepted for the focused Milestone E implementation surface; no
  unresolved high- or medium-severity finding

## Scope and acceptance boundary

This review attempted to reject the opt-in path:

```text
authenticated native Matcore IR v1
  -> verified Matcore MLIR explicit-GEMM module
  -> strict CPU runtime-dispatch record
  -> generated private C backend
  -> matcore_runtime_gemm_f32_v0
  -> ordinary object or executable
```

The implementation is accurately described as runtime-dispatch library
lowering. It does not claim Linalg/Vector loop generation, recovered-loop
execution, map execution, GPU lowering, or a new public ABI.

## Resolved executable-authority findings

The first review pass found one medium-severity authority gap with two concrete
forms:

1. the general v1 bridge verifier permits the compatibility bootstrap producer
   for inspection, and the executable lowerer did not independently require the
   native producer; and
2. an otherwise explicit module carrying `mdsl.analysis_only` was not rejected
   at the executable boundary.

Commit `230e142` resolves both forms in the shared CPU lowerer:

- any presence of `mdsl.analysis_only` is rejected before structural bridge
  verification;
- executable lowering requires the exact producer
  `clang-libtooling-v1` after strict structural verification; and
- `records` is cleared before either rejection, so a caller cannot observe a
  stale or partially published dispatch record.

Focused negatives mutate an otherwise valid explicit module with each taint.
They prove actionable diagnostics and empty output records. A real recovered
analysis module is also rejected at the same boundary. The general bridge can
continue to inspect bootstrap captures without granting them execution
permission.

## Adversarial review results

### Frontend and option selection

- The semantic route requires an explicit
  `--semantic-pipeline=matcore-mlir`, explicit Matcore IR version 1, and the
  native frontend.
- Duplicate and unknown selectors fail closed.
- A selector without `--matcore-target=cpu` is rejected by `mdslc++`.
- Bootstrap plus semantic execution is rejected by both the driver and direct
  extractor, with no silent capture-v0 fallback.
- Options after `--`, and values belonging to compiler options that consume
  their next argument, remain compiler arguments rather than being
  reinterpreted as MDSLC wrapper controls.
- When MLIR support is not built, an explicit semantic request fails with an
  actionable diagnostic and no output artifact.

### Semantic authenticity and legality

- The executable path does not parse caller-supplied MLIR. It bridges the
  verified typed capture held by the native extraction process.
- The strict bridge verifier authenticates the closed explicit F32 GEMM
  structure, CPU/error policy, numerical profile, destination/result identity,
  effects, alias requirements, tensor contracts, and source provenance.
- The executable lowerer now adds the native-producer and analysis-only gates
  that are intentionally broader in the inspection verifier.
- Unsupported map/domain composition, recovered origins, altered execution
  intent, malformed numerical policy, or broken structural contracts fail
  before backend generation.
- Empty semantic modules remain valid deliberately, allowing ordinary C++ with
  no Matcore sites to pass through unchanged.

### Backend, ABI, and publication

- The lowering record retains the canonical site, exact CPU/error and
  numerical envelope, source range, and required runtime-guard families.
- Backend generation consumes a normalized site entry and emits the existing
  private `matcore_generated_backend_<site>_v0` function.
- That function forwards through the stable
  `matcore_runtime_gemm_f32_v0` C descriptor ABI. No MLIR or C++ type crosses
  the runtime ABI.
- The semantic backend replaces the capture-v0 backend when the semantic route
  is selected; the `.mlir` file is not merely an unused inspection sidecar.
- Semantic verification and backend-text generation finish before any output
  is published. Individual extractor destinations use temporary-file rename.
  The report does not claim a group-atomic transaction across all destination
  files.
- Output/input and output/output identity checks cover lexical aliases,
  symlinks, and hard links. Driver save-temp semantic MLIR also participates in
  dependency-closure snapshots and mutation-path collision checks.
- Runtime descriptor, overlap, alignment, workspace, and FP-environment guards
  remain on the stable dispatch route; the generated shim performs no output
  write before that call.

### Determinism, defaults, and packaging

- Repeated direct extraction produced byte-identical capture IR, semantic MLIR,
  rewritten source, sites, stubs, and backend bytes.
- The driver proof produced an ELF relocatable object, an ordinary externally
  linked executable, the expected generated-backend symbol, and the unresolved
  stable runtime reference before final link.
- `MDSLC_ENABLE_MATCORE_MLIR` remains `OFF` by default.
- The internal MLIR semantic libraries are neither installed nor exported.
  The optional `matcore-mlir` tool is installed only as a leaf executable.
  Generated CMake export/config files contained no MLIR target, worktree, user
  home, or local MLIR package path.
- The build-tree extractor dynamically needs the existing coherent Clang/LLVM
  21 libraries and has the expected build-tree LLVM runpath. The install rule
  clears that build runpath; this focused review does not relabel the build-tree
  runpath as an installed-path leak.

## Reproduced evidence

The exact opt-in build used Clang/LLVM/MLIR 21.1.8 and serialized compilation at
low priority with at most two jobs:

```text
nice -n 10 cmake --build \
  /home/hamza-usta/.cache/mdslc-semantic-e-integration-d131ba3 \
  --target matcore_mlir_semantics_tests \
           matcore_mlir_cpu_runtime_lowering_tests \
           matcore_mlir_recovered_gemm_tests \
           matcore-extract mdslc_driver -j2
```

Result: build passed.

```text
nice -n 10 ctest --test-dir \
  /home/hamza-usta/.cache/mdslc-semantic-e-integration-d131ba3 \
  -R '^(mlir\.semantic\.core|mlir\.cpu\.runtime_dispatch_lowering_v1|mlir\.semantic\.recovered-gemm|integration\.semantic_mlir_cpu_pipeline)$' \
  --output-on-failure -j2
```

Result: **4/4 passed** in 7.77 seconds.

Direct focused binaries additionally reported:

```text
CPU runtime-dispatch lowering: 18 checks, 0 failures
Matcore recovered GEMM bridge: 78/78 checks passed
```

The default-OFF build was rebuilt independently and exercised without MLIR:

```text
nice -n 10 cmake --build \
  /home/hamza-usta/.cache/mdslc-semantic-e-default-off-d131ba3 \
  --target matcore-extract mdslc_driver -j2

nice -n 10 ctest --test-dir \
  /home/hamza-usta/.cache/mdslc-semantic-e-default-off-d131ba3 \
  -R '^integration\.semantic_mlir_unavailable$' \
  --output-on-failure -j2
```

Result: build passed; unavailable gate **1/1 passed** in 0.41 seconds.

Additional near-miss probes in this review session established:

- hard-linked IR and semantic-IR destinations were rejected with both sentinel
  byte sequences preserved;
- duplicate semantic selectors were rejected without producing an artifact;
- omitted selector and explicit `capture-v0` produced byte-identical typed IR
  (`sha256:28f5ff7aca3edfbc8b23f49bc8b9647c388d96c632de03aa15721a3caf28c5b7`);
- a bootstrap producer and an explicit module carrying analysis-only taint each
  failed transactionally; and
- repeated semantic extraction and driver execution remained deterministic and
  numerically correct in the focused integration test.

## Findings and remaining gates

No high- or medium-severity finding remains in the reviewed implementation.

The following are acceptance boundaries, not claims of failure:

- this was the requested focused review, not a replacement for the later full
  Release, Debug, sanitizer, install/relocation, external-consumer, repository
  hygiene, ABI, and Windows beta gates;
- Windows semantic execution was not run by this Linux-focused review; and
- the semantic pipeline remains explicitly opt-in and default-OFF pending those
  broader integration gates.

Within that boundary, the explicit Matcore IR v1 to Matcore MLIR to stable CPU
runtime-dispatch object/executable proof is accepted.
