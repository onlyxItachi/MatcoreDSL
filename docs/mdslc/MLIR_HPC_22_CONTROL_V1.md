# MLIR 22.1.8 schedule control experiment

2026-09-09 local date. **Research compatibility evidence, not a product migration.**
Canonical source start: `0e8c040809ae1fe6b0f1c01e41fc509b07212be7`.
Unmodified research inputs and runner:
[`6ce4b3f6035f1a4ae8f5395753f9e0202eabdffc`](https://github.com/onlyxItachi/MatcoreDSL/tree/6ce4b3f6035f1a4ae8f5395753f9e0202eabdffc/compiler/experiments/mlir_hpc_v1).
No production source, toolchain installation, runtime, ABI or execution issuer changed.

## Observed result

The same three upstream Transform schedules used in the 21.1.8 research lane
ran through coherent 22.1.8 MLIR, translation, LLVM optimization and actual
Linux x64 execution without changing a pass name, option or input:

| Schedule | Ordinary execution | Generated ASan + harness ASan/UBSan |
| --- | --- | --- |
| Masked vector 4x8x1 | 77,985 checks, zero failures | 77,985 checks, zero failures |
| Interchanged masked vector 4x1x8 | 77,985 checks, zero failures | 77,985 checks, zero failures |
| Cache tile 4x64x32, inner M/K/N | 77,985 checks, zero failures | 77,985 checks, zero failures |

All three intentionally invalid input capacities caused the required generated
heap-buffer-overflow; this is instrumentation evidence, not a promise that the
unguarded private leaf validates capacities. The unchanged runner completed
with status zero. Cases include random rectangles, exact-size tails, shared
inputs, both carried GEMMs, exceptional f32 values and K32/K64 split-sum
counteroracles; see the linked runner/report for exact scope.

The cache schedule's printed scheduled MLIR was **byte-identical** between
21.1.8 and 22.1.8. Its LLVM IR and machine object changed. Actual 22 cache object
contains separate SSE `mulps`/`addps`. The two explicit-vector variants again
produce identical machine objects to each other. No timing was performed;
different code generation is not evidence of a speedup.

The deliberately incompatible vector-contract/outer-product alternative still
lowers to `llvm.fmuladd`. Baseline x64 execution printed `00000000`; actual
x86-64-v3 execution on the already validated Ryzen AI 9 HX 370 printed
`337ffffe` against strict expected `00000000`, returning the expected failure
status 1. Thus upgrading alone does not resolve this numerical-contract trap.

## Reproduction and artifact identity

Existing coherent toolchain:
`/home/hamza-usta/.local/toolchains/llvm-22.1.8-linux-x64`.
Clang identifies upstream commit `ca7933e47d3a3451d81e72ac174dcb5aa28b59d1`.
Run the linked research `run.sh` with its `MDSLC_RESEARCH_MLIR_BIN`,
`MDSLC_RESEARCH_CLANGXX`, `MDSLC_RESEARCH_CLANG` and `MDSLC_RESEARCH_OPT`
overrides pointing to that tuple. `MDSLC_RESEARCH_RUN_V3=1` was used only on
the previously physically validated v3 host; do not assume that permission on
an arbitrary reproduction machine.

Local generated directory: `/tmp/mdslc-mlir-22-control.Knk65l`.

| Artifact | SHA-256 |
| --- | --- |
| `mlir-opt` | `45092a31bf61175e2917ce59e12e71f8e287a73bb6bb241142b362f2c1e03850` |
| `mlir-translate` | `d7b45c78ebde8830f3951f5f7845b9c584d33dcafebc93aafdc32ae53168fb65` |
| Clang | `31a40dc31f3c15a47aa119aa148f339bf363d8f61202ddbdacbd3f24b71ba113` |
| `opt` | `fc19bcd0ab81c35746c93aad355e8bb81f34ab070e98014cf06ce8433e91d4b1` |
| Both vector objects | `2a6429c7c09f6fd8cb5fd4d313cca524a807e5a29715e8cfa0ef23335b138ea9` |
| Cache object | `0e2fde92f2401128f754a9efc60365ae7a5acb837913ae2d9dd3e339e7d81c71` |
| Cache LLVM | `edce9fa3de60ddd6aa167d8dc47f8463577c948246d5ad5b0b528da72ee47c38` |
| Cache scheduled MLIR | `39d37ea30f438b954aebdf016438ef6634991a3f9f83acabf1f56bd91f4de3d1` |
| Cache ASan object | `5144971b77549c706d844d2bd56c9bca3c378a9c0222f7051f33a1109c747203` |

## Decision and limits

**Inferred:** these schedule mechanisms do not need a permanent Matcore-specific
replacement to bridge these two versions. Keep product issuance pinned to
21.1.8: this experiment neither recompiles the MDSLC C++ API integration nor
validates its 22 product driver, packaging, source authentication or hosted CI.
It does not establish that every Transform/Vector interface is compatible.
The `test-transform-dialect-erase-schedule` CLI helper remains research-only;
production uses separate schedule/payload modules and upstream interpreter APIs.

Preserve this as a version-control specimen, not a new platform capability or
reason to mix an upgrade into the current schedule proof. There is no new
private schedule language, target backend, storage policy or performance claim.
