# MDSLC compiler-archaeology re-entry reconciliation v1

Date: 2026-09-04

Status: accepted documentation-only reconciliation. No compiler, planner,
runtime, provider, public API, or ABI behavior changed.

## Repository and corpus identity

The canonical starting point was clean `main` and `origin/main` at
`6708b48a5647698469a9af191941bd4755adab7b`, the normal merge of PR #18. The
annotated tag object
`4259112d13325b7abdec36337478f1a16711f852` for
`mdslc-cpu-beta-v1` peels to that commit. Live GitHub state on 2026-09-04 showed
the post-merge `ci` run 31526332463, `mdslc-native` runs 31526332472 and
31528637110, `mdslc-windows` runs 31526332616 and 31528637169, and
`repository-hygiene` runs 31526332556 and 31528637095 passing. Issue #17 and
milestone #6 are closed, no GitHub Release exists for the tag, and Issue #15 /
milestone #5 remain open.

The corpus was inspected from
`origin/research/windows-lowering-corpus-llvm20-22` at
`8aab0e295eb41ca5d2e1bd52c47201b05de9636b`. Its merge base with canonical
`main` is exactly the CPU-beta merge commit above. This research branch also
contains later experimental product changes; the branch as a whole is not an
accepted implementation source.

Stable control-plane identities are:

| Artifact | SHA-256 | Indexed coverage |
| --- | --- | --- |
| `corpus/manifests/windows-corpus-v1.json` | `8d5d3e6d6af89fd78d3af5bd53b530a96e55920cbda65c38d30e7646cd01e187` | 15 cases, 75 artifacts |
| `corpus/manifests/windows-corpus-v2-archaeology.json` | `68435d26abc4e498a76432290446bddfd428a64b81e59fe4640218016d3c754d` | 15 cases, 120 artifacts |

The committed `corpus/` control plane has 116 files: 11 inputs, 3 recipes, 3
environment descriptors, 4 schemas, 2 manifests, 3 fingerprint ledgers, 68
findings, 2 gold files, 19 scripts, and its README. The separate committed
`proof/` plane has 41 files and is not indexed by either manifest. Its MLIR
proof result records an assembly digest matching a CRLF reconstruction rather
than the committed LF-normalized blob; no assembly line-ending identity is
pinned.

## Coverage actually inspected

Both manifests cover the same five CPU sources across Windows x64 LLVM
20.1.8, 21.1.8, and 22.1.8. Version 21.1.8 matches the accepted MDSLC semantic
compiler baseline. The v1 stages are O0/O2/O3 LLVM IR and retargeted AVX2 and
AVX-512 assembly. The v2 stages are frontend-raw, SROA-normalized, final
optimized LLVM IR, vectorization remarks, and target-aware AVX2/AVX-512 LLVM IR
and assembly. The declared target triple is `x86_64-pc-windows-msvc`.

All five committed CPU input hashes match both manifests. All 195 manifest
artifact tuples match the three exported fingerprint ledgers, 65 per LLVM
version. All committed `.json` files and four of five nonempty JSONL files parse
as strict JSON. `campaign_benchmark_results.jsonl` has six records containing
the non-standard bare token `inf`; two other JSONL files are zero-byte
files. The three environment descriptors and all 30 embedded cases
validate against the supplied applicable Draft 2020 schemas.

That schema result came from an independent full Draft 2020-12 validator over
each descriptor and embedded case, not from the bundled validator's success
banner. `corpus/scripts/windows/validate-schemas.py` does not open either
manifest and does not validate array items; with the external data plane absent
it checks only the three environment descriptors. The synopsis in
`verify-corpus.ps1` therefore overstates the schema coverage it delegates to
that script.

This establishes internal control-plane consistency, not independent artifact
authentication. The external data plane declared as
`C:\Users\hamza\MDSLC-Corpus\windows-x64` was unavailable in this Linux
workspace. The raw artifact bytes, sizes, per-case descriptors, actual compiler
binaries, commands, flags, and targets therefore could not be rehashed or
replayed. Compiler executable hashes are not recorded, environment
`archive_sha256` fields contain `"audited"`, the fingerprint ledgers export
manifest values rather than independently derive them, and there is no
manifest schema.

The committed MLIR, CUDA, HIP, NVVM-oriented, and ROCDL-oriented inputs and
analysis were inspected as control-plane material. They are not manifest cases
and provide neither a fingerprinted connected lowering trace nor physical GPU
execution evidence. The main corpus generator's synopsis advertises NVPTX and
AMDGPU output, but its implementation iterates only the CPU inputs and emits
x86 artifacts. No GPU claim follows from this corpus.

## Evidence findings

The classifications below preserve the corpus evidence boundary.

| Class | Finding | Reconciliation and falsification boundary |
| --- | --- | --- |
| OBSERVED | The authenticated explicit-call pipeline retains operation, type, shape, stride, layout, memory, mutability, effect, alias, destination, and provenance contracts. Matcore IR v1 carries its representable capture semantics; the mandatory reviewed bridge context supplies the more detailed numerical profile, and verified `mdsl.gemm` carries the combined contract. | Classified `ALREADY CORRECTLY REPRESENTED`. Existing bridge/verifier negative tests would falsify this if a represented field could be silently lost or accepted malformed. |
| OBSERVED | Alias and alignment declarations are required preconditions, not proven facts; the runtime establishes overlap, actual alignment, descriptor, capability, and floating-point environment legality before output mutation. | Classified `ALREADY CORRECTLY REPRESENTED`. Overlap, misalignment, unsupported-capability, and FP-environment rejection tests are the falsifiers. |
| OBSERVED | Recovered ordinary-C++ GEMM remains sealed, provenance-bound, analysis-only, and rejected by executable lowering. `map`/`sin` composition is also inspection-only. | Classified `ALREADY CORRECTLY REPRESENTED`. Any recovered or composed module reaching runtime execution would falsify it. |
| OBSERVED | `proof/mlir_avx2/run.ps1` lowers `01_input.mlir` to one artifact, but separately translates a hand-authored `03_llvm.mlir`; `02_vectorized.mlir` is not consumed. | Classified `INSUFFICIENT EVIDENCE`. Removing or corrupting `02_vectorized.mlir` without changing the result demonstrates the disconnected chain. |
| OBSERVED | The structured proof's direct `linalg.matmul outs(%C)` reads and accumulates initial C, while Matcore explicit GEMM overwrites a write-only destination. Its hand-authored LLVM stage instead starts at zero and also maps row-major vector lanes incorrectly. | Classified `REJECT / CONTRADICTED` as a lowering template. Nonzero-C, zero-input and basis-matrix execution are the required falsifiers for any replacement. |
| OBSERVED | The committed AVX2 assembly contains a narrow spill-free packed-FMA pattern, but it is not derived by a connected, correctness-executed Matcore-to-assembly pipeline. | The machine pattern is only local feasibility evidence. It is not an executable compiler or performance claim. |
| OBSERVED | The CPU campaign evidence does not authenticate source/toolchain/environment/run identity, does not supply same-checkpoint OpenBLAS parity, and contains strict-JSON, correctness-oracle coverage, and requested-thread fidelity defects. | Classified `INSUFFICIENT EVIDENCE` for planner policy, packing thresholds, scaling, or Issue #15 closure. A complete authenticated forward/reverse envelope is the falsifier. |
| OBSERVED | The research branch's static-AOT path bypasses accepted descriptor, overlap, alignment, ISA/OS-state, numerical-policy, and FP-environment gates; its static dimensions do not round-trip coherently through the versioned capture boundary. | Classified `REJECT / CONTRADICTED`; none of those product commits are adopted. Canonical pre-mutation negatives would falsify their legality. |
| INFERRED | A structured tensor/DPS inspection stage can preserve compositional opportunity longer than immediate library dispatch, if overwrite initialization and every semantic contract are retained or consumed legally. | Classified `SHOULD INFORM NEXT MILESTONE`, not a proven implementation. A connected exact-MLIR-21 trace plus semantic negative tests is required before execution work. |
| ARCHITECTURAL IMPLICATION | Matcore should own semantic intent and legality; upstream MLIR should own canonical structured transformation machinery; LLVM/backends should own instruction selection, register allocation, and machine scheduling; authenticated libraries should remain provider candidates. | This agrees with current WHAT/HOW/MACHINE boundaries. Fixed tiles, register caps, packing policy, fusion, provider thresholds, and target routes remain evidence-dependent rather than universal rules. |

## Architecture reconciliation matrix

The implementation review followed the actual frontend, bridge, verifier,
lowerer, planner, and runtime seams in `compiler/lib/frontend/native_frontend.cpp`,
`compiler/lib/ir/matcore_ir_v1_bridge.cpp`,
`compiler/lib/mlir/MatcoreV1Bridge.h`,
`compiler/lib/mlir/MatcoreOps.cpp`,
`compiler/lib/mlir/MatcoreCpuRuntimeLowering.cpp`,
`compiler/lib/planner/cpu_planner_v3.cpp`,
`compiler/lib/runtime/cpu_runtime.cpp`, and
`compiler/lib/runtime/cpu_openblas.cpp`.

| Layer | Current state against corpus evidence | Disposition |
| --- | --- | --- |
| Native frontend | Authenticates explicit calls and preserves ordinary C++; recognition is not permission. | ALREADY CORRECTLY REPRESENTED |
| Matcore IR v1 | Typed capture/provenance DTO retains its v1-representable facts; v1-to-v0 fails closed on unsupported loss. The detailed fixed numerical profile is deliberately supplied at the mandatory bridge context rather than claimed as a complete v1 field set. | ALREADY CORRECTLY REPRESENTED |
| Matcore MLIR/verifier | Combines verified v1 capture with the reviewed bridge context and owns semantic WHAT: destination overwrite, effects, alias/alignment requirements, numerical policy, shapes/layouts, and provenance. | ALREADY CORRECTLY REPRESENTED |
| Structured transform boundary | No connected `mdsl.gemm` to Linalg/Tensor inspection route exists on canonical main. | SHOULD INFORM NEXT MILESTONE |
| CPU lowering | The canonical internal driver sends authenticated explicit GEMM to the stable runtime-dispatch lowering; recovered and composed forms fail closed. The lowerer itself does not reauthenticate a source snapshot. | ALREADY CORRECTLY REPRESENTED |
| Planner/runtime | Legality is established before packing or mutation; planning remains deterministic and evidence-bounded. | ALREADY CORRECTLY REPRESENTED |
| External providers | OpenBLAS is an explicit authenticated candidate, not a parity claim or silent fallback. | ALREADY CORRECTLY REPRESENTED |
| Packed-B storage | Caller-owned storage, lifetime, source identity, shape, packing parameters, and provenance are explicit and validated. | ALREADY CORRECTLY REPRESENTED |
| Vector/machine lowering | Canonical vector lowering, instruction selection, register allocation, and target scheduling are not Matcore semantic responsibilities. | SHOULD REMAIN UPSTREAM RESPONSIBILITY |
| Fusion, tiles, register caps, packing hoisting/reuse scheduling, activation thresholds, provider crossover | Corpus observations are incomplete, host-specific, or disconnected from a verified Matcore route. | INSUFFICIENT EVIDENCE |
| GPU/NPU targets | Inputs and analysis exist, but no manifested connected artifacts or physical execution exist. | REJECT / OUT OF SCOPE |

Two future seams were noted but do not require a current fix. If multiple
executable lowering choices are introduced, the typed lowering decision must
survive code generation rather than collapse prematurely to a site identifier.
If serialized or imported MLIR is ever accepted for execution, producer text
alone must not become execution authority; an authenticated sealed authority
boundary will be required. Both are safe under the current single internal
driver-produced execution route.

## Accepted now

Only documentation changes survived independent adversarial review:

1. Add this durable, evidence-bounded re-entry record.
2. Correct current status documents to say that PR #18 merged, post-merge
   checks passed, the checkpoint tag exists, and Issue #17 / milestone #6
   closed, while making no GitHub Release or freeze claim.
3. Correct three references to the nonexistent `mdsl.return`; the implemented
   function terminator is upstream `func.return`, while `mdsl.yield` terminates
   `mdsl.map` regions.

No source, schema, generated artifact, compiler route, planner rule, runtime
behavior, provider selection, public interface, or ABI changed.

## Deliberately rejected or deferred

- Do not cherry-pick the research branch's static dimensions, generated AVX2
  C source, direct static route, Decision Function F, fusion route, or Level-2
  routing experiments.
- Do not infer universal tile sizes, register-pressure percentages, thread
  caps, packing thresholds, or provider crossover points.
- Do not treat handwritten fusion or AVX2 proofs as Matcore transformations.
- Do not begin bufferization, vector lowering, generated execution, runtime
  autotuning, map/sine execution, recovered-loop execution, GPU/NPU work, or a
  public API/ABI/backend-contract freeze.
- Do not close Issue #15. The corpus does not provide a complete authenticated
  same-checkpoint native/OpenBLAS forward/reverse envelope, full scaling
  aggregates, full-envelope planner regret, or evidence for activating
  cooperative packed-B preparation.

## Validation

Before documentation integration, the exact canonical starting tree was
configured as Release with Clang/LLVM 21.1.8, extracted MLIR 21.1.8,
`MDSLC_ENABLE_MATCORE_MLIR=ON`, default `matcore-mlir`, and OpenBLAS disabled.

```text
cmake --build build-reentry-release --parallel 2
  PASS: 128/128 build steps

ctest --test-dir build-reentry-release --output-on-failure -j1
  PASS: 63/63 tests, 0 failed, 170.82 seconds
```

This covered the registered frontend, Matcore IR/MLIR, explicit-GEMM execution,
planner/runtime, ABI, package/install/consumer, source-inaccessible, benchmark
contract, and MLIR CLI surfaces in that profile. It does not claim an
OpenBLAS-enabled run, Windows execution, a new sanitizer run, physical GPU
execution, or performance parity in this re-entry session. Because the
accepted diff is documentation-only, post-change validation is repository
hygiene, link/reference inspection, diff checking, and independent review; the
compiled source tree is unchanged.

## Exact next development boundary

The smallest coherent next milestone is an opt-in, internal, inspection-only
verified conversion of authenticated explicit `mdsl.gemm` into structured
tensor/DPS MLIR. Before erasing the semantic operation, Matcore must encode or
verifier-authenticate destination overwrite identity, shapes/layouts, effects,
numerical permissions, target policy, and provenance, or reject conversion.
Alias and alignment requirements must remain explicit preconditions and must
not become upstream optimizer facts without static proof or a dominating guard
in a later executable phase. The structured form must explicitly initialize
the output to zero, for example with `linalg.fill`, before `linalg.matmul`;
direct accumulation into initial C is incorrect.

MLIR should own canonical tensor/Linalg construction and verification. This
milestone must stop before bufferization, vector lowering, LLVM lowering,
generated execution, planner/default changes, or performance claims. LLVM and
the backend remain responsible for instruction selection, registers, and
machine scheduling. The existing runtime/provider route remains the only
executable route and keeps authenticated external BLAS eligible for standard
large GEMM.

Entry evidence must include exact upstream MLIR 21.1.8 contract inspection and
a connected, command-recorded semantic-to-structured trace with no hand-authored
skipped stage. Completion requires deterministic textual goldens; dynamic and
static-shape coverage; nonzero-initial-C overwrite tests; basis/sentinel layout
tests; alias, alignment, effect, numerical-policy, and provenance negatives;
rejection of recovered/map/malformed inputs; package non-leakage; and unchanged
existing CPU execution regressions. This boundary is a handoff, not
authorization to execute that milestone during this reconciliation.
