# P2: strict output-tile upstream optimization trace

Research checkpoint based on canonical `243f6772ebf8cc7600d05776a2adbb8d8a0515d0`
(PR #99). This record changes no compiler, default, API, runtime or execution
authority. It follows Issue #101 and the bounded P1 evidence in PR #100.

## Verdict

**Observed:** the frozen stock loop obtains early LLVM scalar promotion of its C
accumulator. The frozen 5x3 loop retains conditional C accesses after late local-N
unrolling, and subsequent LICM leaves those recurrences in memory. Both actual
objects were reproduced byte-for-byte before diagnosis. This explains an
optimization opportunity, not the measured whole-call ratio's cause or a speedup.

P1 remains frozen and overall `INCONCLUSIVE_TIMING`. Its sole qualified stock
comparison (M/K/N 33/35/37) recorded 5x3 taking 2.45145 times stock's declared
source-call interval. No P1 sample was repeated, dropped or reclassified here.

## Controlled evidence

Original producer/source checkpoint: `06affd2ffea4a5459158a7ddcd0c4e75009ea90e`.
Its Git tree is identical to canonical PR #99 merge. Native Clang/LLVM 21.1.8,
Ubuntu revision 6ubuntu1, executable SHA-256
`412bbe8c60571a1eb06f48fde89635033621caeb01a9b4ee76d46711bae8e932`.

| Artifact | SHA-256 |
| --- | --- |
| Stock LLVM | `171063e15ecf94bffdb881f9c7fb93db412731244cf07c74610abfe969ffc540` |
| Requested 5x3 LLVM | `f367c78707e4dd67d72537bf5354a1913fdcdd320aaffa2f3d0fa77c81cca012` |
| Stock original, reproduction and diagnostic objects | `07a18b41182f864d2ddbf9332f7c476d02406290c7df4e14dec3cd4286eb95a4` |
| 5x3 original, reproduction, diagnostic and control-flow-hoisting objects | `70a5c9396f73b7351a7bc22dbcbdf276038ae27ab4225d7761535edce0560770` |

Controls reproduced each route's actual recorded compiler argv, changing only the
output path. Stock and requested source instances do not have identical compiler
argv: the source-time route explicitly binds resource directory, target, PIC and
FP-contraction settings. This study compares the actual existing routes; it is not
a claim that every machine-code difference is caused by a single isolated flag.
Diagnostic flags print pass-manager/IR/remark information but leave object bytes
unchanged. Complete commands and tool/input/output hashes are retained.

Stock's inner K loop is promoted to a float PHI and exit store before late K-by-4
unrolling. Tile's bounded local-N loop unrolls to three guarded columns later.
**LICM does run afterward**, with promotion enabled; it hoists C address
computations but reports conditional-load refusals and retains three per-K
load/add/store sequences. The native trace and actual called C-interface assembly
agree. Tail columns may not exist: disjointness alone does not authorize new
speculative accesses.

## Rejected explanations and shortcuts

- **Missing C noalias — contradicted.** Both LLVM signatures already annotate
  aligned-C argument 15. C is freshly owned while A/B remain live; replacement of
  an output handle happens after execution. A/B may be the same Value. Allocated-C
  is descriptor bookkeeping, not another accessed matrix root. Public Storage
  views may overlap; no input/descriptor noalias or nonnull fact is justified.
- **No LICM after late unrolling — contradicted.** The actual pass trace and
  [pinned upstream pipeline](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Passes/PassBuilderPipelines.cpp)
  show it running. The [pinned LICM implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Transforms/Scalar/LICM.cpp)
  checks safe access/store insertion, not merely alias freedom.
- **Enable control-flow hoisting — negative.** The compile-only native option
  produced exactly the same object, conditional-load refusals and memory
  recurrences. Its default-off setting is observed; no unsupported explanation of
  why upstream chose that default is asserted.
- **Enable nontrivial unswitching — insufficient.** A compile-only counterfactual
  promotes column 0 only; columns 1/2 remain conditional memory recurrences. It
  also changes inlining (reported cost 230 exceeds threshold 225), so it is not
  an isolated performance ablation. Its object was neither linked nor executed.
  Neither hidden flag is recommended as a product setting.

## Exactly one compile-first proposal, not an adopted design

Test upstream outer-N full-tile/tail peeling in an experimental closed **5x3** HOW
derivation before scalar lowering. The frozen scheduled payload contains outer-N
step 3 and `affine.min(N-j,3)`. Pinned
[Transform loop.peel](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/SCF/TransformOps/SCFTransformOps.cpp)
calls [peelForLoopAndSimplifyBounds](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/Dialect/SCF/Transforms/LoopSpecialization.cpp),
which simplifies affine min/max in the main/remainder loops. That is a concrete
mechanism to investigate, not a promise it will produce suitable machine code.

Falsification gate: full N=3 bodies must lose per-K tail decisions and unchanged
native O2 must produce **three scalar accumulator PHIs / exit stores**. The
remainder must retain exact bounded coverage; K/order/separate f32 arithmetic,
zero/empty behavior, private storage and original witnesses must remain intact.
Simply producing two loops or compiling successfully is insufficient.

The current private verifier recognizes an unpeeled loop nest; it does **not**
accept this proposed shape. Experimental structural/coverage proof is needed
before any production issuer change; execution and new performance evidence
would require their own reviewed gates. No new semantic permission, pointer
assertion, private accumulator optimizer, schedule sweep or default change is
authorized by this diagnosis.

## Independent review and retained evidence

An independent lane rechecked all 34 recorded stdout/stderr files, every listed
identity, the baseline/object joins, exact relevant pass IR, native refusal
remarks, counterfactual inlining and the pinned peeling interface. No substantive
blocker was found for the compile-only mechanism claims or the falsifiable
proposal. It did not run another compiler or benchmark. Seven serial compiler
processes (including one hidden-help query) returned zero; recorded process time
was 0.395062107 seconds and retained regular files 2172687 bytes. This is resource
accounting, not a performance measurement.

Raw directory: `/var/tmp/mdslc-p2-codegen-UUAnuP/`.
`REPORT.md` SHA-256:
`e9cadef4e20da69a0bfef40de099349fc7e17909434c95f0a595d1e8bf0da21a`.
`commands.json` SHA-256:
`690396c26bf529f55afc68bf595341e340b1102ab219c73a6e419d9f2fb49257`.
Ignored local archival copy:
`benchmark_reports/strict-tile-codegen-2026-11-10-v1/trace.tar.gz`, SHA-256
`535919508d41b130dbcab0af0ef75eff400b315887c3f93cafc79fafaff9f148`
(about 296 KiB). This is evidence retention, not an executable import or portable
runtime bundle. The archive's report bytes were checked against the original.

Precision erratum to the raw recorder's README: it removed `LD_PRELOAD` and
`LD_LIBRARY_PATH`, not every loader/environment control. Exact baseline/diagnostic
object equality supports the narrow inspection; this is not hostile-host
attestation or an independently qualified clean-environment performance campaign.
Archived original receipts were not rewritten to hide that limitation.
