# P5: authenticated, untimed native preparation

Status: **PREPARED; no performance collection**. This is a qualification record,
not a speedup, schedule selection or Issue #15 completion claim. The fixed
[protocol](../../../compiler/benchmarks/source_pattern_peel/README.md) is unchanged.

## Identity and scope

- Harness source: `b63ed32028955594708985519c3d59cedbad7652`. The subsequent
  normal merge `3a351cba3123f9db3347fd42517750c0da2b0970` adds only PR105's
  Release CI budget correction/report; all pinned harness/source bytes match.
- Clean detached compiler producer: `1ad7fe56b5fcb10efe37e523206228e9ef303f6a`.
  Its existing Release installation is relocated under
  `build-source-peel-release/tests/installed-consumer/relocated/matcoredsl prefix ünicode`.
  The source and build stay frozen. PR105's CI-only follow-up does not relabel
  these locally executed compiler bytes as a new build.
- Actual Clang/LLVM/MLIR: 21.1.8, baseline Linux x64, uninstrumented. The
  source driver uses `/usr/bin/x86_64-linux-gnu-ld.bfd`, not the different linker
  named by the producer's general CMake cache. The receipt pins actual tool bytes.
- Capsule: `/var/tmp/mdslc-p5-campaign-FhMFbP/prepared-v1`.
  `prepared.json` SHA256:
  `add6dfba5ea56d46a743479194bb289f631d780bc85dfa59660a6862e5f8afa6`.
- Plan SHA256: `2933f108ba13bb7e7bd417e8f53efe416650d22506b0a7e2e514278e94b42d85`.
  One rendered source SHA256:
  `61863394d768c2006ff11a554e9deea9482756be778ac45e9860f13f5882946e`.

Preparation completed 130 recorded commands / 68 ordered events. It produced
12 source executables and ten request-owned private bundles, using the same SDK
for stock, old 3x5/5x3 and peeled 3x5/5x3 roles. Generated binary/bundle/source/
inspection bytes: 2,563,470; raw log bytes: 638,735. Prerequisite SDK/toolchain
bytes are separately inventoried, not miscounted as new artifacts.

## Actual correctness and identity evidence

- All five timed-program binaries ran **only `--verify-only`**, each passing
  469 checks over the fixed numerical, alias, failure, observation and FP-state
  controls. No `--measure` invocation occurred.
- All five independent connected-source controls passed 68 cases / 111,292
  checks each, zero failures.
- Peeled 3x5 end-K control: 792 cases / 206,372 checks; peeled 5x3: 792 cases /
  196,420 checks. Each reports 728 full-domain hits, 364 width-one and 364
  width-two tail hits; these counters overlap. Each excludes 40 inapplicable-K
  combinations rather than counting them as passed numerical cases.
- All eight primitive artifact hashes join each requested timed binary to its
  connected/end-K controls. Original WHAT stages agree; distinct requested HOW,
  source binding and private DSO identities are actually consumed.
- The stock role's scalar-mnk profile is reconstructed from the frozen producer,
  exact issuer/object/link commands and the installed DSO byte join after standard
  CMake RPATH removal on a nonexecuted inspection copy. Stock is not inferred
  from its role label, nor borrowed from the older P1 installation.
- Root offline `audit_prepared` passed against all retained bytes and raw command
  records without launching another compiler or program. A separate reviewer
  repeated the complete audit with process creation blocked, checked 48 explicit
  primitive-field comparisons, then rejected all 32 top-receipt-only in-memory
  mutations at their expected boundaries. Those mutations covered source/helper
  identity, realization/request/schema/tiles, tool/header/provider substitution,
  stock build/issuer/link/RPATH metadata, codegen, raw chronology/argv/timeouts,
  correctness/end-K, budgets and a forged performance claim. A final positive
  audit passed, with zero subprocess launches and all capsule hashes unchanged.
  External audit: `/var/tmp/mdslc-p5-prepared-audit-DX8uCW/audit.json`, SHA256
  `50a35d06c348c4050fbb5bb857f20749c828ee9c6cc5c85aae03a84b91a9c9a0`.
  These trusted-collector integrity checks are not hostile-origin attestation.

## Actual code-generation observations, not performance

Sizes below are bytes of the two local functions in each actual owning DSO,
not research specimens, whole executable sizes or timing measurements.

| Role | GEMM leaf | ABI wrapper |
| --- | ---: | ---: |
| stock | 724 | 769 |
| old 3x5 | 826 | 843 |
| peeled 3x5 | 2500 | 78 |
| old 5x3 | 778 | 763 |
| peeled 5x3 | 1717 | 78 |

In retained disassembly `raw/0076/stdout.log`, old 5x3 updates C memory inside
the conditional K loop. `raw/0096/stdout.log` shows peeled 5x3's full-N block
holding one two-lane output accumulator plus one scalar accumulator
through K, with separate multiply/add instructions and stores after that loop.
The wrapper/inlining and code-size differences are simultaneous changes. These
observations support actual consumption of the proposed realization, not isolated
causal attribution or a speed prediction. The fixed source-call experiment must
still be collected after product/CI/review qualification; no timings are reported
here and no default, runtime/provider policy or numerical permission changes.
