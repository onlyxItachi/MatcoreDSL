# Source-time output pattern integration evidence v1

Status: work in progress. This record distinguishes baseline checks from new
implementation qualification; pending work is not a passing result.

## Authenticated start (2026-10-12)

- Local canonical `main`, freshly fetched `origin/main`, and live GitHub main:
  `3077d9c3f5df1479f2fcadfcceeb9444018d6ba0`, clean checkout.
- PR91 merged engineering `c09a9d0e4384441238bf169e7e1122bf13ea5e23` from
  `4fde88b8c3e6b09bc88ce466a0bd28d9dafe6600`; all 26 original hosted checks
  still report success. PR92 is the subsequent checkpoint documentation.
- Canonical post-merge runs at `3077d9c`: native `38095151945`, Windows
  `38095151897`, legacy `38095151947`, hygiene `38095151904`, all success.
- SPIR-V PR84 remains open/draft at `5dc14db686b6b2668a2d3a47f5ef24c0e2bb4b97`;
  Issue89 remains open. They are not this task's integration base or dependency.
  Issues15 and20 remain open, with their prior scope unchanged.
- New integration worktree `mdslc-work/source-pattern-v1/integration`, branch
  `mdslc/source-time-output-pattern-v1`; independent core/test worktrees share
  the exact canonical starting commit. Existing worktrees are not reused or cleaned.
- Main agent read complete AGENTS, CURRENT_STATE, pattern contract and actual
  driver/compiler/issuer/runtime/package code. Independent lanes challenged
  ownership, designed implementation, and authored falsification fixtures.

## Baseline local control, not new-feature qualification

Fresh Release build at `mdslc-work/source-pattern-v1/build-release`, coherent
Clang/LLVM/MLIR 21.1.8, native + bootstrap + experimental regions, scalar default,
authenticated OpenBLAS0.3.32 required, GPU issuers compiled but real GPU candidate
execution disabled. LLVM/Clang are from `/usr/lib/llvm-21`; MLIR comes from the
existing matching private `mlir-21.1.8-6ubuntu1` package prefix. No package/toolchain
mutation. Build used two jobs with serialized heavy links and completed all523
steps. The integration head at this point differs from main only by design docs.

Baseline focused control passed **35/35**, zero skips,41.69 seconds: all32 prior
output-pattern controls, private DSO symbol contract, source driver contract,
and implementation-ownership adversaries. JUnit `baseline-focused.xml` SHA256:
`782892441d391a7aca4f5f20544b593618ff801763266d530de957a621e263c1`.

Retained local artifact identities before implementation integration:

| Artifact relative to build | SHA256 |
| --- | --- |
| `lib/libmatcore_closed_candidates_isolated_v1.so` | `ac3f754b0306e23008730e228275546fed59fb823772685ff70b1b0eb5511774` |
| `lib/libmatcore_runtime.so` | `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014` |
| `lib/regions/strict-normal.o` | `07a18b41182f864d2ddbf9332f7c476d02406290c7df4e14dec3cd4286eb95a4` |
| `lib/regions/strict-normal.ll` | `171063e15ecf94bffdb881f9c7fb93db412731244cf07c74610abfe969ffc540` |

These are same-environment regression controls, not portable release digests or
proof of performance. New-source-path qualification remains pending below.

## Incremental implementation qualification

At implementation/test head `01362399b31cddb7814e928ed72b8090978564a2`,
Release source-pattern scope passed **4/4**, zero skips,135.64 seconds:
`driver.source_pattern.connected`, `refusals`, `runner`, `phase_mutations`.
The independent connected oracle executed **68 cases / 111292 checks / zero
failures for each of 3x5,5x3,1x1,64x64**. It also exercises rhs-carried and
same-Value operands, eligible forwarding versus different-descriptor MAY-alias
reads, no-request stock routes, artifact identity and no-clobber publication.
Earlier integrated original-pattern/ownership regression scope passed35/35,
zero skips,45.01 seconds. These are incremental checks, not yet complete-suite
or hosted qualification.

The original runtime, stock candidate DSO, default leaf object and LLVM retain
all four baseline hashes above after integration. Manual original two-GEMM
source requests3x5 and5x3 both compiled and executed; the ASan3x5 control also
executed. Those small controls do not replace the independent full runner.

Useful negative results preserved:

- Review found that a source-bound DSO could be cross-wired into compilation of
  another authenticated source. The compiler now checks the exact original
  source/entry seal, policy, optimization, bytes and worker/sanitizer profile.
- Review then found correctly compiled A could be passed to a raw final linker
  with instanceB. Typed publication now checks the immutable exact instance,
  produced LLVM and frozen input snapshots, original Installation owner and
  output/bundle association before native linkage. No generic IR importer or
  new public authority was added.
- Initial source-ASan64x64 refused `__asan_report_load_n`. The existing issuer's
  exact diagnostic object demonstrated that one additional compiler-rt reporting
  import; the sanitized-only closed list was corrected, not replaced by a prefix
  allowance. The refusal produced no executable or retained failed bundle.
- The first connected runner incorrectly demanded forwarding of a read through
  descriptorD following publication through descriptorC. MAY-alias is not
  identity, so this was a test defect. The corrected test preserves that case
  and adds a genuinely eligible same-C read with checked shape-failure frontiers
  and none/forwarding call discriminators. Production forwarding was unchanged.

Complete regression, sanitizer, relocated production-package and hosted results
remain to be appended before engineering qualification.

The matching Debug/ASan+UBSan source-pattern scope at the same implementation
head also passed **4/4**, zero skips,154.13 seconds. Both Release and sanitizer
phase tests executed69 checks with zero failures, including successful correctly
paired typed publication after refusal controls. The sanitizer connected runner
required heap-buffer-overflow attribution to the actual requested DSO's local
generated leaf (not a separately recompiled copy), plus original-host ASan evidence.
This also exercised1x1 and64x64 source requests after the observed import fix.

Initial focused JUnit identities (retained outside Git):

| Profile | SHA256 of `source-pattern-qualified.xml` |
| --- | --- |
| Release | `210dc7385a5e3463fe9b5ff839681e4681455663aff9c1d0579ccf5e8f25e541` |
| Debug ASan+UBSan | `5ebb63ea69c8c148dae0868a76a8002047b3097ac6c02dc6539c952199ec6bb6` |

All28 retained NVVMsm89/ROCDLgfx1150 single and fused-pair issuer artifacts are
byte-identical between the canonical baseline and integrated build. This is
compile-artifact regression evidence only, not new physical GPU qualification.
The [independent static review](source-pattern-adversarial-review-v1.md) accepts
the exact core and connected test design while explicitly separating its static
verdict from these integration-owned execution results.
