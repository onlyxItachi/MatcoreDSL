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
