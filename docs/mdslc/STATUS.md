# MDSLC status

Status date: 2026-08-11

- Canonical `origin/main` and merged Milestone 7 bounded disposition:
  `e5069758ad04bdb459de2026cad8498b47fda707`
- Strategic semantic-foundation branch:
  `mdslc/semantic-compiler-foundation-v1`
- Milestone 6 pull request: `#14`, merged normally
- Milestone 6 umbrella issue / GitHub milestone: `#13` / `#4`, closed
- Milestone 6 immutable tag: `mdslc-cpu-performance-audit-v1`
- Milestone 7 integration branch: `mdslc/native-blas-parity-v1`
- Milestone 7 pull request: `#16`, merged normally with all hosted Linux,
  Windows x64, and repository-hygiene checks passing
- Milestone 7 umbrella issue / GitHub milestone: `#15` / `#5`, open
- Completed Milestone 5 merge before history sanitation:
  `091d74072a710389b4a8e9d51f696ad9773021e6`
- Focused Windows validation branch: `mdslc/windows-x64-v1`
- Milestone 5 GitHub milestone / umbrella issue: `#3` / `#9`
- Milestone 5 pull request: `#11`, merged normally
- Focused Windows validation pull request: `#12`, hosted checks and
  independent review passed at `216c81210e2dcbc4599b384e99ceb90a91aab4ba`
- Milestone 5 immutable tag: `mdslc-cpu-backend-v2`
- Milestone 4 pull request: `#10`, merged normally
- Milestone 4 GitHub milestone/issue: `#2` / `#8`, closed
- Milestone 4 immutable tag: `mdslc-cpu-performance-foundation-v1`
- Milestone 1 rewritten checkpoint tag: `mdslc-native-cpu-proof-v1`
- Milestone 2 is preserved in rewritten mainline checkpoint tag:
  `mdslc-mainline-cpu-proof-v2`
- Milestone 3 mainline pull request: `#6`, merged normally into `main`; the
  later controlled history sanitation preserved its source tree while
  remapping commit IDs.
- Milestone 3 tracker: GitHub milestone `#1`, completed

## Strategic semantic-compiler pivot

**The primary next objective is a compositional Matcore semantic compiler and
CPU-first beta. It is not a parity-only continuation and it does not begin a
public API/ABI/backend-contract freeze.**

The existing Matcore IR v1 remains the deterministic typed capture and
provenance DTO. The accepted next optimizer representation is a Matcore MLIR
dialect with textual namespace `mdsl`, generic SSA values, verified operation
semantics, regions/use-def relationships, numerical intent, domains, effects,
aliases, mutation, and source provenance. The v1-to-dialect bridge must be
exact for every represented v1 field and fail closed otherwise. Because v1
does not yet contain sufficient numerical-policy granularity, ADR-0009 freezes
the internal `explicit-gemm-f32-v1` policy and Milestone B must encode and
verify it. The policy permits FMA and reassociation only within a GEMM K
reduction, leaves K order implementation-defined, preserves NaN/non-finite
semantics without payload/order guarantees, relaxes signed zero, forbids
approximate math, overwrites an explicit non-aliasing destination, and forbids
in-place operand mutation. It additionally requires round-to-nearest-ties-even,
masked/non-trapping exceptions, gradual subnormals with FTZ/DAZ disabled, and
makes no exact exception-status-flag preservation guarantee. Recovered loops
require source-derived proof and cannot inherit these permissions. No bridge is
called fully lossless until those semantics are represented and verified. No
second JSON optimizer schema is planned.

Responsibilities are now explicit:

```text
WHAT     authenticated capture + Matcore semantic MLIR
HOW      planning/transformation plus Linalg/Tensor/MemRef/Vector substrates
MACHINE  LLVM or target-specific dialects and platform/vendor toolchains
```

Lowering may consume a semantic fact only after the next representation
structurally encodes it or every optimization needing it has completed.
Linalg/Tensor/MemRef/Vector remain HOW while alternatives are still possible;
being an upstream dialect does not itself make a representation MACHINE-level.
Recognition of an ordinary C++ idiom is not permission to replace it. Failed
or unproven implicit raising preserves ordinary C++ behavior.

The `mdsl.gemm` result is defined as the post-overwrite semantic value tied to
the explicit write-only destination. It is not an independent allocation;
bufferization must alias it to destination storage, and the observable write
cannot be removed because the SSA result is unused. V1 alignment and alias
requirements are preconditions rather than facts. An optimization may consume
them only after static proof or a dominating runtime guard, with dynamic
rejection before packing or output mutation.

The Linux x86-64 CPU runtime now validates the execution thread's rounding
mode, trap state, and FTZ/DAZ state before packing or destination mutation. It
also admits every active native worker before parallel work and validates the
linked single-thread OpenBLAS adapter against the supported numerical
envelope. The exact source-evaluation compile profile is enforced. Rejection
uses the additive status
`MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0` (numeric value 26)
without changing an existing C record layout or function signature. Focused
normal and ASan/UBSan review passed, and the later full local CPU-beta matrix
passed at `69d099e`. Windows FP support remains bounded by its hosted
compatibility lane rather than this physical Linux acceptance.

The first staged work is architecture freeze, the MLIR core and deterministic
v1 bridge, GEMM-to-SIN domain composition, conservative explicit/recovered
GEMM equivalence, and an end-to-end CPU MLIR route that reuses the validated
planner/runtime. The full dependency graph and merge gates are in
[ADR-0009](../adr/0009-mdslc-semantic-compiler-foundation.md) and the
[roadmap](ROADMAP.md).

### Toolchain gate

The installed standalone frontend tuple remains Clang/LLVM 21.1.8, Ubuntu
revision `1:21.1.8-6ubuntu1`. A system APT simulation showed that installing
matching MLIR 21 development packages globally would remove the installed
MLIR 22 surface. The exact `libmlir-21`, `libmlir-21-dev`, and
`mlir-21-tools` 21.1.8 Debian payloads were instead extracted to a versioned
user-local development prefix without changing system packages.

The toolchain lane validated an in-process `clang-cpp` + MLIR + LLVM program,
a TableGen-generated toy dialect/op, and a narrow static
`MLIRIR`/`MLIRSupport` executable with neither a shared `libMLIR` dependency
nor a local-prefix RUNPATH. Production configuration may accept the prefix via
`MLIR_DIR`, but no installed artifact or package export may hardcode it. The
frontend must not mix MLIR 22 or modify the legacy root MLIR 18.1.3 contract.
The semantic-foundation implementation may advance only while this coherent
isolated toolchain and its no-path-leak gate remain green.

### Semantic milestones A through H

**Milestone A is complete on `mdslc/semantic-compiler-foundation-v1`.**
ADR-0009, this status, the roadmap, the pre-freeze decision log, and the
repository agent guidance agree on the WHAT/HOW/MACHINE boundary, destination
identity, precondition-versus-fact rule, numerical environment, recognition
permission, and execution-intent boundaries. Independent review reported no
unresolved high- or medium-severity finding.

**Milestone B is implemented and independently accepted.** The opt-in
`MDSLC_ENABLE_MATCORE_MLIR` build now provides a TableGen-generated `mdsl`
dialect, destination-aware and effectful `mdsl.gemm`, a deterministic verified
Matcore IR v1 bridge, and the internal `matcore-mlir` inspection tool. Every
site is emitted as an independent semantic entry symbol so ordinary host C++
control flow is not fabricated; that public MLIR symbol visibility is an
internal liveness root, not a public C/C++ ABI promise. The installed tool links
only the required static MLIR components and does not export MLIR or the
user-local development prefix through package metadata or RUNPATH.

The dialect verifier distinguishes three exact origin/numerical contracts:

- authenticated explicit calls use `explicit-gemm-f32-v1`;
- recovered loops whose effective source semantics match the relaxed profile
  use `source_proven_guard_required`; and
- ordinary strict increasing-K loops use the analysis-only
  `recognized_rewrite_rejected` contract.

The strict form cannot enter the explicit v1 bridge envelope, and the relaxed
form does not claim that its runtime guard has already executed. Focused
evidence passes 204/204 semantic checks, 9/9 CLI checks, and 2/2 MLIR CTest
entries. Independent review found zero unresolved high/medium findings. The
last in-flight whole-suite run passed 51/52 after the branch advanced during
execution; the sole failure was the expected authenticated source-commit guard
rejecting a stale benchmark binary. Later focused installed-profile validation
refreshed exact clean-head provenance and passed its owned gates. The later full
local matrix rebuilt the settled code candidate `69d099e` and passed every
registered supported gate, including provenance-sensitive tests.

**Milestone C is implemented and independently accepted for the internal
composition-v1 optimizer boundary.** `mdsl.map`, `mdsl.sin`, `mdsl.yield`,
`mdsl.return`, and the closed `all`, slice, indices, and predicate domain forms
have deterministic verified semantics and source-backed provenance checks.
This is an inspection/optimizer model only: the v1 CPU lowerer rejects every
map/domain pipeline, and no public map operation or map/sine execution route is
claimed.

**Milestone D is implemented and independently accepted for analysis and
equivalence inspection.** One canonical ordinary-C++ row-major F32 GEMM loop
can be recognized, sealed against an authenticated source snapshot, and
compared with an independently authenticated explicit `mdsl::gemm` through the
common mathematical fingerprint. Recognition is not rewrite permission.
Strict and guard-required recovered forms remain analysis-only, the executable
CPU lowerer rejects them, and rejected recognition preserves ordinary C++.

**Milestone E is implemented and independently accepted for the focused Linux
explicit-GEMM path.** Authenticated native Matcore IR v1 is bridged into a
verified Matcore MLIR module, lowered to a private CPU runtime-dispatch backend,
compiled into an ordinary object/executable, and executed through the stable
`matcore_runtime_gemm_f32_v0` boundary. The executed backend is produced by the
semantic lowering rather than by an unused inspection sidecar. This is a
library-dispatch lowering, not Linalg/Vector loop generation. The fresh local
Release, Debug, sanitizer, installed-package, relocation,
source-inaccessible, and ABI matrix passed at `69d099e`; hosted Linux/Windows,
normal merge/tag, and final independent review remain Milestone H work.

**Milestone F has an accepted bounded technical-limit disposition.** The
unchanged Milestone 7 performance contract still lacks a complete authenticated
forward/reverse pair, so native-BLAS parity remains unproven, Issue #15 and
milestone #5 remain open, and no parity tag exists. CPU beta may select the
fastest legal authenticated provider, including OpenBLAS, without turning that
selection into a native-parity claim.

**Milestone G is independently accepted for the bounded existing-version
contract.** Packed-B v1 remains caller-owned, serial, synchronous borrowed
storage with manual invalidation; returned C strings remain borrowed; existing
versions evolve additively. General transformed-operand ownership, structured
report iteration, execution intent, forced-variant policy, and support-duration
decisions remain inputs to the later public freeze. This acceptance does not
freeze any API, ABI, or backend contract.

**Milestone H is active.** The compatibility source-tree configuration remains
`MDSLC_ENABLE_MATCORE_MLIR=OFF` with default semantic pipeline `capture-v0`.
The Linux CPU-beta profile deliberately enables exact MLIR 21.1.8 support and
sets the configured default to `matcore-mlir`. Installed packages publish
`MatcoreDSL_MATCORE_MLIR_AVAILABLE` and
`MatcoreDSL_DEFAULT_SEMANTIC_PIPELINE`; `matcoredsl_add_executable` accepts an
explicit `SEMANTIC_PIPELINE capture-v0|matcore-mlir` and fails closed on
invalid or unavailable combinations. Windows Release, Debug, and supported
sanitizer profiles remain explicitly MLIR-disabled/default-`capture-v0`; they
must prove the semantic route unavailable rather than imply Windows MLIR
execution.

The full local Milestone H matrix passed at code candidate `69d099e`: the
Release MLIR/OpenBLAS 2x2 matrix, MLIR-present/default-capture compatibility,
full Debug with OpenBLAS, focused ASan+UBSan and TSan scopes, installed and
source-inaccessible packages, strict C17 ABI, legacy frontend, artifact,
planner-sanity, and repository-hygiene gates are green. Commits after that
candidate only record this evidence or correct review-document whitespace;
they do not change compiler/runtime/package behavior. Hosted pull-request and
Windows results, normal merge, and the final independent CPU-beta review remain
pending. No CPU beta release is claimed yet. The bounded product contract and
exact remaining gates are in [CPU_BETA_V1.md](CPU_BETA_V1.md).

## Milestone 7 native BLAS parity

**Milestone 7 has a reviewed manual partial disposition merged through PR #16.
The non-performance implementation, correctness, artifact, sanitizer, ABI,
package, consumer, Linux hosted, Windows x64 hosted, and hygiene gates passed,
but performance acceptance was not evaluated because the declared
forward/reverse evidence pair is incomplete. Issue #15 and milestone #5 remain
open; no completion tag exists.**

The private CPU backend now has a wider AVX-512 4x32 full-tile body,
two-dimensional output-task planning, exact task-capacity diagnostics, and
cooperative packed-B preparation infrastructure. Independent review rejected
the broad production activation rule because only one retained shape had
bounded performance evidence, so cooperative preparation is dormant pending a
predeclared final-checkpoint boundary matrix. The ordinary serial B-preparation
path remains active. No hidden allocation, cross-call cache, K split, or public
ABI change was added.

Four bounded cell-median point estimates at cooperative-packing checkpoint
`4719528354575f5aff74def97b534e763cb2033c` favored the candidate by
1.715--1.879x for AVX2 and 1.720--1.809x for AVX-512 versus intra-Milestone-7
checkpoint `6a26994849aadf738910e18a0cebb66ea9b238dc`. They are not direct
Milestone 5 or final-code-checkpoint evidence. An intermediate `a008a57`
diagnostic measured 2.14x AVX2 and 1.82x AVX-512 four-thread speedup, but later
runtime changes mean final-code-checkpoint scaling remains unestablished. A
proposed serial AVX2 full-tile routing change was 0.62--2.47% slower on stable
complete calls and was reverted rather than counted as progress.

The schema-v6/manifest-v3 parity runner and deterministic summarizer
authenticate Git source, tracked runner and summarizer bytes, benchmark binary,
plan, raw digests, timing scopes, results, exact actual threads, placement,
correctness, and forward/reverse coverage. The full physical sweep could not
complete on the shared host. An external, untracked exact-checkpoint forward
receipt with SHA-256
`26e75ecbcfbb19d024fa8a5fa9790b65a2deb5743b39f16a4f22dd39381cfe69`
records 258/368 cases, but no complete forward/reverse pair exists and the
summarizer emitted no performance verdict. Full native/OpenBLAS family ratios
and full-envelope regret are therefore unestablished.

Exact local validation at final code checkpoint `ff483af` passed
Release 50/50, Debug 50/50, OpenBLAS-disabled 50/50, ASan/UBSan 19/19, TSan
4/4, package/consumer 4/4, ISA artifacts 3/3, the native
`.mdsl -> ELF .o -> executable` proof, repository hygiene, and the 14-case
legacy frontend contract. The OpenBLAS-disabled forced request failed closed.
Independent static review found no unresolved high- or medium-severity code
finding. PR #16 subsequently passed the OpenBLAS-required and
OpenBLAS-disabled Linux jobs, generic build/test, Windows x64, and repository
hygiene jobs before its normal merge. The partial performance disposition did
not become a parity claim through that merge.

The merged bounded disposition is documented in
`docs/performance/cpu/native-blas-parity-v1.md` and
`docs/mdslc/agent-reports/m7-integration-validation.md`. Milestone 7 must not
be called complete, issue #15/milestone #5 must not be closed, and no parity
completion tag may be created unless a later exclusive-host run satisfies the
declared contract.

## Milestone 6 CPU performance deep audit

**Milestone 6 is published and complete for the bounded audit scope. PR #14
passed hosted checks, merged normally into `main` at `ddda3cc`, issue #13 and
milestone #4 are closed, and the immutable
`mdslc-cpu-performance-audit-v1` tag anchors the merge.**

The final schema-v6 collection is authenticated by immutable external
manifests. The forward order contains 711 executable cases: 583 accepted raw
reports, 128 exact legality rejections, and 58 predeclared runtime-bound
skips. Its manifest SHA-256 is
`b3f872bd0085b15a8cd0cfcc7663af2a41f445355a3e3237c979dc52618362c0`.
The paired complete/one-shot reverse control contains 539 cases: 429 accepted
reports, 110 exact legality rejections, and 49 skips. Its manifest SHA-256 is
`4939c0c77586e4115dfe5c1aab1ff044d716e9a5d060c9f2ef52f265634df7f8`.
The fail-closed summarizer reconstructs the frozen case matrix and
authenticates measurement configuration, commands, raw hashes, expected
rejections, and selected variants.

On the declared Ryzen AI 9 HX 370 host, single-thread fastest-native/OpenBLAS
median throughput ratios were 0.868 for medium square, 0.849 for large square,
0.884 for short-wide, 0.843 for tail-heavy, and 0.795 for tall-skinny shapes.
Vector-like shapes favored existing native paths, but that is not a general
GEMM parity result. The weakest declared cells were the tall-skinny
`4096x64x4096` and `8192x32x1024` cases at approximately 0.75. Complete
comparable-placement planner regret was median 1.000, p95 1.159, and maximum
1.213. These are host/provider-specific audit measurements.

The ranked findings identify the narrow microkernel/blocking implementation
space, transient packing for vector-like calls, M-only 128-row parallel
tasking, serial full-B preparation, and missing task-wave/placement detail as
the most actionable Milestone 7 experiment areas. No production planner,
kernel selection, runtime ABI, or public API behavior changed in Milestone 6.

Fresh Release, Debug, and OpenBLAS-disabled configurations each completed
48/48 CTest tests successfully. The supported ASan/UBSan set passed 17/17,
TSan shared-state coverage passed 4/4, the source-inaccessible installed
consumer and safety gates passed, the native `.mdsl -> ELF .o -> executable`
proof printed `MDSLC CPU GEMM PASS`, and the legacy frontend contract passed
14/14 checks. The installed-prefix absolute-path scan, repository hygiene, and
`git diff --check` passed.

Physical hardware counters were unavailable because
`kernel.perf_event_paranoid=4`. Multi-thread OpenBLAS concurrency and placement
were not authenticated and remain diagnostic-only. The audit therefore makes
no hardware-counter causal-share claim and no native-BLAS-parity claim.
Evidence and the accepted independent review are recorded in
`docs/performance/HPC_KERNEL_ENGINEERING_HANDBOOK.md`,
`docs/performance/cpu/cpu-performance-deep-audit-v1.md`, and
`docs/mdslc/agent-reports/m6-final-adversarial-review.md`.

## Milestone 5 advanced CPU backend

**Milestone 5 is published for the validated Linux host scope. PR #11 passed
hosted checks, merged normally into `main` at `091d740`, and is anchored by the
immutable `mdslc-cpu-backend-v2` tag. The separately bounded Windows x64
compatibility candidate on PR #12 passed its hosted compiler, runtime,
package, sanitizer, artifact, and independent-review gates.**

Planner v3 evaluates the five Milestone 4 candidates plus three advanced native
implementations, for eight stable F32 variants:

```text
cpu.reference.f32.v1
cpu.tiled.f32.v1
cpu.compiler-vectorized.avx2-fma.f32.v1
cpu.external.openblas.f32.v1
cpu.native-packed.avx2-fma.f32.v1
cpu.native-packed.avx512-fma.f32.v1
cpu.native-parallel.avx2-fma.f32.v1
cpu.native-parallel.avx512-fma.f32.v1
```

Capability record v2 separates hardware, OS architectural state, compiler
support, compiled implementation, and physical runtime-validation evidence.
Legality is authenticated against the exact worker placement rather than the
calling thread alone. On the declared Ryzen AI 9 HX 370 host, packed and
parallel AVX2/FMA and AVX-512F/FMA F32 paths executed successfully. The
isolated AVX2 and AVX-512 microkernels contain the required YMM and ZMM packed
FMA instructions; no global AVX2 or AVX-512 compilation requirement was added.

The additive opaque C execution context creates persistent workers once,
reports requested and actual threads, fixes affinity/SMT/NUMA policy, assigns
deterministic row bands, and uses caller-owned shared and per-worker workspace.
Native workers and OpenBLAS execution are mutually exclusive, so the runtime
does not nest provider and native pools. Internal C++ `shutdown()` is
repeat-safe; the public C handle remains a consume-once destroy contract. The
installed runtime exports exactly 15 public C functions and preserves all
earlier symbols.

Topology v1 reports logical processors, physical cores, packages, cache-sharing
groups, CPU-to-node membership, and discovery completeness. Compact/scatter
affinity and allowed-processor restrictions are implemented. Physical
validation covers this host's one NUMA node; multi-node planning is
synthetic-only. No page allocation, binding, migration, or interleaving is
performed or claimed.

Typed BF16-to-F32 and I8-to-I32 GEMM reference semantics, independent oracles,
IR verification, and additive C entry points are implemented. There is no
optimized AVX-512 BF16, AVX-512 VNNI, AMX-BF16, or AMX-INT8 variant. This host
does not expose AMX, and no AMX runtime claim is made.

`matcore-bench` schema v4 authenticates the exact build-time commit and tracked
worktree state, records exact execution context and placement, and balances
complete-call planner-regret measurements in forward and reverse registry
order. The final guarded 20-shape compact calibration at source checkpoint
`5f634aef2a0b47cd033df77c40d709456603b405` measured median regret 1.005,
nearest-rank p95 1.047, and maximum 1.363; no shape exceeded 1.5 or 2.0. These
are host-specific measurements, not universal performance claims. Raw results
remain outside Git; the reviewed sanitized summary is
`docs/performance/cpu/milestone-5-advanced-cpu-2026-07-22.md`.

The independent Linux acceptance matrix passed fresh Release 42/42, Debug
42/42, ASan/UBSan 31/31, TSan 3/3, and OpenBLAS-disabled 37/37, plus package,
C17 ABI, generated-object, installed-consumer, hygiene, and legacy frontend
checks. A separate whole-diff adversarial review repeated Release 42/42,
ASan/UBSan 15/15, TSan 4/4, 100/100 executor stress, exact ISA inspection,
provider-absence, install, ABI, and path-leak checks. It found no unresolved
high- or medium-severity issue. The validated production and test tree is
unchanged by the later evidence-only commits.

The focused Windows x64 compatibility phase is validated on GitHub-hosted
Windows Server 2025 with clang-cl/LLVM 21.1.8, MSVC tools 14.51.36231, Windows
SDK 10.0.26100.0, and lld-link. The native LibTooling frontend, valid-C++
`.mdsl` pipeline, COFF objects, PE executables, runtime DLL/import library,
installed package, and external consumer all passed with space- and
Unicode-bearing paths. Release passed 35 tests with one intentional AVX-512
hardware skip; focused Debug passed 26 with the same explicit skip. The
runtime DLL and generated host/stub/backend pipeline also executed under the
supported focused clang-cl AddressSanitizer scope. The packaged LLVM Tooling
executables are not claimed ASan-instrumented because the authenticated LLVM
archive's allocator conflicts with the Windows static ASan allocator thunk;
Windows UBSan is not claimed.

The hosted CPU has two physical cores, four logical CPUs, one socket, and one
NUMA node. Reference, tiled, compiler-vectorized AVX2/FMA, native packed
AVX2/FMA, and persistent parallel AVX2/FMA variants are runtime-validated.
AVX-512 packed and parallel functions are compile/disassembly-validated only:
the host exposes no OS-usable AVX-512 and forced execution fails closed.
OpenBLAS was deliberately omitted from the Windows package. Multi-node NUMA is
synthetic-only, and the Microsoft Visual C++ 2015--2022 x64 Redistributable is
an explicit external prerequisite; a clean-machine installer experience was
not validated. The CI distribution ZIP contains 17 installed files, exposes
15 C ABI functions, passed recursive import and absolute-path-leak checks, and
has SHA-256
`b2c633192d3084585198f24eedba3957a85552c5d483d3b656bfdeda60480cd2`.

## Milestone 4 CPU performance foundation

**Completed and published for the declared single-thread Linux host scope. PR
#10 passed hosted checks and merged normally into `main` at `e4dc0af`; issue #8
and milestone #2 are closed, and tag `mdslc-cpu-performance-foundation-v1`
anchors the merge.**

Planner v2 evaluates five stable implementations:

```text
cpu.reference.f32.v1
cpu.tiled.f32.v1
cpu.compiler-vectorized.avx2-fma.f32.v1
cpu.external.openblas.f32.v1
cpu.native-packed.avx2-fma.f32.v1
```

OpenBLAS 0.3.32 is optional, authenticated through LP64 CBLAS, controlled with
the provider's process-local thread API, and bounded by the provider-reported
thread ceiling before planning. The native packed engine uses caller-owned
64-byte-aligned workspace, MC=128, NC=256, KC=256, MR=4, and NR=16. Its exact
microkernel object contains YMM packed-FMA instructions; the rest of the
runtime remains generic and capability-gated.

The existing one-shot C ABI remains compatible. Additive v1 APIs query and
execute with explicit caller workspace and support caller-owned prepacked B.
Insufficient/misaligned/overlapping storage and forced illegal providers or
ISAs fail before output mutation. MDSLC adds no hidden packing/workspace
allocation or host/device tensor copy, and no silent fallback; an opaque
OpenBLAS provider may manage internal memory under its own contract.

`matcore-bench` freezes the JSON benchmark contract and distinguishes complete
one-shot, reused workspace, prepacked B, and diagnostic-only packed-compute
intervals. Raw results remain ignored outside Git. On the pinned validation
host, the 30-shape deterministic calibration produced median regret 1.000,
p95 1.124, maximum 1.132, and no choice above 2.0. Native packed beat the prior
compiler-vectorized candidate on 27/30 shapes; OpenBLAS was fastest on 26/30.
These are host/provider-specific observations, not universal or BLAS-parity
claims.

Independent exact-tip validation passed fresh Release 27/27, Debug 27/27,
focused ASan/UBSan 8/8 with repeated benchmark smoke, OpenBLAS-disabled 5/5,
package/install checks, seven-symbol C ABI inspection, exact AVX2/FMA
disassembly, repository hygiene, and a fresh `.mdsl -> ELF .o -> executable`
run printing `MDSLC CPU GEMM PASS`. The review resolved four medium findings:
provider-thread overcommit, a misleading compute-only mode, double-live
one-shot allocation/memory accounting, and flaky timing-smoke acceptance.

Evidence is in ADRs 0006/0007,
`docs/performance/cpu/milestone-4-single-thread-calibration-2026-07-22.md`, and
`docs/mdslc/agent-reports/m4-final-adversarial-review.md`.

Windows had only the versioned portability seed at this milestone. The later
Milestone 5 branch implements Linux AVX-512 F32, persistent native parallelism,
and typed BF16/I8 reference semantics; it still does not claim Windows,
accelerated BF16/I8, AMX, real multi-node NUMA, GPU, or autotuning support.

## Milestone 3 mainline checkpoint

**Completed and published. PR #6 merged normally; the later controlled history
sanitation preserved the source tree while remapping commit IDs.**

The integration branch starts from `origin/main`, contains the historical
`feature/device-resident-tensors` line and the complete reviewed MDSLC lineage,
and uses ordinary merge commits. It does not rebase, squash, force-update, or
delete either history. The `compiler/` tree is byte-identical to the accepted
Milestone 2 commit.

Fresh committed-tree Release and Debug standalone suites each passed 14/14.
A coherent temporary MLIR 18 legacy build linked successfully; the meaningful
legacy regression set passed 68 pytest-compatible cases, 4 CUDA graph cases,
the CPU dtype/shape matrix, 24 elementwise GPU cases, and 7 softmax GPU cases.
The machine's prior MLIR 22 package surface was restored after that proof.

The integration decision and full evidence are recorded in
`docs/adr/0004-mdslc-mainline-history-consolidation.md` and
`docs/mdslc/agent-reports/mainline-consolidation-validation.md`.

## Milestone 2 verdict

**Completed and published. Typed Matcore IR v1 and deterministic CPU GEMM
planning passed their local, independent-review, hosted, and merge gates.**

The native driver now routes every authenticated `matcore::mdsl::gemm` through
a verified v0-to-v1 boundary. IR v1 carries typed shape, dtype, accumulation,
layout, stride, alignment, memory, mutability, effects, alias, synchronization,
policy, requirement, provenance, and exact source-range contracts. Only a
lossless canonical subset projects into the existing rewrite/codegen and v0
execution ABI.

The CPU runtime validates descriptors before discovering versioned host
capabilities. It evaluates a fixed reference/tiled/compiler-vectorized registry,
rejects illegal variants, applies saturating deterministic integer costs, emits
complete candidate and selected-plan diagnostics, and executes exactly the
selected lowering. The additive `matcore_runtime_plan_gemm_f32_v1` query and
installed `matcore-plan` tool expose the same decision without executing or
modifying output.

Current validation host capability record:

```text
x86_64; discovery complete; portable scalar f32, AVX2, FMA; 256 vector bits
```

The final fresh Release suite passed 14/14 CTest tests in 62.11 s, and the
final fresh Debug suite passed 14/14 in 63.14 s. The ASan/UBSan focused set
passed 9/9, and a generated executable built through the same instrumented
pipeline printed `MDSLC CPU GEMM PASS`. Runtime tests passed an independent
double-precision oracle; five representative benchmark shapes compared every
legal implementation with the forced reference implementation and passed
generous absolute regression guards. Results and exact contracts are recorded
in `docs/mdslc/agent-reports/matcore-ir-v1-cpu-planner-final.md`.

Independent review found and reproduced implementation and evidence defects,
including a vector-named Debug implementation whose `-O0` body was scalar.
The fixes preserve Release `-O3`, enable `-O2` only for Debug/default, require
function-local YMM packed-FMA instructions, and suppress vector capability
when sanitizer instrumentation scalarizes that body. All confirmed
high/medium findings were revalidated before final sign-off.

At the Milestone 2 checkpoint no coherent BLAS development package was
available, so no adapter was part of that historical gate. Milestone 4 later
added the optional authenticated OpenBLAS implementation described above.

## Milestone 1 foundation

**Architecture proof passed for the standalone native CPU frontend/runtime
vertical slice.**

The default `.mdsl` operation path now uses an in-process Clang 21.1.8
LibTooling frontend, not AST JSON. It authenticates the resolved public header,
canonical declarations, exact annotations, public ABI, and SourceManager token
ranges, then reuses the verified JSON IR/codegen/C ABI/runtime pipeline.

No Python, nanobind, MLIR, legacy JIT, or bootstrap subprocess participates in
the default compiler/runtime path. The AST-JSON frontend remains explicit
compatibility/differential mode only and is never a silent fallback.

## Implemented pipeline

```text
valid C++ foo.mdsl
  -> mdslc++ (native frontend by default; forces -x c++)
  -> pre-extraction dependency scan and immutable input closure
  -> in-process ClangTool + PPCallbacks + parse/Sema
  -> direct resolved trusted-header identity and content/ABI authentication
  -> ASTMatcher + getDirectCallee() + canonical FunctionDecl
  -> exact AnnotateAttr("matcore.op.gemm") authentication
  -> SourceManager/Lexer source locations and call/argument token ranges
  -> deterministic, verified Matcore JSON IR v0 compatibility capture
  -> explicit upgrade to typed, verified Matcore JSON IR v1
  -> loss-checked projection into existing rewrite/codegen
  -> exact call rewrite + generated sites/stubs/backend
  -> three ordinary Clang C++ objects
  -> clang++ -r combined relocatable object
  -> ordinary clang++ link against versioned libmatcore_runtime
  -> capability-v2 discovery + topology-v1 allowed-processor restriction
  -> exact-context hardware/OS/compiler/implementation/runtime authentication
  -> deterministic planner-v3 legality, cost, placement, and diagnostics
  -> selected serial or persistent-context F32 implementation (eight variants)
  -> caller-owned shared/per-worker workspace or prepacked-B execution
  -> synchronous checked CPU result through the stable/additive C ABI
```

The main source and every non-system dependency are snapshotted before
extraction and checked after extraction, each generated compilation, linking,
and dependency publication. The rewritten host is compiled through a VFS
mapping at the original `.mdsl` path, preserving quote includes and diagnostic
context. User VFS/PCH/module injection is rejected. Semantic compile arguments
contribute to stable site identity; weak generated wrapper/backend definitions
allow equivalent deterministic sites to co-link across independent roots.

## Authentication and rewrite contract

- The main translation unit must directly resolve `matcore/mdsl.h` to the
  build/install package's trusted `FileEntry` unique identity.
- Clang's parsed header buffer must match the physical snapshot; the expected
  record/enum/field/type/value/default/signature ABI is checked after Sema.
- External macros may not alter public Matcore declarations.
- A GEMM call must resolve directly to the trusted canonical
  `matcore::mdsl::gemm`, have the supported signature, and contain exactly the
  expected non-inherited annotation. `out` is authenticated similarly.
- Namespace aliases work through canonical identity. Fake/copied/shadowed
  headers, user overloads, unqualified/ADL calls, indirect calls, templates,
  lambdas, macro-generated or header-originating sites, constexpr/unevaluated
  contexts, side-effectful arguments, and unsupported ABI/layout/policy cases
  fail with source diagnostics.
- Rewriter ranges are half-open SourceManager/Lexer token ranges from the same
  parsed source snapshot; no regular-expression call matching or manual token
  boundary estimation is used.

## Historical Milestone 1 validation results

| Validation | Result |
| --- | --- |
| Final fresh Release, `/tmp/matcore-native-v1-final5-release.1R1Ecm` | 19/19 build steps; 8/8 CTest tests passed in 61.55 s |
| Final fresh Debug, `/tmp/matcore-native-v1-final5-debug.YZDSYz` | 19/19 build steps; 8/8 CTest tests passed in 62.94 s |
| Native focused frontend | 16 checks passed |
| Native primary frontend | 44 checks passed |
| Native/parity/adversarial core after final hardening | 536 checks passed |
| Native driver pipeline | 72 checks passed |
| Integration matrix | 63 passed, 0 failed, 6 future capabilities not counted |
| Strict `-Wall -Wextra -Wpedantic -Werror` build | build passed; full standalone CTest passed 8/8 |
| Native-disabled/bootstrap-enabled | 3/3; default failed, explicit bootstrap passed |
| Native-enabled/bootstrap-disabled | 4/4 tests passed |
| Installed native frontend checks | 7 checks passed |
| Runtime unit test | all CPU GEMM v0 cases passed |

Native/bootstrap parity compares every semantic IR field and generated
host/sites/stubs/backend output after excluding only the intentional producer
field. The native producer is `clang-libtooling-v1`; compatibility output is
`clang-ast-json-bootstrap-v0`. A deliberate mismatch self-test proves that the
comparator detects semantic drift.

### Sanitizers

`/tmp/matcore-native-v1-final5-sanitize.WdaSNP` used:

```text
-fsanitize=address,undefined -fno-sanitize=pointer-overflow
-fno-omit-frame-pointer
```

The supported focused set passed 4/4: native focused, native primary, native
core, and runtime. A separately sanitizer-instrumented generated GEMM under
`/tmp/matcore-native-v1-final5-artifacts.NM2iTj` executed
`MDSLC CPU GEMM PASS` with leak detection and no report. Its object contains
ASan and UBSan instrumentation symbols.

The full sanitizer CTest driver/package set is deliberately not counted:
configure-only sanitizer flags are not automatically forwarded into the child
consumer/final links those tests launch. This is a test-orchestration limitation,
not a claimed sanitizer pass. Pointer-overflow instrumentation is disabled only
for the known RapidJSON 1.1 compatibility expression in the explicit bootstrap
implementation.

### Install and consumer

Final fresh installation to
`/tmp/matcore native v1 final5 direct.wK4ayb/install prefix` produced:

- `bin/mdslc++` and native `bin/matcore-extract`;
- `include/matcore/{mdsl.h,runtime_c.h}`;
- versioned `libmatcore_runtime.so`;
- relocatable `MatcoreDSLConfig.cmake`, targets, version, and compile helper.

The final external consumer at
`/tmp/matcore native v1 final5 direct.wK4ayb/consumer build`
configured through `find_package(MatcoreDSL REQUIRED)`, built, ran, and returned
to a no-op Ninja build. The final Release installed-consumer test separately
verified regeneration after `.mdsl` and included-header edits. The prefix
deliberately contains spaces. Binary/text scans found no source or build-tree
absolute-path leak and no Python/nanobind dependency.

### CPU and artifact proof

The final saved artifact proof is
`/tmp/matcore-native-v1-final5-artifacts.3aSn01`.
`hello_host.mdsl` prints `5`. `gemm_v0.mdsl` prints:

```text
host-before
MDSLC CPU GEMM PASS
```

`file`, `readelf`, `nm -C`, and `ldd` confirm an ordinary ELF64 relocatable
combined object, ordinary generated C++ site/backend symbols, an unresolved
`matcore_runtime_gemm_f32_v0` boundary before final link, a normal ELF PIE after
ordinary Clang linking, and resolution through `libmatcore_runtime.so.0`.
`--save-temps` retains the host source, VFS overlay, verified JSON, sites header,
stubs, backend, three component objects, and combined object. A repeat build
produced byte-identical IR and generated sources; the JSON SHA-256 is
`afd693d72e2574d27aae53a8ccd50e975404bee6cf19425cc05f64739016d480`.
Verbose driver evidence invoked the native extractor and ordinary Clang
compilations; no AST-dump invocation or bootstrap marker appeared.

## Primary commands

```sh
cmake -S compiler -B /tmp/matcore-native-v1-final5-release.1R1Ecm -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-native-v1-final5-release.1R1Ecm -- -j2
ctest --test-dir /tmp/matcore-native-v1-final5-release.1R1Ecm \
  --output-on-failure -j1

/tmp/matcore-native-v1-final5-release.1R1Ecm/bin/mdslc++ \
  -std=c++20 --matcore-target=cpu --save-temps -c \
  compiler/examples/gemm_v0.mdsl \
  -o /tmp/matcore-native-v1-final5-artifacts.3aSn01/gemm_v0.o
/usr/bin/clang++-21 \
  /tmp/matcore-native-v1-final5-artifacts.3aSn01/gemm_v0.o \
  -L/tmp/matcore-native-v1-final5-release.1R1Ecm/lib \
  -lmatcore_runtime \
  -Wl,-rpath,/tmp/matcore-native-v1-final5-release.1R1Ecm/lib \
  -o /tmp/matcore-native-v1-final5-artifacts.3aSn01/gemm_v0
/tmp/matcore-native-v1-final5-artifacts.3aSn01/gemm_v0
```

## Historical legacy regression and device inventory

Selected legacy Python tests: 26 passed; 3 failed only because the legacy
`_matcore_native` extension is not built in this standalone worktree. A fresh
root CMake probe still fails at the known MLIR contract: root asks for 18.1.3,
while available MLIR configuration is 22.1.2. No legacy root build is claimed
green.

CPU execution is runtime-validated. NVIDIA RTX 4060/CUDA 13.3, AMD
`gfx1150`/ROCm 7.1, and `aie2p` NPU are detected only. CUDA, cuBLAS, HIP, NPU,
and all accelerator compiler paths were not attempted.

## Known limitations

- Linux and hosted Windows x64, Clang/LLVM 21.1.8, one `.mdsl` input, and
  synchronous host-resident rank-2 row-major contiguous GEMM are the validated
  compiler/runtime scope. Linux remains the only performance-calibration host.
- `matrix_view` is a minimal host f32 view, not a general tensor framework.
- Driver-managed shared/static/PIE modes and opaque response/linker option
  forms remain rejected; emit `-c` and perform an ordinary explicit final link.
- Bootstrap remains compatibility-only and is not the semantic authority.
- AVX-512F/FMA F32 and persistent native parallel AVX-512 execution are
  physically exercised only on the declared Linux host. Windows AVX-512 is
  compile/disassembly-only because hosted hardware lacks OS-usable AVX-512.
  BF16/F32 and I8/I32 have typed reference semantics only; accelerated BF16,
  VNNI, and AMX variants are not implemented.
- One-node topology discovery is physically validated on Linux and hosted
  Windows. Multi-node NUMA policy is synthetic-only, with no runtime page
  placement or migration.
- Windows OpenBLAS, a clean-machine installer test, GEMV, GEVM, ReLU-GEMM,
  Windows Matcore-MLIR semantic execution, generated Linalg/Vector compute,
  CUDA/cuBLAS, HIP, Metal, NPU placement, fusion, and autotuning are not
  implemented or claimed. Linux explicit-GEMM Matcore-MLIR runtime-dispatch
  lowering exists; map/domain and recovered-loop routes remain inspection-only.

The standalone native CPU architecture proof, Milestone 4 performance
foundation, published Linux Milestone 5 backend, and focused hosted Windows
x64 compiler/runtime/package validation remain passed. Performance evidence
and unavailable-hardware status remain platform-specific.
