# MDSLC roadmap

Status date: 2026-08-11

The roadmap is gate-driven and additive to the existing Python/JIT lineage.
The primary direction is now a compositional semantic compiler and CPU-first
beta, not native-BLAS parity followed immediately by an API freeze. A backend
is supported only after it compiles, links, executes, validates correctness,
and passes its declared platform gates.

## Authenticated pivot baseline

The pivot begins from canonical `main` at
`e5069758ad04bdb459de2026cad8498b47fda707`:

- PR #16 is merged normally;
- Issue #15 and GitHub milestone #5 remain open;
- Milestone 7 has an accepted partial disposition and no completion tag;
- no public API, ABI, or backend-contract freeze has begun;
- the native Clang frontend, typed Matcore IR v1, deterministic CPU planner,
  Linux/Windows package paths, and CPU GEMM vertical slice remain validated.

Milestone 7 is still valuable evidence work. It is no longer the only critical
path and native parity is not a CPU-beta prerequisite.

## Completed foundations

1. **Standalone valid-C++ compiler.** `mdslc++` and native Clang 21
   LibTooling authenticate canonical `matcore::mdsl` calls and preserve
   ordinary host C++.
2. **Typed capture boundary.** Deterministic Matcore IR v0 compatibility and
   typed, verified Matcore IR v1 preserve operation, type, shape, layout,
   alignment, memory, mutability, effects, alias, policy, provenance, and
   source-range contracts.
3. **CPU planner and runtime.** Deterministic fail-closed planning selects
   validated reference, tiled, compiler-vectorized, native packed/parallel, or
   optional OpenBLAS implementations under explicit workspace and capability
   contracts.
4. **Native artifacts and distribution.** Ordinary objects/executables,
   versioned C ABI, installable CMake package, Linux x86-64, and bounded hosted
   Windows x64 paths are validated.
5. **Repository and performance foundations.** Mainline/history sanitation,
   generated-artifact hygiene, the CPU benchmark contract, Advanced CPU
   Backend v2, and CPU Performance Deep Audit are complete and tagged.
6. **Milestone 7 bounded hardening.** Two-dimensional task geometry, wider
   AVX-512 full tiles, exact ISA artifact guards, and fail-closed parity
   evidence tooling merged. A complete authenticated parity envelope remains
   uncollected, so parity is not claimed.

The standalone native CPU frontend/runtime verdict remains passed. The
semantic-foundation branch now also contains the independently accepted
Matcore MLIR core, map/domain composition model, conservative recovered-GEMM
analysis, and explicit-GEMM CPU runtime-dispatch lowering. Map/domain and
recovered-loop execution are deliberately not part of that executable claim.

## Architectural delta

The old near-term ordering was:

```text
finish native BLAS parity
  -> freeze current public/backend contracts
  -> add operations and target lowerings later
```

The new ordering is:

```text
freeze semantic responsibilities
  -> preserve Matcore IR v1 as capture/provenance DTO
  -> bridge represented facts exactly into a Matcore MLIR optimizer dialect
  -> prove multi-operation and recovered-intent semantics
  -> prove the semantic loop through the existing CPU planner/runtime
  -> resolve pre-freeze contracts
  -> ship a CPU-first beta
```

The invariant is that lowering consumes information only after its meaning is
structurally represented downstream or every optimizer that needs it has made
its decision. WHAT, HOW, and MACHINE responsibilities are defined by
[ADR-0009](../adr/0009-mdslc-semantic-compiler-foundation.md).
Linalg, Tensor, MemRef, and Vector remain HOW-level structured substrates;
MACHINE begins when LLVM or a target-specific dialect/toolchain structurally
encodes the selected target choice.

## Milestones A--H

### A — Semantic Compiler Architecture Freeze

Status: complete on the semantic-foundation branch. ADR-0009 and the
cross-document invariant set passed independent review with no unresolved
high- or medium-severity finding.

Freeze the internal architectural boundary, not the public API:

- Matcore IR v1 remains the versioned capture/provenance DTO;
- Matcore MLIR becomes the SSA/region/use-def optimizer representation;
- WHAT/HOW/MACHINE responsibilities are explicit;
- semantic information-consumption rules are verifier obligations;
- numerical, effect, alias, mutation, domain, execution-intent, policy, and
  capability boundaries are inspectable;
- recognition and permission are separate decisions.

Gate: ADR/roadmap/status/pre-freeze log agree, repository truth is current,
the coherent MLIR 21 dependency gate is documented, and independent review has
no unresolved high or medium issue.

### B — Matcore MLIR Core

Status: complete on the semantic-foundation branch. The implementation uses an
exact LLVM/Clang/MLIR 21.1.8 tuple, a TableGen-generated `mdsl` dialect, a
deterministic verified v1 bridge, and an internal `matcore-mlir` inspection
tool. Independent review found no unresolved high- or medium-severity issue.

Implement the `mdsl` textual dialect with generic SSA values, `mdsl.gemm`,
minimal type/attribute mapping, exact destination/effect/alias semantics,
source provenance, and a deterministic verified Matcore IR v1 bridge. Because
v1 currently lacks a complete numerical policy, B must encode and verify the
`explicit-gemm-f32-v1` profile frozen by ADR-0009: F32 accumulation,
contraction allowed, K-reduction-only reassociation, implementation-defined K
order, NaN/non-finite preservation without payload/order guarantees, relaxed
signed zero, no approximate math, explicit destination overwrite, and no
input/output aliasing or in-place operand mutation. Recovered loops require
source-derived proof and cannot inherit these permissions. The core dialect
also represents a strict increasing-K recovered-loop form as analysis-only;
its `recognized_rewrite_rejected` permission cannot enter the executable v1
bridge envelope. A separately authenticated source-proven relaxed form records
that a dominating runtime guard is still required.

`mdsl.gemm` produces the post-overwrite semantic value of its explicit
write-only destination, not independent allocated storage. Bufferization must
alias result storage to that destination and preserve the observable write.
Alignment and no-alias fields are required preconditions, not proven facts;
they become optimization facts only after static proof or a dominating runtime
guard whose rejection occurs before output mutation.

Gate: matching LLVM/Clang/MLIR 21.1.8; generated dialect build; deterministic
parse/print and v1 bridge goldens; destination/result and precondition/fact
verifier negatives; malformed semantic negatives; unchanged v0/v1
compatibility tests; independent semantic review. These focused gates pass:
the core semantic harness reports 204/204 checks, the CLI harness 9/9, and the
two MLIR CTest entries 2/2. The later installed-package profile also exercises
the core through the semantic route. The final settled candidate's full
configuration matrix remains Milestone H evidence rather than a reason to
reopen this accepted semantic-core boundary.

### C — Multi-op and domain semantics

Status: implemented and independently accepted for the internal
composition-v1 optimizer/inspection boundary. It does not authorize map/sine
CPU lowering or a public map API.

Implement `mdsl.map`, `mdsl.sin`, `mdsl.yield`/`mdsl.return` as required, with
`all`, slice, indices, and predicate/mask domain concepts. Prove canonical
GEMM-to-SIN(all) and one partial-domain case. Use SSA/use-def order and explicit
effects rather than unnecessary total program order.

Gate: domain bounds/shape, untouched-element, mutation, numerical, alias,
effect, and reordering negative tests plus deterministic textual goldens. The
focused semantic binary reports 319 checks with zero failures, and the core
plus map/domain CTest pair passed 2/2 in independent review. The CPU lowerer
must continue to reject this unsupported composition rather than execute GEMM
alone.

### D — Explicit/recovered equivalence prototype

Status: implemented and independently accepted for analysis and authenticated
structural-equivalence inspection. It remains deliberately non-executable and
does not rewrite ordinary C++.

Conservatively recognize one canonical non-dependent ordinary C++ GEMM loop
and raise it directly into the same `mdsl.gemm` semantics as explicit
`matcore::mdsl::gemm`. Recognition is not permission. Rejected replacement
preserves ordinary C++ behavior.

Gate: semantic equivalence; source provenance; dependence/alias/numerical/
barrier/macro/volatile/atomic negatives; unchanged object/execution behavior
when raising is rejected. Independent rereview accepted the sealed native
evidence and explicit/recovered comparison with no unresolved high or medium
finding. Both strict and guard-required recovered modules remain analysis-only
and are rejected by the executable CPU lowerer.

### E — CPU MLIR lowering proof

Status: implemented and independently accepted for the focused Linux
authenticated explicit-F32-GEMM runtime-dispatch path. Full CPU-beta
configuration and hosted evidence remains Milestone H work.

Route authenticated capture through verified Matcore MLIR into legal CPU
planning/lowering, reusing the current runtime and implementation registry.
Structured Linalg/Tensor/MemRef/Vector work may be added as HOW-level
substrates only when it provides a validated, explicit alternative. LLVM and
target-specific dialect/toolchain lowering begins MACHINE. Do not replace
proven code merely to claim MLIR usage.

Gate: correct `.mdsl -> v1 -> mdsl MLIR -> CPU -> .o -> executable`,
static proof or dominating pre-mutation guards for alias/alignment,
round-to-nearest-ties-even, non-trapping exceptions, gradual subnormal
handling with FTZ/DAZ disabled, backend numerical conformance, independent
oracle, artifact inspection, forced-illegal/environment failure, Release,
Debug, sanitizers, package relocation, external consumer, and Windows
compatibility.

The focused proof passes through the stable
`matcore_runtime_gemm_f32_v0` descriptor ABI and produces an ordinary native
artifact whose backend carries the semantic-lowering producer marker. The
runtime now enforces the required source-evaluation and execution-thread FP
environment on the physically validated Linux x86-64 scope, using additive
status 26 on rejection. This is library-dispatch lowering; Linalg/Vector loop
generation, map execution, and recovered-loop execution are not claimed.

### F — Milestone 7 evidence closure or bounded technical limit

Status: accepted bounded technical limit for the semantic-pivot dependency.
Issue #15 and milestone #5 remain open because native-BLAS parity itself is
still partial and may be revisited only on an exclusive host.

Run the unchanged authenticated forward/reverse envelope. If it passes, record
the result. If it fails, retain best-provider selection and document the
largest measured limits. Cooperative packed-B preparation remains dormant
until final-checkpoint evidence justifies a legal activation rule.

Gate: unchanged provider, shapes, thread rules, timing, legality, and regret
scope; complete authenticated pair or an explicitly reviewed bounded limit;
no benchmark gaming or fabricated counter evidence. The reviewed bounded
disposition is the second allowed outcome: the incomplete 258/368 forward
receipt is not reused as a parity result, cooperative packed-B preparation
remains production-dormant, and no parity tag is created.

### G — Pre-freeze contract resolution

Status: independently accepted for bounded existing-version decisions. The
broader public API/ABI/backend-contract freeze remains explicitly deferred.

Resolve transformed-operand ownership/identity/immutability/lifetime,
cross-context sharing, caller-owned transformed storage, report iteration,
forced variant identity, device-neutral execution context, requested/actual
resources, dynamic shape constraints, diagnostic serialization, execution
intent, and operation evolution/deprecation.

Gate: additive compatibility design, retained symbol/layout tests, Linux and
Windows installed consumers, and independent ABI/backend-contract review.
This gate still does not freeze an interface merely by documenting it.

The accepted bounded decisions define packed-B v1 as caller-owned borrowed
storage with serial synchronous reuse and manual invalidation, define returned
C strings as borrowed, and require additive version evolution. General
transformed-operand ownership/sharing, structured plan/report iteration,
execution intent, forced variant identity, and long-term support/deprecation
policy remain open for the separate freeze milestone.

### H — CPU Beta

Status: active. A--G have their required bounded dispositions. The full local
integration matrix passed at immutable code candidate `6796fd8`, superseding
the historical `69d099e` matrix after later product hardening. Final independent
evidence review, hosted Linux/Windows integration, normal merge, and beta tag
remain pending.

The intended initial claim is valid C++ `.mdsl`, preserved ordinary C++,
explicit F32 rank-2 GEMM, verified capture and semantic IR, deterministic legal
CPU planning, reference/native/OpenBLAS routes, Linux x86-64, bounded Windows
x64 compiler/runtime/package validation, native artifacts, installable CMake,
inspection tools, explicit resources, no hidden copies, and no silent
fallback.

Gate: fresh Release, Debug, supported sanitizers, package, external consumer,
Windows, native artifact, IR/verifier, explicit/recovered recognition,
planner, performance sanity, hygiene, and independent adversarial review.

Product-profile boundary:

- source compatibility default: Matcore MLIR disabled and semantic pipeline
  `capture-v0`;
- Linux CPU-beta profile: exact MLIR 21.1.8 enabled and configured default
  `matcore-mlir`;
- Windows x64 compatibility profile: Matcore MLIR disabled and configured
  default `capture-v0`, with an explicit semantic request required to fail
  unavailable and leave no artifact.

Installed packages report `MatcoreDSL_MATCORE_MLIR_AVAILABLE` and
`MatcoreDSL_DEFAULT_SEMANTIC_PIPELINE`. Consumers may pass
`SEMANTIC_PIPELINE capture-v0|matcore-mlir` to
`matcoredsl_add_executable`; invalid, unavailable, or bootstrap/MLIR pairings
fail during configuration. The detailed supported/unsupported claim and
remaining acceptance matrix are frozen in [CPU_BETA_V1.md](CPU_BETA_V1.md).

## Dependency graph

```text
A -> B -> C ----------------+
      +-> D ----------------+-> E --+
current M7 contract ------------> F -+-> G -> H
                         A/B/C/D/E ---+
```

F is evidence-parallel but cannot change A--E semantics. G requires a bounded
F disposition, not a manufactured parity pass. H does not depend on GPU work.

## Branch/worktree plan

- `mdslc/semantic-compiler-foundation-v1` starts from authenticated clean
  `main` and owns A plus the integrated B--E vertical slice.
- Architecture, dialect/bridge, frontend recognition, CPU integration,
  performance evidence, and adversarial review use isolated sibling worktrees
  and non-overlapping files.
- Milestone 7 remains tracked separately by Issue #15 and milestone #5.
- Bounded contract resolution and the CPU-beta candidate are integrated on
  `mdslc/semantic-compiler-foundation-v1`; publication still requires a normal
  pull request from that branch after the final H gates pass.
- Integration uses focused commits and normal merges. No rebase, history
  rewrite, tag movement, or deletion of legacy/user work is part of this plan.

## Toolchain gate

The standalone compiler uses LLVM/Clang 21.1.8, Ubuntu revision
`1:21.1.8-6ubuntu1`. A system-install simulation showed that adding MLIR 21
would remove the installed MLIR 22 surface. The exact `libmlir-21`,
`libmlir-21-dev`, and `mlir-21-tools` payloads were therefore extracted into a
versioned user-local development prefix without changing system packages.
The lane validated `clang-cpp` + MLIR + LLVM integration, TableGen dialect
generation, and narrow static MLIR component linking.

The local prefix is never a product path: builds accept it through
`MLIR_DIR`, prefer narrow imported MLIR components, and must not export or
embed the prefix. The system LLVM/Clang CMake packages remain version 21.1.8;
mixing the installed MLIR 22 surface is a hard error.

MLIR support remains isolated under `compiler/`. It neither changes the legacy
root CMake request for MLIR 18.1.3 nor makes the compatibility CPU path depend
silently on MLIR. No LLVM source build is planned.

## Deferred work

- Public GEMV, GEVM, and ReLU-GEMM APIs are not part of A--H. Private design
  work remains input to later semantic-operation milestones.
- Broad CUDA, HIP, Metal, Vulkan, NPU, heterogeneous placement, GPU fusion,
  and accelerator kernel work waits until the CPU semantic loop passes.
- Runtime autotuning, a general graph compiler, and a full tensor framework
  remain out of scope.
- A GPU backend will reuse the same `mdsl.gemm` WHAT and introduce explicit
  target capability, legalization, planning, runtime, artifact, correctness,
  and performance gates rather than a backend-specific source foundation.
