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

The current architecture verdict remains passed for the standalone native CPU
frontend/runtime slice. It does not yet include a compositional MLIR semantic
optimizer.

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

## Milestones A--H

### A — Semantic Compiler Architecture Freeze

Status: in progress.

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

Status: queued behind A's reviewable contract and the MLIR 21 dependency gate.

Implement the `mdsl` textual dialect with generic SSA values, `mdsl.gemm`,
minimal type/attribute mapping, exact destination/effect/alias semantics,
source provenance, and a deterministic verified Matcore IR v1 bridge. Because
v1 currently lacks a complete numerical policy, B must add a reviewed
canonical explicit-GEMM policy with explicit F32 accumulation, contraction,
reassociation, reduction-order, NaN, signed-zero, and approximation fields.
Recovered loops require source-derived proof and cannot inherit permissions.

Gate: matching LLVM/Clang/MLIR 21.1.8; generated dialect build; deterministic
parse/print and v1 bridge goldens; malformed semantic negatives; unchanged
v0/v1 compatibility tests; independent semantic review.

### C — Multi-op and domain semantics

Status: depends on B.

Implement `mdsl.map`, `mdsl.sin`, `mdsl.yield`/`mdsl.return` as required, with
`all`, slice, indices, and predicate/mask domain concepts. Prove canonical
GEMM-to-SIN(all) and one partial-domain case. Use SSA/use-def order and explicit
effects rather than unnecessary total program order.

Gate: domain bounds/shape, untouched-element, mutation, numerical, alias,
effect, and reordering negative tests plus deterministic textual goldens.

### D — Explicit/recovered equivalence prototype

Status: depends on B; may develop alongside C with separate ownership.

Conservatively recognize one canonical non-dependent ordinary C++ GEMM loop
and raise it directly into the same `mdsl.gemm` semantics as explicit
`matcore::mdsl::gemm`. Recognition is not permission. Rejected replacement
preserves ordinary C++ behavior.

Gate: semantic equivalence; source provenance; dependence/alias/numerical/
barrier/macro/volatile/atomic negatives; unchanged object/execution behavior
when raising is rejected.

### E — CPU MLIR lowering proof

Status: depends on B and the relevant C semantics.

Route authenticated capture through verified Matcore MLIR into legal CPU
planning/lowering, reusing the current runtime and implementation registry.
Structured Linalg/Vector/LLVM work may be added only when it provides a
validated, explicit alternative. Do not replace proven code merely to claim
MLIR usage.

Gate: correct `.mdsl -> v1 -> mdsl MLIR -> CPU -> .o -> executable`,
independent oracle, artifact inspection, forced-illegal failure, Release,
Debug, sanitizers, package relocation, external consumer, and Windows
compatibility.

### F — Milestone 7 evidence closure or bounded technical limit

Status: Issue #15 and milestone #5 remain open; may run in parallel when an
exclusive host is available.

Run the unchanged authenticated forward/reverse envelope. If it passes, record
the result. If it fails, retain best-provider selection and document the
largest measured limits. Cooperative packed-B preparation remains dormant
until final-checkpoint evidence justifies a legal activation rule.

Gate: unchanged provider, shapes, thread rules, timing, legality, and regret
scope; complete authenticated pair or an explicitly reviewed bounded limit;
no benchmark gaming or fabricated counter evidence.

### G — Pre-freeze contract resolution

Status: depends on A--E and a bounded F disposition.

Resolve transformed-operand ownership/identity/immutability/lifetime,
cross-context sharing, caller-owned transformed storage, report iteration,
forced variant identity, device-neutral execution context, requested/actual
resources, dynamic shape constraints, diagnostic serialization, execution
intent, and operation evolution/deprecation.

Gate: additive compatibility design, retained symbol/layout tests, Linux and
Windows installed consumers, and independent ABI/backend-contract review.
This gate still does not freeze an interface merely by documenting it.

### H — CPU Beta

Status: depends on E, G, and an honest F disposition.

The intended initial claim is valid C++ `.mdsl`, preserved ordinary C++,
explicit F32 rank-2 GEMM, verified capture and semantic IR, deterministic legal
CPU planning, reference/native/OpenBLAS routes, Linux x86-64, bounded Windows
x64 compiler/runtime/package validation, native artifacts, installable CMake,
inspection tools, explicit resources, no hidden copies, and no silent
fallback.

Gate: fresh Release, Debug, supported sanitizers, package, external consumer,
Windows, native artifact, IR/verifier, explicit/recovered recognition,
planner, performance sanity, hygiene, and independent adversarial review.

## Dependency graph

```text
A -> B -> C -----------+
      +-> D -----------+-> E ----+
current M7 contract -------> F ---+-> G -> H
                  A/B/C/D/E ------+
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
- Contract-resolution and beta publication receive new branches from the
  then-current clean `main` after their entry gates pass.
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
