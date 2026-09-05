# MDSLC contraction campaign reconciliation v1

Date: 2026-09-05

Status: durable architecture and evidence checkpoint. The canonical surviving
implementation entering this reconciliation is
`8ac6c4189b6f79aadee150007b6d26894de02660`. This record changes guidance and
status only; it grants no new execution authority and freezes no public API,
ABI, serialized IR, or backend contract.

## 1. Identity and provenance

The campaign preserved every reviewed branch head as a parent of a normal
merge. No squash, rebase, force update, or branch deletion was used.

| Proof point | Reviewed pre-merge head | Canonical merge |
| --- | --- | --- |
| CPU beta, PR #18 | `c4ea32203e6e9c972ff7174ff7b203d5f5c54391` | `6708b48a5647698469a9af191941bd4755adab7b` |
| corpus re-entry, PR #19 | `d80911e388c3883ceedff935bc7be80b9b756c57` | `f2e501a5fbf474fa6d8d7ec67bec7c10dacc1030` |
| structured GEMM, PR #21 | `4a7a8fc80c7a5489c22038f4ed4b80fd79d1069d` | `327530d287e41c4115365598e76b17e149a1c45a` |
| contraction topology, PR #24 | `1b7bc9fa212894e330777c18c918c7c533d05c4b` | `5983c2c1bf067bed9e69d0172b0944b7a4c14c00` |
| provider evidence auditor, PR #23 | `ae1852a13f446a935330ee4938dd2e02f8262bab` | `ec7a55a61c0ceb2ee4985211bbf887ccb3b1368e` |
| exact-22 portability control, PR #22 | `370fa09ade1e886f4469f80037d7fe7df4c05a67` | `8ba5861affe3b70e375b6e18eae46530b556dde9` |
| reusable derived certificate, PR #25 | `e38b51310a0e2534b1902acbe200136e382dfddf` | `09c96b980020f53bf1d352f9ee6c28fb470540ea` |
| certified bufferization, PR #26 | `7538e3ac4d15da6174a93f3e00a79a349fc74845` | `72f02d15e1a0d3a2dae2bab76ea0c1ee968e67de` |
| static vector readiness, PR #27 | `7b9501ec12fe99575cfabdd3e738d32f817370df` | `8ac6c4189b6f79aadee150007b6d26894de02660` |

The compiler-archaeology source of evidence remains the detached research tree
at `8aab0e295eb41ca5d2e1bd52c47201b05de9636b`. The two inspected manifests are:

- `corpus/manifests/windows-corpus-v1.json`, SHA-256
  `8d5d3e6d6af89fd78d3af5bd53b530a96e55920cbda65c38d30e7646cd01e187`;
- `corpus/manifests/windows-corpus-v2-archaeology.json`, SHA-256
  `68435d26abc4e498a76432290446bddfd428a64b81e59fe4640218016d3c754d`.

The corpus inventory, representative identity checks, compiler/target coverage,
and evidence limitations are frozen in
`CORPUS_REENTRY_RECONCILIATION_V1.md`. This campaign did not regenerate or copy
the corpus into the product repository.

## 2. Resulting compiler map

```text
authenticated explicit .mdsl GEMM
              |
              v
       Matcore IR v1 capture                 model-only topology carriers
       provenance + authority          GEMV / DOT / GER / aligned batch GEMM
              |                                      |
              v                                      X
      semantic mdsl.gemm                    no source/runtime authority
          /          \
         /            \ inspection only
        v              v
CPU runtime dispatch   certified structured Linalg
        |               | exact source pairing + retained contract
        |               |
        |               +--> certified One-Shot buffer specimen
        |               |      exact-program allocation/copy/destination checks
        |               |
        |               +--> exact-21 static whole-problem Vector specimen
        |                      Transform success plus exact postcondition
        v
native/OpenBLAS execution                              X
                                               no generated execution
```

The two downstream arrows from certified structured Linalg are siblings. This
campaign proves neither buffer-before-vector nor vector-before-buffer as a
universal ordering.

## 3. Evidence reconciliation

### OBSERVED

1. Authenticated explicit `mdsl.gemm` already carries operation identity,
   output overwrite semantics, shapes/layouts, effects, alias/alignment
   requirements, numerical permissions, target policy, and provenance into the
   verified semantic dialect.
2. The structured handoff preserves that contract while representing exact
   GEMM computation as positive-zero `linalg.fill` followed by canonical
   `linalg.matmul`. It remains inspection-only.
3. One internal topology model mechanically represents the standard dense
   families GEMM, GEMV, DOT, GER, and aligned batch GEMM, including admitted
   operand orientations. Their semantic identities remain distinct.
4. Positive unit M, N, or K does not reclassify a GEMM. Static zero extents are
   rejected; dynamic extents remain unknown positive runtime obligations.
5. Upstream One-Shot Bufferize, under the exact admitted options and program,
   maps static and dynamic GEMM specimens to identity-layout memrefs, fills and
   contracts into argument 2, returns argument 2, and introduces no allocation,
   deallocation, copy, cast, or tensor/buffer bridge. Negative option and
   conflicting-use controls do allocate or copy and are rejected.
6. Upstream MLIR 21.1.8 Transform vectorization produces a canonical
   `vector.contract` for admitted positive static rank-2 f32 GEMM specimens.
   The same Transform reports success but creates no Vector operation for the
   reviewed dynamic specimen; exact postcondition verification rejects it.
7. Exact coherent LLVM/Clang/MLIR 22.1.8 can exercise the semantic, structured,
   certificate, and buffer proofs. Removed transpose-named Linalg operations
   are represented by upstream generic operations with explicit canonical
   indexing maps. No Matcore lowering shim is needed.
8. The available provider corpus contains no authenticatable complete
   Native-BLAS-parity manifest pair. A forced two-thread OpenBLAS control ran
   with one actual provider thread and was correctly rejected.

### INFERRED

1. Canonical topology is the strongest reusable contraction substrate justified
   today. It is not evidence for one erased public `mdsl.contract` operation.
2. Source/site/type/opaque-contract binding and ordered-site pairing are
   operation-neutral. Admission and body semantics remain operation-specific.
3. A useful transformation result is trustworthy only when Matcore verifies an
   exact postcondition after upstream machinery runs; pass success alone is not
   proof.

### ARCHITECTURAL IMPLICATION

- Matcore owns semantic admission, legality, retained-fact consumption,
  provenance pairing, target policy/planning decisions, orchestration, and the
  postconditions required to preserve Matcore meaning.
- MLIR owns canonical Linalg, Tensor, MemRef, Bufferization, Transform, and
  Vector machinery.
- LLVM and target backends own instruction selection, register allocation,
  target scheduling, and machine lowering.
- Authenticated native/external providers remain valid execution candidates.
  Structured, bufferized, and vector artifacts do not gain execution authority
  merely by becoming more concrete.

### UNRESOLVED

- Source-level and authoritative semantic contracts for GEMV, DOT, GER,
  batched GEMM, and transpose-oriented operations are not implemented.
- Alpha/beta/update behavior, ranks/views, layout/stride and alias contracts,
  and source admission for those operations require a dedicated semantic
  decision. Issue #20 remains a design proposal and was not implemented.
- Batch support is an aligned single-batch-dimension topology control; broadcast
  and more general batches are not proved.
- Dynamic/tiled/masked Vector paths, a preferred buffer/vector ordering,
  physical output identity, callable memref ABI, generated execution, target
  legality, and profitability remain unproved.
- Exact 22.1.8 remains a Linux x64 compatibility control. It has no Windows,
  Debug, sanitizer, redistribution, product-support, or stable-interchange
  claim.

## 4. Surviving implementation matrix

| Result | Disposition | Mechanical boundary |
| --- | --- | --- |
| explicit GEMM semantics and CPU route | canonical | only current authenticated/executable operation path |
| GEMM/GEMV/DOT/GER/aligned-batch topology | canonical internal foundation | ranks, maps, orientations, loop roles; non-GEMM remains model-only |
| reusable derived-source certificate | canonical | deterministic substitution detector and exact pairing, not a signature, semantic verifier, or authority token |
| One-Shot bufferized GEMM specimen | canonical inspection proof | exact source/program/options and exact postcondition only |
| static tensor-to-Vector GEMM specimen | canonical, default OFF, exact-21-only inspection proof | whole-problem positive static rank-2 f32 only |
| exact LLVM/Clang/MLIR 22.1.8 | canonical internal compatibility control | explicit opt-in, coherent Linux x64 tuple, non-product |
| Native-BLAS parity evidence auditor | canonical diagnostic infrastructure | reports missing/invalid evidence; cannot issue a parity verdict without complete manifests |
| Issue #20 tensor/view frontend | deferred | design proposal only |

## 5. Mechanical invariants and falsification

1. Every downstream artifact remains `analysis_only` with
   `inspection_only` execution authority; the existing CPU lowerer rejects it
   without a partial execution record.
2. The generic certificate must be paired with the exact verified source and an
   operation-specific verifier. Standalone verification establishes internal
   self-consistency only.
3. Per-site fingerprints bind source identity, types, locations, and the opaque
   semantic contract; the aggregate binds the complete ordered site set.
   Reorder, drop, duplicate, substitute, or forge cases fail closed.
4. The topology decoder regenerates the selected standard operation and compares
   the full model. It cannot silently reinterpret GER as singleton-K GEMM or
   unit geometry as another operation.
5. Buffer verification rechecks the authoritative GEMM contract, exact memref
   types/maps, positive-zero fill, argument-2 destination/return path, scalar
   region, and absence of disallowed allocation/copy/bridge operations.
6. Vector verification rechecks the authoritative GEMM contract, exact full
   input transfers, canonical add contraction, positive-zero accumulator, no
   initial-C read, full output write, and result path. Nontrivial
   `M=2, K=1, N=4` prevents all-unit coverage from masquerading as a unit-K
   proof.
7. Numerical, alias/alignment, effects, layout/memory-space, dynamic-shape,
   provenance, and target-policy obligations remain explicit when they are not
   structurally consumed.
8. Exact toolchain token and loaded-library identity checks reject near-version,
   copied-library, and mixed-distro substitutions.
9. Requested/actual provider-thread equality is mandatory evidence. A requested
   policy does not authenticate actual execution.

## 6. Rejected alternatives

- one public operation identity that erases GEMM/GEMV/DOT/GER distinctions;
- treating GER as K=1 GEMM or classifying by degenerate extents;
- inferring universal in-place or zero-copy behavior from DPS;
- treating Transform success as Vector readiness;
- selecting a universal buffer-before-vector or vector-before-buffer order;
- rebuilding provenance independently in every downstream seam;
- using whole-module byte serialization as semantic identity;
- adding permanent Matcore shims for ordinary upstream operation evolution;
- promoting corpus tile widths, register-pressure observations, packing rules,
  or provider crossover guesses into semantic IR or planner policy;
- weakening actual provider-thread authentication;
- adopting the old research branch wholesale.

### Preserved experimental checkpoints

The rejected paths remain auditable rather than being deleted or rewritten:

- buffer experiments `12187a8023906a1c758800ed5d2e1346549155c0`,
  `e47b2e70726a6253874143c0a7f5b5797b2e371e`, and
  `1efa342737cc623bd7e7174c81552a604bada366` were superseded by the clean PR
  #26 restack because the survivor had to use the one canonical certificate,
  one topology owner, exact RTTI/dependency wiring, and final paired verifier;
- vector experiment `d16f83cbc29dbd20a59ebeae9e056fa3378962c2`
  and pre-restack proof `fa3b422e7e7726e6d6fae090977c701b63cfbbf3`
  remain evidence, not merge sources. The PR #27 survivor dropped its duplicate
  certificate, inherited canonical buffer/certificate ancestry, and added the
  nontrivial unit-K falsifier and exact hosted opt-in lane.

## 7. Validation record

- PRs #24, #23, #22, #25, #26, and #27 each received independent exact-head
  review and passed 19/19 hosted checks before normal merge.
- Contraction topology: 277/277 direct checks.
- Structured GEMM: 366/366 direct checks; its deterministic golden remained
  unchanged through the generalization.
- Generic derived certificate: 117/117 direct checks.
- Bufferization: 296/296 adversarial checks on exact 21.1.8 and exact 22.1.8;
  exact-21 full CTest 69/69; exact-22 full CTest 69/69.
- Vector readiness: 285/285 adversarial checks; composed exact-21 full build
  146/146 and CTest 70/70; exact-22 vector-disabled control build 142/142 and
  CTest 69/69. Vector-on plus exact-22 and vector-on without Matcore MLIR both
  fail configuration.
- Portability reviewed head: exact-21 and exact-22 full CTest 67/67 each, plus
  installed runtime-DSO identity adversaries.
- Provider auditor: 0/368 authenticatable records in each declared order and no
  authenticatable forward/reverse manifest pair; a requested-two/actual-one
  thread control was rejected. This is negative evidence, not a parity result.

The full suites include frontend, IR/MLIR, explicit GEMM CPU execution,
runtime/planner, OpenBLAS-on/off, C ABI, install/consumer/source-inaccessible,
benchmark-contract, CLI, sanitizer, repository-hygiene, and hosted Windows
surfaces according to each PR's declared matrix. No GPU/NPU execution or
performance claim follows.

## 8. Issue and authority disposition

Issue #15, **MDSLC Milestone 7 — Native BLAS Parity**, remains open under its
original criteria. The campaign added a fail-closed evidence auditor but found
no same-checkpoint complete forward/reverse manifests, no complete scaling
aggregates, no full-envelope planner regret, and no authenticated provider
multi-thread execution. Cooperative packed-B preparation remains
production-dormant. No threshold or parity claim was inferred.

Issue #20, **Design: explicit semantic tensor/view types in the C++ frontend**,
remains open and design-only. The campaign did not add its proposed source
types or use the topology model to decide that frontend design indirectly.

The existing runtime/provider dispatch remains the only execution authority.
No public API/ABI/backend/interchange freeze, GPU/NPU backend, heterogeneous
placement, broad fusion, runtime autotuning, or new public operation surface
began here.

## 9. Smallest justified next compiler milestone

The next compiler milestone is a **certified tiled/dynamic GEMM
transformation-ordering study**, not authenticated non-GEMM frontend expansion
and not generated execution.

Run independent tensor-first and buffer-first evidence branches from this
checkpoint using upstream Transform/Linalg/Vector/One-Shot machinery. Test-only
tile values may exercise the transformations; no tile, vector width, packing,
unroll, cache, thread, warp, provider threshold, or target matrix shape may
enter semantic IR or product policy.

A survivor must mechanically prove:

1. exact paired-source identity through every derived stage;
2. static non-divisible and representative dynamic geometries;
3. complete output coverage and no initial-C read;
4. K-reduction ordering and numerical-permission accounting;
5. remainder/mask and out-of-bounds legality;
6. allocation/copy accounting and post-buffer destination identity;
7. retention of unproved alias, alignment, layout, memory-space, residency,
   dynamic-shape, provenance, and target-policy obligations;
8. rejection by the current execution lowerer;
9. exact-21 regression, package, and dependency-firewall validation, with
   exact-22 results recorded separately as bounded compatibility evidence; and
10. independent adversarial review that can reject both candidate orderings.

In an independent performance-evidence lane, Issue #15 may advance only through
a provider-worker observability experiment that proves floating-point state and
actual multi-thread use. A parity sweep is not justified until that control
succeeds.
