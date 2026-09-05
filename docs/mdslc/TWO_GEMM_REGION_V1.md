# Authenticated ordered two-GEMM region admission v1

Date: 2026-09-05. Canonical starting checkpoint:
`5f455bacde0959983b2b888f15fd5dabd4b1ceaa` (PR #28).

This is the bounded milestone accepted after the adversarial architecture
review, not the previously proposed tiled/dynamic GEMM milestone. The existing
CPU runtime/provider route is unchanged. No public API, ABI, interchange format
or backend contract is frozen.

## What this establishes

Native Clang authentication can admit the real source dependency
`gemm(out(C), A, B); gemm(out(E), C, D);` into one inspection-only semantic
region. Declaration identity, descriptor identity, mathematical tensor values
and observable effects are separate. A source-paired verifier checks the actual
structured computation and effect chain, not just matching certificate banners.

```text
ordinary valid C++ / authenticated direct calls
                       |
       sealed native source + declaration bindings
                       |
   descriptor handles (distinct handles may alias bytes)
                       |
                    begin
                       |
                 guard call 0             conditional obligations, not executed
                       |
                read A / read B
                       |
        empty + positive-zero fill + GEMM
                       |
                 commit C -> C1            observable write + new tensor version
                       |
                 guard call 1              cannot precede commit C
                       |
                    read D                 may alias the bytes just written
                       |
        empty + positive-zero fill + GEMM(C1, D)
                       |
                 commit E -> E1
                       |
                      end
                       X                   no generated execution authority
```

`!mdsl.region_descriptor` is a source descriptor reference, not a public view,
an ABI argument, or a disjoint storage allocation. `!mdsl.region_order` carries
successful-continuation ordering. A commit returns the logical post-write
tensor and a continuation token; it does **not** promise an atomic write or
rollback. The retained failure obligation is
`may_write_output_before_failure_no_rollback`. No retry is introduced.

The model preserves two ordered opaque failure frontiers, not a lowered C++
exception CFG. Exact diagnostics, handler selection, noexcept termination and
host control flow remain in unchanged C++. A second-call failure must not erase
the first successful write. The executable tests demonstrate this on the old
per-call route; they do not execute region guards or derived MLIR.

## Admission and trust boundary

- Exactly two adjacent direct GEMM statements in the same ordinary function
  body, or one direct try body. Nested blocks, conditionals, loops, host
  statements (including empty statements), descriptor mutation, cross-function
  joins and unsupported control are barriers. Three calls form at most one
  nonoverlapping pair plus an unadmitted call.
- Only the first output feeding the second lhs is admitted. The second output
  must have a different descriptor binding. This is not a physical-disjointness
  assertion. RHS dependency and broader multi-operation graphs are unsupported.
- Binding identity comes from Clang declarations, not spelling. Transparent
  automatic reference chains can resolve to their descriptor root; persistent
  references, call/cast-derived references and ambiguous macro declaration
  locations fail closed. Qualified names with equal spelling remain distinct.
- User GEMM bodies, including definitions after use, and untrusted declaration
  attributes cannot acquire mathematical region authority. Trusted intrinsic
  attributes are checked against their own header origin. Ordinary extraction
  behavior is not changed by these inspection-only restrictions.
- Existing immutable native evidence owns the options, parsed main/dependency
  snapshots and candidates. Effective arguments and loaded compiler version
  bind the capture identity. Mutable diagnostic DTOs cannot replace the seal.
  This is in-process compiler provenance, not cryptographic authentication of
  an executable, future filesystem contents, or a persistent artifact cache.
- Individual GEMM meaning still comes through the existing verified v0-to-v1
  semantic bridge. Rich source bindings do not pass through the lossy execution
  projection, and no second JSON optimizer schema is added.

## Mechanical region contract

`MatcoreTwoGemmRegion.cpp` derives and source-pairs the module. The standalone
module verifier establishes self-consistency only; native evidence pairing is
the authority check. Each paired source contract retains dtype/accumulator,
row-major layout, output overwrite semantics, indexing and numerical policy,
provenance, shape and alias/alignment requirements, and target/provider policy.
All dimensions captured here remain dynamic. Initialization does not read the
old destination tensor. No static shape or noalias facts are guessed.

The guard groups are descriptor/rank/dtype/layout/host, positive compatible
dimensions, alignment, output/input non-overlap, floating-point environment,
and target/provider policy. They are conditional normal-continuation
requirements: no runtime predicate is discharged or reordered. Input/input
aliasing remains legal, including `A == B`; no independent restricted tensor
imports are created. Boundary memory effects conservatively alias ordinary
memory and every descriptor handle.

The verifier follows SSA uses, source bindings, per-call stages, guard/commit
ordering, exact GEMM indexing and scalar computation, full positive-zero
initialization and current input-derived output dimensions. It accepts actual
upstream named-Linalg generalization, canonicalization, CSE and SymbolDCE.
Locations, dead constants and equivalent admitted scaffolding are not identity.
Unmodeled effects, unsafe computations and unvalidated structured operations
are rejected. "Pure" is not sufficient: even unused integer division by zero
is not harmless scaffolding.

The new ops have no bufferization interface or execution lowering. The existing
CPU lowerer rejects the region even with forged producer/authority labels.
Capability, retry or target banners cannot turn inspection into execution.

## Acceptance and falsification ledger

Every review counterexample is tracked below. D means a direct region contract
test; N means rejection at an intentionally unopened authority boundary.
Negative tests are not a claim that general candidate selection, artifact
authentication, rollback or region bufferization has been implemented.

| ID | Kind | Counterexample and mechanical oracle |
| --- | --- | --- |
| F1 | D | Second GEMM consumes first committed value; stale input/wrong destination substitutions reject. |
| F2 | D | Host observer between calls blocks admission; unchanged execution observes first output. |
| F3 | D | Descriptor mutation blocks admission; per-call snapshots and wrong dynamic extent sources reject. |
| F4 | D | Same spelling cannot unify distinct declarations or cross scopes/functions; reference identity stays distinct. |
| F5 | D | Distinct descriptors may alias; D is read after C commit; execution alias oracle sees updated bytes. |
| F6 | D | A=B is admitted without input/input noalias or multiple restricted imports. |
| F7 | D | Guard 1 cannot hoist before commit 0; old-route second failure leaves C=6, E=-9. |
| F8 | D | Upstream DCE retains both observable commits/public roots; erasing a commit or region rejects. |
| F9 | D | Real named-to-generic lowering and canonicalization/CSE are accepted with equivalent meaning. |
| F10 | D | Changed numerical policy, indexing, reduction, dtype, overwrite seed or guards reject under source pairing. |
| F11 | N | Pure DPS control loses dead arithmetic; materialization control exposes alloc/copy without deallocation. Neither becomes an authenticated region/executable. Existing buffer proof rejects copies. |
| F12 | N | Atomic rollback promise rejects; potential partial-write failure retained; retry labels grant no CPU authority. |
| F13 | D/N | Changed source/dependencies/options change sealed identity; cross-capture pairing and forged producer labels fail. No binary cache/signature claim. |
| F14 | N | Changed target/permission/capability labels cannot authorize different mathematics or CPU execution. |

Additional falsifiers cover competing intrinsic bodies/attributes, static
reference bindings from previous invocations, macro declaration collisions,
unused undefined computations, output aliases of source/dependencies, and
incompatible CLI modes. Failed admission publishes no artifact and preserves
existing output files.

## Use and ownership

With the Linux native frontend and exact product MLIR build:

```sh
build-region-release/bin/matcore-extract \
  --input compiler/tests/mlir/two_gemm_region_source.mdsl \
  --two-gemm-region-out /tmp/two-gemm-region.mlir \
  -- /usr/bin/clang++-21 -std=c++20 -Icompiler/include \
  compiler/tests/mlir/two_gemm_region_source.mdsl
```

This standalone inspection request cannot also request a rewrite, per-call IR,
backend, stubs or executable semantic pipeline. Unsupported builds diagnose
the unavailable mode; they do not silently fall back.

Matcore owns source admission, bindings, legality requirements, observable
ordering and their verification. MLIR owns the structured mathematics and
tested generalization/canonicalization. LLVM/backend scheduling and instruction
selection are untouched. The runtime/provider still exclusively executes the
authenticated per-call route. GPU/NPU, fusion, tiled/vector work, public tensor
types (Issue #20), generated execution and Issue #15 parity are not advanced by
this proof.

## Validation checkpoint

Validation is recorded below once the candidate is tested. Agent findings and
their narrower claims are under `agent-reports/two-gemm-region-*.md`.

The corpus identity and limitations remain those in
`CORPUS_REENTRY_RECONCILIATION_V1.md` and
`CONTRACTION_CAMPAIGN_RECONCILIATION_V1.md`. No corpus observations become
performance thresholds, new target support, or benchmark evidence here.

## Single next justified boundary

An inspection-only **per-call guard/discharge ledger for this exact region**:
map each retained descriptor/shape/alias/fenv/policy requirement to a concrete
predicate that is source-proved or still required at runtime, preserving both
failure frontiers. Use existing runtime validation as the oracle. Do not
implement fusion, generated execution or generic candidate selection as part
of that next boundary. It is a recommendation, not work begun in this milestone.
