# CPU beta final independent review

Date: 2026-08-11

Branch: `mdslc/semantic-compiler-foundation-v1`

Base: `e5069758ad04bdb459de2026cad8498b47fda707`

Tested code candidate: `6796fd85963f985fb652eb8242d37538b29f0765`

Reviewed evidence/status tip before this report:
`83bf2cb976355f39757bbe0b0953e52dc6814cf5`

## Verdict

**Accepted for the complete declared local Linux Milestone H scope.** The fresh
adversarial review found no unresolved high- or medium-severity issue in the
semantic-foundation CPU beta candidate, its exact local evidence, package and
ABI boundary, or its bounded product claims.

This acceptance authorizes the candidate to proceed to a normal pull request
and hosted gates. It does not claim hosted Linux or Windows validation, merge,
tag, publication, native-BLAS parity, executable recovered/map semantics, a
public API/ABI/backend-contract freeze, or accelerator support.

## Independent review method

The review inspected the full `origin/main...6796fd8` product diff and the
docs-only evidence commits through `83bf2cb`. Separate reviewers challenged
semantic/runtime behavior and evidence/package/CI truth before comparing their
conclusions. Review activity was read-only except for this acceptance report
and the narrow status updates that record its result.

The independent review did not rebuild the product. It inspected the immutable
validation clone and raw logs under:

```text
/home/hamza-usta/.cache/mdslc-h-matrix-6796fd8-VB03Wo/
/home/hamza-usta/.cache/mdslc-h-logs-6796fd8-aM084y/
```

The detached source remained exactly `6796fd8`, clean, and free of untracked
files. Independent static checks at the documentation tip passed full-range
`git diff --check` and `tests/check_repository_hygiene.sh`.

## Exact local evidence accepted

- six complete Release/Debug configurations passed 368/368 registered tests:
  R1 63/63, R2 63/63, R3 58/58, R4 58/58, C1 63/63, and D1 63/63;
- the Release grid covered Matcore MLIR enabled/disabled crossed with OpenBLAS
  required/disabled, plus MLIR-present/default-`capture-v0` compatibility;
- ASan+UBSan passed its exact 20/20 in-process scope;
- TSan passed its exact 4/4 persistent-runtime scope;
- the legacy Python frontend contract passed 14/14;
- the fresh R1 install, strict C17 consumer, external CMake consumer,
  source-inaccessible consumer, and source-inaccessible safety test passed;
- the shared runtime retained 15 public C exports and SONAME
  `libmatcore_runtime.so.0`;
- installed artifacts exposed no private MLIR target, library, development
  path, or absolute producer build/source path;
- the source-inaccessible package test authenticated configure-time clean HEAD
  `6796fd8`, cloned that exact commit, and failed closed against later HEAD or
  worktree drift; and
- the guarded 256-cubed OpenBLAS run passed correctness and timer/provenance
  sanity. Its timing is not accepted as native-parity or calibration evidence.

Compiler caching remained enabled and input-keyed. The evidence uses fresh
build, link, install, and test roots; it does not claim a zero-hit cache.

## Semantic and execution boundaries

The executable semantic path is real: authenticated explicit capture enters
typed Matcore IR v1, crosses the strict v1-to-Matcore-MLIR bridge, passes the
closed CPU lowering verifier, produces the private runtime-dispatch backend,
and executes through the retained stable C runtime symbol. Failure in this
path is terminal rather than a fallback to the capture-v0 backend.

Recovered ordinary-C++ GEMM remains analysis-only. The CPU lowerer rejects its
producer/envelope, so recognition cannot silently replace ordinary C++.
Map/domain/sine composition also remains non-executable. Its trusted source
snapshot authenticates bytes and ranges, not C++ semantic recognition or
numerical permission; no production caller of that trusted composition entry
exists yet.

Composition verification now requires unique canonical site IDs, canonical
symbol/site binding, and contiguous capture ordinals in module order. Shape,
dtype, accumulation, numerical profile, effects, aliases, mutability, domains,
source provenance, and SSA use-def constraints fail closed. The current map
model does not authorize a correctly-rounded sine lowering or any fusion claim.

## Runtime, provider, and allocation boundary

The public C ABI retained all existing exported signatures and layouts. Status
26 is additive; private planner/resource fields and OpenBLAS observers do not
cross the shared-library boundary.

Non-context v2 forced reference, tiled, compiler-vectorized, native-packed, and
prepacked-B paths do not inspect or execute OpenBLAS conformance. Their reports
distinguish build linkage from deliberately uninspected conformance. Automatic
and forced-provider requests include the process-once conformance boundary.
Default resource records do not imply authentication from linkage alone.

Execution-context creation remains an explicit all-variant validation
boundary. A conformant linked provider can run one process-once conformance
GEMM and two finite/special provider validation GEMMs for each context. Later
v3 reports reuse that context's authenticated evidence rather than discarding
it on forced-native requests.

The one-shot and workspace APIs promise no MDSLC-owned packing/workspace
allocation or tensor copy, not that an opaque linked provider can never manage
internal memory. Public comments and product documentation now state where
provider metadata, conformance, validation, or execution may initialize such
state. Prepacked-B ownership, identity, invalidation, and serial-reuse rules
remain unchanged.

## Package, Windows, CI, and performance truth

The installed package exports the compiler, runtime, inspection tools, public C
and eDSL headers, and relocatable CMake targets without exporting private MLIR
implementation libraries or headers. The package default fails closed when the
requested semantic pipeline was not compiled.

Windows remains a compatibility gate, not a local claim. The candidate keeps
Windows MLIR disabled with default `capture-v0`; no Windows Matcore-MLIR,
OpenBLAS, NUMA, or new performance claim is made. Hosted Linux and Windows jobs
must still run on the pull-request head, including native artifact, DLL/import
library, installation, external consumer, paths-with-spaces, and ZIP checks.

Milestone 7 remains partial and open. The single local sanity sample selected
OpenBLAS and cannot establish native/OpenBLAS parity, planner regret, or a
general throughput result. Cooperative packed-B preparation remains dormant.
No benchmark envelope or provider comparison was changed to manufacture a beta
claim.

## Findings resolved during final review

The review initially rejected intermediate tips for the following confirmed
medium findings; each was fixed and re-reviewed:

1. composition-v1 accepted duplicate semantic site identities and did not
   enforce deterministic root order;
2. allocation wording incorrectly covered opaque provider-managed state;
3. provider conformance and metadata were touched by forced non-provider and
   prepacked-B paths;
4. linked-but-uninspected, linked-nonconformant, and unlinked provider states
   produced misleading candidate diagnostics;
5. context-backed v3 planning temporarily discarded already authenticated
   provider evidence;
6. deferred conformance under an incompatible FP environment was mislabeled as
   an evaluated failure;
7. new provider-conformance evidence fields initially defaulted to true instead
   of failing closed; and
8. the source-inaccessible package test could configure at one commit and clone
   a later live HEAD, mixing product evidence.

Documentation-hygiene and stale-evidence findings were also corrected. The
historical `69d099e` matrix is no longer presented as proof for product changes
through `6796fd8`.

## Accepted low-risk follow-ups

- The clean-process OpenBLAS observer directly proves the internal exclusion
  boundary; individual public forced/prepacked entry points are additionally
  supported by static call-site inspection rather than a test-only exported
  observer.
- Before source-authenticated map/sine composition gains a production caller,
  a trusted Clang/Sema producer must independently prove recognition,
  replacement permission, and numerical legality.
- Generated MLIR code emits benign unused-parameter warnings in current local
  builds; this is warning debt, not a failed correctness or package gate.

## Remaining acceptance gates

The local candidate is accepted. Milestone H and the CPU beta are not complete
until the exact pull-request head passes hosted Linux and Windows checks, the PR
is reviewed and merged normally, and the intended beta tag/publication action
is explicitly authorized. Milestone 7 and the separate public
API/ABI/backend-contract freeze remain outside this acceptance.
