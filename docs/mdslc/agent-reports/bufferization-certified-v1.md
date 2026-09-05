# Certified GEMM bufferization handoff v1

Date: 2026-09-05

Status: stacked candidate on `mdslc/bufferization-certified-integration-v1`.
This is an
inspection-only proof boundary. It does not authorize generated execution,
claim a general zero-copy property, or change the existing CPU
runtime/provider route.

## Checkpoints and evidence identity

- Canonical campaign base:
  `327530d287e41c4115365598e76b17e149a1c45a`
- Reviewed contraction/certificate foundation head:
  `1b7bc9fa212894e330777c18c918c7c533d05c4b`
- Canonical contraction/certificate foundation merge:
  `5983c2c1bf067bed9e69d0172b0944b7a4c14c00`
- Fresh operation-neutral derived-identity commit:
  `71764e47d3b89c3f3621fad90be4bf7dac979540`
- Fresh certified bufferization implementation commit:
  `ea3202ff144c5ec4b7ae6dcbc2cf77a5e2045365`
- Current canonical main to be incorporated by a normal merge before final
  validation (provider-evidence PR #23):
  `ec7a55a61c0ceb2ee4985211bbf887ccb3b1368e`
- Preserved historical original bufferization implementation:
  `12187a8023906a1c758800ed5d2e1346549155c0`
- Preserved historical reconciled implementation:
  `e47b2e70726a6253874143c0a7f5b5797b2e371e`
- Preserved historical proof/report head:
  `1efa342737cc623bd7e7174c81552a604bada366`

The survivor is cleanly stacked from canonical foundation merge `5983c2c` as
the shared derived-identity seam `71764e4`, then buffer-specific implementation
and tests `ea3202f`, then this durable record. The historical branch is retained
unchanged as proof of the earlier reconciliation. Current canonical main
`ec7a55a` is incorporated only after these focused commits by a normal merge;
the provider-evidence paths do not alter the certified bufferization contract.

The inspected compiler-archaeology corpus was the detached research tree at
`8aab0e295eb41ca5d2e1bd52c47201b05de9636b`. Its manifests were:

- `corpus/manifests/windows-corpus-v1.json`, SHA-256
  `8d5d3e6d6af89fd78d3af5bd53b530a96e55920cbda65c38d30e7646cd01e187`;
- `corpus/manifests/windows-corpus-v2-archaeology.json`, SHA-256
  `68435d26abc4e498a76432290446bddfd428a64b81e59fe4640218016d3c754d`.

This lane used the pinned Ubuntu LLVM/MLIR 21.1.8 package extraction at
`/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`.
Observed identities were:

| Artifact | Identity |
| --- | --- |
| `mlir-opt --version` | `Ubuntu LLVM version 21.1.8`, optimized build |
| `bin/mlir-opt` | SHA-256 `18c322575dc399010d8186ff72cbf10047ff7ce99d8e11aec162f33820c89ed3` |
| `OneShotAnalysis.h` | SHA-256 `67a08385c3ca9d1222b13f272a0cf119c8d9120a220c8a23b8ee943201bc8da5` |
| `Passes.td` | SHA-256 `ccf1cc05e4831616222378e1f53d2ea29d80370e5e3bdb6cb7300a34cb3b7697` |

No system package or toolchain was changed.

## Evidence reconciliation

**OBSERVED:** Upstream One-Shot Module Bufferize transforms each exact
structured GEMM specimen with these options:

```text
allowUnknownOps = false
bufferizeFunctionBoundaries = true
copyBeforeWrite = false
functionBoundaryTypeConversion = IdentityLayoutMap
```

The dynamic function has this buffer-level dataflow:

```text
func.func(...,
          %A: memref<?x?xf32>,
          %B: memref<?x?xf32>,
          %C: memref<?x?xf32>) -> memref<?x?xf32> {
  %zero = arith.constant +0.0 : f32
  linalg.fill ins(%zero) outs(%C)
  linalg.matmul ins(%A, %B) outs(%C)
  func.return %C
}
```

The static non-square specimen has the same body with
`memref<2x3xf32>`, `memref<3x4xf32>`, and `memref<2x4xf32>`. A two-site
specimen containing that static geometry followed by an independent dynamic
geometry also satisfies the certificate. In every admitted case, pass
statistics report zero buffer allocations, zero deallocations, and zero
out-of-place tensor decisions. The exact four-operation verifier additionally
excludes `memref.copy`, `bufferization.clone`, casts, and tensor/buffer bridges
from each certified function.

**OBSERVED NEGATIVE:** Disabling function-boundary bufferization for the same
structured body preserves a tensor function boundary and creates
`memref.alloc`, `bufferization.to_buffer`, and `bufferization.to_tensor`. It
does not prove caller-output memref identity.

**OBSERVED NEGATIVE:** Enabling `copyBeforeWrite` emits an allocation and
`memref.copy`, and returns the new allocation rather than argument 2.

**OBSERVED NEGATIVE:** Adding a generic-MLIR-valid read of original C after the
contraction creates a read-after-write conflict. MLIR 21.1.8 allocates an
out-of-place result and returns it rather than C.

These negative controls falsify the strong statement in
`corpus/findings/upstream-mlir-findings.md` that DPS plus materialization
guarantees in-place output without allocations or copies. They support the
conditional statement in `corpus/findings/adversarial-audit.md`: DPS is a
useful structural basis, but allocation, copy, and identity are properties of
the exact program and options and must be checked after transformation.

**ARCHITECTURAL IMPLICATION:** MLIR owns bufferization analysis and the
tensor-to-memref transformation. Matcore owns admission from its authenticated
structured schema and checks the postconditions needed to preserve Matcore
meaning. Neither layer may infer runtime noalias, alignment, residency, or
physical storage facts from the successful transformation.

## Reconciled implementation

The original `MatcoreBufferizedGemmHandoff` remains the buffer-specific layer.
It owns the exact One-Shot options, identity-layout memref envelope,
allocation/copy rejection, fill/matmul/return dataflow, and the
`matcore-bufferized-gemm-handoff-v1` ledger.

The reconciliation removes its parallel reconstructed-provenance path and
extends `MatcoreStructuredHandoffCertificate` with reusable downstream-source
identity:

- every derived site retains the exact structured tensor function type and a
  deterministic `matcore-structured-semantic-fingerprint-v1` digest;
- the digest binds capture/source module identity, ordinal, source operation,
  semantic symbol, site, original function type, function location, ordered
  block-argument locations, and the complete opaque semantic contract;
- a module digest binds the ordered, complete set of site digests and count;
- standalone verification recomputes the retained per-site and aggregate
  digests and therefore proves internal self-consistency only;
- paired verification recomputes them from a supplied verified structured
  source and rejects source substitution, site reorder, drop, duplication, or
  cross-site substitution.

The digest is a deterministic substitution detector. It is not cryptographic
authentication of imported bytes, a public serialized contract, or execution
authority. The reusable helper checks identity envelopes only. Its
caller must still authenticate the operation-specific source body and verify
operation-specific derived postconditions.

`MatcoreStructuredGemmHandoff` now exposes its internal certificate profile
and an operation-specific retained-contract verifier. The latter constructs a
temporary tensor-typed `mdsl.gemm` witness with the retained function and
argument locations and invokes the authoritative dialect verifier. Buffer code
therefore no longer interprets GEMM's 13-field source contract independently.

```text
authenticated mdsl.gemm semantics
              |
              v
verified structured GEMM + generic certificate
              |
              | exact upstream One-Shot options
              v
derived-source identity certificate
  - original tensor type
  - site fingerprint
  - ordered module/site-set fingerprint
              |
              +--> retained mdsl.gemm contract reverified by GEMM owner
              |
              +--> buffer-only postconditions verified by buffer owner
              v
inspection-only identity-layout memref specimen
```

No public operation, tensor/view type, API, ABI, serialized interchange
promise, schedule, or runtime route was added.

## Mechanical invariants

The derivation plus exact-source paired verification proves all of the
following; standalone verification proves only the internally checkable
derived-envelope subset:

1. The derivation API admitted an exact verified structured-GEMM-v1 module
   before the pass, and the paired verifier binds the result back to that
   supplied source.
2. The derived module remains `analysis_only` with `inspection_only` authority.
3. Its producer chain, source file and translation unit, semantic version,
   site/symbol/ordinal, function location, and ordered argument locations are
   bound by the retained structured fingerprint.
4. Its original structured function type is exactly three rank-2 f32 tensors
   and one destination-tied rank-2 f32 result accepted by `mdsl.gemm`.
5. Its three arguments and result are exact default-memory-space,
   identity-layout rank-2 f32 memrefs whose shapes match the retained tensors.
6. The result type equals output argument 2's type, and `func.return` returns
   that exact block argument.
7. Exact positive f32 zero is filled into argument 2 before the contraction.
8. Canonical `linalg.matmul` reads arguments 0 and 1 and writes argument 2.
9. The operation order is exactly constant, fill, matmul, return. No allocation,
   deallocation, copy, cast, tensor/buffer bridge, call, or unrelated operation
   can hide in the admitted function or its exact scalar regions.
10. Matmul retains canonical `(m,k),(k,n),(m,n)` maps, parallel M/N plus
    reduction K iterators, exact `mul` -> `add` -> `yield` scalar dataflow, and
    no fast-math flags.
11. The original semantic contract still passes the authoritative GEMM
    verifier; no buffer-specific redefinition substitutes for that check.
12. Standalone certificate self-consistency and exact-source pairing are named
    and checked as different claims.

## Consumed, preserved, and unresolved facts

| Fact | Disposition at this boundary |
| --- | --- |
| rank, f32 dtype, static extents, dynamic positions | encoded in memref types and checked against retained tensor types |
| row-major contiguous layout | encoded as identity function-boundary layout for the admitted specimen |
| initial C value | consumed: full positive-zero fill precedes the only contraction accumulator read |
| destination/result identity | consumed: fill, matmul, and return use exact output argument 2 SSA identity |
| allocation/copy behavior | checked postcondition: none inside each admitted function under the exact options |
| effects | retained and checked against the exact read/write body |
| numerical profile and K-reduction permission | retained and scalar-region checked, but unconsumed by bufferization |
| dynamic M/N/K equality and positivity | retained requirement; no concrete runtime values or guards exist here |
| output-vs-LHS/RHS noalias | retained required precondition; no optimizer noalias fact is created |
| minimum alignment | retained required precondition; no memref/LLVM alignment fact is created |
| host memory-space and CPU policy | retained semantic contract; default memref space is not physical residency proof |
| external source pairing | proven only by the paired verifier; standalone verification proves self-consistency |
| runtime execution authority | unconsumed and forbidden; artifact remains inspection-only |

The ledger value
`retained_unconsumed_scalar_region_checked` deliberately avoids implying that
bufferization consumed numerical permissions. Any future tiling/vector or
lowering step must independently prove that its reduction order and
contractions are permitted.

## Falsification coverage

The focused suite includes dynamic, static non-square, mixed static/dynamic
multi-site, source immutability, cross-context print/parse, and exact-source
positive cases. Generic-MLIR-valid mutations reject:

- execution-authority escalation;
- module translation-unit provenance drift;
- forged noalias, alignment, or runtime-guard consumption;
- removed alias requirements or altered numerical reassociation permission;
- retained source tensor shape drift;
- non-default memory space or nonidentity memref layout;
- wrong result identity, wrong fill target, missing fill, wrong contraction
  destination, or swapped inputs;
- function-location or block-argument-location drift;
- same-shaped alternate source pairing;
- reordered, dropped, duplicated, or cross-site-substituted functions;
- a forged ordered site-set digest.

The no-function-boundary, conflicting-use, and copy-before-write controls run
upstream MLIR successfully and then prove why their resulting IR cannot satisfy
this certificate.

## Exact validation

At the patch-equivalent historical implementation commit
`e47b2e70726a6253874143c0a7f5b5797b2e371e`, CMake was refreshed with Release,
Clang/Clang++ 21, native and bootstrap frontends enabled, Matcore MLIR enabled,
the exact MLIR 21.1.8 package, OpenBLAS disabled, and testing enabled.

| Validation | Exact result |
| --- | --- |
| Complete incremental Release build after clean-head reconfigure | `114/114` steps completed; only pre-existing generated ODS unused-parameter warnings |
| Full standalone CTest | `66/66` passed, `0` failed, 205.21 s |
| Focused structured + bufferized CTest | `2/2` passed, `0` failed |
| `matcore_mlir_bufferized_gemm_handoff_tests` | `274` checks, `0` failures |
| `matcore_mlir_structured_gemm_handoff_tests` | `366` checks, `0` failures |
| `matcore_mlir_contraction_model_tests` | `277` checks, `0` failures |
| `git diff --check` | passed |

The full surface includes frontend, semantic IR/MLIR, structured handoff, CPU
runtime/planner, C ABI, installed consumer, source-inaccessible package,
benchmark-contract, and CLI tests. It does not constitute Debug, sanitizer,
Windows, OpenBLAS-enabled, performance, parity, or generated-code execution
validation.

## Independent foundation review

Independent read-only review of exact foundation head `1b7bc9f` is **GO** for
canonical integration, subject to its already stated inspection-only scope:

- operation identity is not erased by the topology substrate: GEMM, GEMV, DOT,
  GER, and aligned-batch GEMM remain distinct standard identities; GER is an
  outer-product update without a reduction, and unit/degenerate geometry does
  not silently reclassify GEMM;
- topology contains ranks, logical loops, affine maps, orientations, and
  iterator classes only. It does not absorb extents, layouts, numerical rules,
  destination behavior, scheduling, or authority;
- the generic certificate binds envelopes and opaque contracts but expressly
  leaves operation admission and body semantics to operation-specific
  verifiers;
- the profile admits only `inspection_only`, and model-only GEMV/DOT/GER/batch
  carriers receive neither authenticated source nor runtime authority;
- final function and ordered argument-location anchors close the observed
  provenance substitution gap.

No blocking correctness or ownership finding survived review. The important
usage constraint is not a defect: neither the topology verifier nor generic
certificate alone is an operation semantic verifier. Downstream code must pair
it with the relevant operation-specific verifier, as this bufferization branch
now does.

## Claim boundary and next justified step

This checkpoint does not prove general zero-copy behavior, concrete runtime
noalias/alignment/residency, ownership outside the function, a callable memref
ABI, generated execution correctness, target lowering, vectorization,
portability to another MLIR version, performance, Native BLAS parity, or
provider policy. The authenticated CPU runtime/provider route remains the only
executable GEMM path. Issue #15 is unchanged.

The immediate integration action is to merge current canonical main into this
already foundation-based survivor, independently review the combined diff, and
keep it as a separate inspection-only checkpoint. A later callable-boundary
milestone would first need explicit runtime descriptor-to-memref ABI ownership,
dominating dynamic-shape/noalias/alignment/residency guards, and an
execution-authority decision. This branch deliberately does not begin that
work.
