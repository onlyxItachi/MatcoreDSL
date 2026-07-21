# ADR-0004: Consolidate MDSLC and legacy history on `main`

- Status: accepted
- Date: 2026-07-21

## Context

The repository had two long-lived lines descending from
`1d5072306513d566b03879ccfb7890517ae8ffc9`:

- `main`, with 45 commits not present on the historical default branch; and
- `feature/device-resident-tensors`, with 37 commits not present on `main`.

The standalone MDSLC compiler was developed additively from the feature line.
Its final reviewed CPU planner checkpoint is
`58d3e7c75eadf5a71416870ac4c99b4829db8566`. Keeping this work indefinitely
off `main` made the supported repository entry point ambiguous and made later
legacy/compiler integration harder.

## Decision

Preserve both histories with ordinary merge commits and make `main` the
integration destination. Do not rebase, squash, force-push, or delete the
historical branches as part of consolidation.

The integration order is:

1. start from the fetched `origin/main` commit;
2. merge the complete native MDSLC lineage while preferring the current
   feature/MDSLC side for overlapping source;
3. merge the final Matcore IR v1 and CPU planner review lineage;
4. compile the combined legacy surface to expose semantic auto-merge defects;
5. repair only demonstrated compatibility defects in focused commits; and
6. validate the standalone compiler and meaningful legacy harnesses before a
   pull request is eligible to merge.

The final `compiler/` tree remains byte-identical to the reviewed Milestone 2
checkpoint. Legacy implementation remains in history and in the combined
tree, but it is not allowed to displace or silently weaken the standalone
MDSLC contracts.

## Compatibility policy

- MDSLC remains additive and Python-free in its normal compiler/runtime path.
- The valid-C++ `.mdsl` source contract, native Clang frontend, typed Matcore
  IR v1, deterministic CPU planner, generated C ABI path, install package, and
  external consumer remain protected acceptance surfaces.
- Existing legacy Python/JIT behavior is retained when it can be reconciled
  without weakening those surfaces.
- In an overlap, prefer the more recently validated feature/MDSLC behavior and
  retain compatible hardening from `main` only after compilation or focused
  tests demonstrate that the combined semantics are sound.
- Future deletion or redesign of legacy code requires a separate reviewable
  commit or pull request. Consolidation itself does not erase that history.
- Accelerator behavior is not inferred from compilation. Only executed tests
  may be described as runtime-validated.

## Consequences

The repository gains one reviewable integration line containing all relevant
history. The historical feature and MDSLC refs can remain as immutable
archaeology/checkpoint references without remaining the repository default.

The legacy root build still requests MLIR 18.1.3 while the machine's restored
development package is MLIR 22.1.2. The combined source was compiled and
tested against a temporarily installed coherent MLIR 18 development surface,
then the pre-existing MLIR 22 package state was restored. This dependency
policy remains a separate legacy concern and is not coupled into the MDSLC
Clang 21 build.

