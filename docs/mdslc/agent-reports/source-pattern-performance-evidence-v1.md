# P1 source-pattern evidence harness — implementation checkpoint

Scope: Issue #98, isolated worktree
`/home/hamza-usta/mdslc-work/correctness-campaign-v1/performance`, branch
`mdslc/source-pattern-performance-evidence-v1`, starting canonical
`b2597bed52de46604574f76eb33663788a5ab994`. AGENTS.md and the reviewed P1 roadmap
were read before implementation. No production compiler/runtime/API or CMake
configuration was changed. All implementation files are under
`compiler/benchmarks/source_pattern/`.

## Status and boundaries

This checkpoint contains the valid-C++ `.mdsl` host, frozen plan/receipt checks,
preparation/collection/offline-audit runner, README and synthetic tests.
**The C++ host has not been compiled; no source executable, installation step,
compiler build or benchmark has been run in this lane.** Integration coordinates
those only after its independently frozen correctness gates. No timings or
passing performance claim exist here.

The primary interval is call-return + Result move with retained observations;
retained returned Result/Observation ownership release and full output validation
are after the stop clock. Internal source-call snapshot/intermediate cleanup is
part of the synchronous invocation and remains inside timing.
Every timed output is checked, not just final C or one checksum. The six M/K/N
cells, exact repetitions, warmups/samples/floor, global forward/reverse process
order, initial/final controls, memory/disk/time caps and noise/spread predicates
are frozen in `contract.py` and documented in the README. There is no adaptive
repetition/floor, winner policy, resume or default/candidate/provider change.

Preparation consumes an already relocated, coherent uninstrumented installation
and compiles the same timed source plus the canonical 68-case source control
for stock/3x5/5x3. Existing source-pattern inspection is reused, with all eight
issued stage/LLVM/object/issuer identities joined from each control to its actual
timed leaf. Actual runtime/executable mappings and loaded-file identities are
separate from NEEDED names. Stock schedule classification requires producer
manifest/cache/object/link evidence, not an unrequested-flag assumption.

CMake normally removes stock build RUNPATH during installation. Root identified
that a direct raw DSO equality gate would reject this honest path. The narrow
inspection-only fix uses standard pinned CMake RPATH_REMOVE on a new nonexecuted
copy and requires exact installed-byte equality afterward; raw producer/SDK
inputs remain unchanged. No custom ELF normalization or runtime code import was
introduced. Root supplies the frozen producer catalog and installation later.

## Independent design/source review incorporated

Root and the issuer/adversarial lanes challenged the design before compilation.
Material findings incorporated in this checkpoint include:

- A complete source-call Result/Observation lifetime label, fixed four-A phase,
  retained-payload/transient/metadata bound distinct from process peak RSS and
  generated-artifact/raw-log disk caps.
- Exact per-result verification and independent bounded integer prefixes;
  nondefault-caller RNE/overflow/invalid arithmetic plus FP flags/errno checks,
  rather than exact integer data that would miss removal of the FP scope.
- Independent backing capacity and exact error categories for shape/capacity
  refusal controls, plus untouched aliased-input cells beyond the publication.
- No public Result thread telemetry fabrication: a named single-thread source
  implementation contract with bounded OS observations, not continuous proof.
- All loader/allocator/provider environment controls sanitized; driver/compiler/
  linker/transitive identities pinned, and per-process active modules audited
  without hashing the entire SDK before every timed child.
- A deadline spanning initial/final controls and final identity/catalog checks;
  failed/noisy/partial campaigns cannot become complete by extending the budget.
- Forward/reverse crossover and within-direction spread remain visible, with
  predeclared inconclusive criteria and no automatic selection.
- Offline summaries revalidate raw stdout/exit/argv/catalog and recompute timings;
  changing only normalized values, dropping raw entries or relabeling incomplete
  data cannot silently manufacture a passing aggregate.

## Validation at this checkpoint

`python3 -B compiler/benchmarks/source_pattern/tests.py` passes 14 unittest
methods using synthetic receipts, toy nonexecuted files and subprocess mocks.
They cover immutable plan/order/lifetime, strict JSON/types, middle-result/hash/
count mutations, floor/noise refusal, reverse crossover, loader/environment
controls, deadline refusal, external-Git/no-clobber paths, all eight leaf joins,
toy standard-RPATH-transform joins and non-RPATH corruption, and a complete
synthetic collection -> JSON -> raw-audited summary round trip with derivation/
catalog/deadline/final-control corruptions. This is not C++ or physical evidence.
`git diff --check` passes.

Pending: final frozen-commit independent review; coordinated actual `.mdsl`
compilation/SDK/profile and source/artifact/control checks; then, only on the
declared exclusive host, a bounded collection. Any future negative, noisy,
inconclusive or unsupported result must be retained without weakened gates.
