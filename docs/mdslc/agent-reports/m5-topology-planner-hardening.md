# Milestone 5 topology/planner hardening report

## Ownership

This lane changed only the CPU topology record/helper, planner v3, their focused
tests, and this report. It did not edit the CLI, public C ABI, runtime dispatch,
or benchmark implementation.

## Implemented

- Added `restrict_cpu_topology_v1`, a deterministic projection onto an allowed
  logical-CPU set. It rejects empty, duplicate, unavailable, and incomplete
  inputs; intersects NUMA/cache groups; drops empty groups; and recomputes SMT
  thread indices so a permitted secondary sibling remains usable.
- Added explicit `CpuPlannerPlacementEvidenceV1`. Parallel variants now require
  complete placement evidence and fail closed when requested affinity was not
  applied, selected node IDs are inconsistent, a single-node policy crosses
  nodes, or cross-node placement omits caller-owned first-touch evidence.
- Single-node planning is bounded by injected local physical/logical capacity.
  Explicit cross-node candidates receive a deterministic conservative static
  cost penalty; this is not a measured universal NUMA-performance claim.
- A count-only processor override no longer authorizes parallel execution.
  Callers must restrict the complete topology first so core and NUMA membership
  remain known.
- Planner diagnostics now serialize thread/placement facts plus priority,
  workspace, runtime-validation status, and separate hardware, OS, compiler,
  and implementation requirement masks for legal and rejected candidates.

## Integration boundary

Callers should:

1. discover the complete platform topology;
2. call `restrict_cpu_topology_v1` with the current allowed CPU IDs;
3. select and apply placement using the restricted topology;
4. populate `CpuPlannerPlacementEvidenceV1`; and
5. pass it after the request argument (or after the legacy count override in
   the capability/topology overload).

Parallel variants remain unavailable when that evidence cannot be established;
serial candidates are unaffected.

## Validation

- Release full build: passed (`-j2`).
- Release focused tests: `platform.cpu_topology.v1` and
  `planner.cpu.v3.advanced`, 2/2 passed.
- Debug focused build/tests: 2/2 passed.
- Full Release CTest: 37/38 passed. The sole failure was the pre-existing
  integration DSO-export allowlist, which has not yet been updated for the
  already-added typed BF16/I8 runtime exports; all 62 behavioral integration
  checks passed. This lane neither adds nor changes a runtime export.
- `git diff --check`: passed.

## Commits

- `67aff07103594652b9c9f6322c8abe9a4bdc1845` — topology restriction.
- `18a9c3a3762bf1cfd97c80b584cfb8b6df2ee9e3` — placement-aware planner.
