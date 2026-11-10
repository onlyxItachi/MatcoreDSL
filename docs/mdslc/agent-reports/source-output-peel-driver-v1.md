# Explicit source output-peel driver lane

Status: implementation/source review only. Based on frozen
`dfc35f9e13653db16c7f9a825db2bb1b2a82ea22`; no configure, compilation, test,
execution or timing was performed by this lane. Root owns those acceptance
gates. Private-issuer qualification does not itself qualify source execution.

## Smallest reused seam

`compiler/tools/mdslc-region/SourcePattern.h` adds the closed
`SourcePatternRealizationV1::{OutputTiledMKN,OutputTiledPeelNMKN}` selector as
the trailing `OutputTileRequest::realization` field. Existing `{m,n}` requests
and parser/validation defaults still select the original realization.

`SourcePatternInstance` in `SourcePattern.cpp` selects the existing
`issueStrictGemmArtifactV1` schedule. Everything after that selection is shared:
stage capture, pinned native-baseline PIC compilation, leaf import/export
checks, exact private archive relink, intrinsic DSO ownership, immutable
source/compiler binding and final no-clobber publication. No second pipeline,
new runtime candidate, callback, artifact importer, cache, registry or public
API was introduced. The original WHAT/Program witness is not transformed.

`main.cpp` accepts the separately explicit
`--gemm-output-tiles-peel-n M,N`. Both source flags share one optional request;
duplicates and mixed flags in either order refuse during parsing. Both require
exactly one explicit `--candidate generated-strict`. `program_main.cpp`
explicitly refuses both flags in program mode. The existing `-c`, candidate,
strict-fused-pair and bounded-host-option exclusions remain in force.

## Identity and authority

The old route's manifest construction, field set and binding domain remain
unchanged in the code. The peeled route uses only these deliberate differences:

- `schema=matcore-source-output-peel-n-v1`;
- one additional field, `realization=output-tiled-peel-n-mkn`;
- binding domain `matcore-source-output-peel-n-binding-v1`.

Both retain the existing issuer-manifest/stage/object/DSO identities and
request-derived private SONAME. Unit-N may produce identical unpeeled kernel
stages/object bytes; that is not a failed specialization. Its requested
realization and issuer manifest still distinguish the source binding. The new
outer schema additionally makes the frozen old manifest parser fail closed,
without editing the P1 harness, plan or historical evidence. The complete P1
plan already fixes the old flag and 3x5/5x3 parameters; this is not a claim of an
existing P1 bypass.

No extra issuer-source hash schema is needed for authority. The issuer is linked
into the compiler-owned driver, not supplied by a source/manifest reader.
Existing `installation_identity.cmake` pins the exact private PIC kit, export
script, Clang/linker and owning DSO/Runtime images; `Installation` and host
capture retain their existing concrete source/header/toolchain closure checks.
The same kit omits exactly the same baseline strict leaf symbol, so a second
peeled archive or package profile is unnecessary. Sanitizer/worker profiles and
the private typed source/compiled-result seals are unchanged. These checks do
not claim hostile toolchain/build-origin attestation or self-authenticating
installed executables.

The primitive issuer still states
`source_authority=none_builtin_primitive_only`. Only its peeled inspection
wording changes to avoid a stale claim that the source route cannot exist:
`pattern_source_guard=caller_obligation_not_discharged_by_primitive_issuer` and
`pattern_source_selection=primitive_issuer_never_authorizes_source`. Neither
string grants source authority or discharges raw leaf preconditions; the
authenticated driver and unchanged checked adapter own that connection.

## Counterexamples and remaining gates

The closed selector is checked by an exhaustive switch before issuance;
cast/unknown enum values refuse instead of falling through to the old schedule.
Canonical decimal syntax and the existing 1..64 bounds are shared. Pending
root/test-owner gates include malformed parameters, both mixed-flag orders,
duplicate candidates, program/compile-only/fused/foreign-candidate refusals,
same-shape old/new stage and compiled-result cross-wires, stale-source donors,
and unit-N equal-kernel/different-binding checks. Existing source phase mutation,
kit/export/header substitution and final publication checks are reused, not
relaxed. The source runner must discriminate the requested full/tail recipe,
not infer it merely from correct output or an arbitrary manifest digest.

Source/installed/ASan/numerical qualification remains pending for this new
request. There is no automatic/default, GPU, nonbaseline-ISA, multi-TU,
source-shape restriction or numerical-permission expansion. No benchmark or
speedup result follows; P1 remains `INCONCLUSIVE_TIMING` with its retained
bounded loss observation unchanged.
