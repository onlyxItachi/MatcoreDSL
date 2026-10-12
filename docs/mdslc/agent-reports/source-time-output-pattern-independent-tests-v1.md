# Source-time output-pattern independent tests v1

Owned scope: new `compiler/tests/source_pattern/`, registration in
`compiler/tests/closed_driver/CMakeLists.txt`, and the existing
`installed_source_inaccessible.py` production-package acceptance extension.
Starting checkpoint: clean canonical main `3077d9c3f5df1479f2fcadfcceeb9444018d6ba0`.
Tests consume the bounded private source issuer from core `665bdb3` plus the
sanitizer-profile binding follow-up `3de498e`; they do not change its production
implementation, default/provider/GPU paths, or the independent SPIR-V frontier.

The four new CTest names are `driver.source_pattern.connected`,
`driver.source_pattern.refusals`, `driver.source_pattern.runner`, and
`driver.source_pattern.phase_mutations`. They cover two explicit opposite
rectangular tile requests through real public source compilation, independent
68-case strict/source-effect math, same-Value input aliasing, noncommuting RHS
carry, explicit 1x1/64x64 source-lifecycle boundary controls with 65x129 numerical
outputs,
forwarding/default coexistence, actual requested private DSO linkage,
post-capture substitution/closure and consumer-binding failures, and no-clobber
bundle behavior. The existing relocated producer-removed package gate is extended
with both requests, all private bundle and SDK hashes, and copied-header forgery.
Only the disposable relocated SDK temporarily hides its relink kit/exports:
stock compilation must still execute while a requested instance fails cleanly;
exact material is restored before the final SDK hash comparison.
See the focused [test contract](../../../compiler/tests/source_pattern/README.md).

The requested source-DSO ASan negative probe deliberately injects an invalid raw
descriptor outside the source caller contract. It demands exact local native ELF
wrapper/RX identity, canonical Runtime linkage, actual issued-leaf top-frame/DSO
attribution and a heap-buffer-overflow diagnostic. It does not create a public
importer or executable-IR authority. Host instrumentation is checked separately.

Agent-side validation at handoff is limited to independent Python classifier
tests, ordinary Clang source/probe syntax checks, mutation-test syntax with the
coherent LLVM/MLIR include tuple, CLI import/help, and repository diff hygiene.
The integration owner owns actual compiler builds, normal/sanitizer physical
source/phase execution, clean full-suite and relocated package execution. No
passing execution, hosted CI, ARM hardware or performance result is inferred.

Retained construction controls: an initial syntax-only mutation invocation lacked
the product MLIR include prefix and failed at `mlir/IR/BuiltinOps.h`; a second
invocation treated upstream headers as ordinary includes and failed their unused
parameters under `-Werror`. The corrected narrow check uses the coherent MLIR
21.1.8 prefix and SYSTEM includes, matching CMake. The source-context inspection
identity intentionally retains its existing `sha256:` prefix rather than
normalizing the production representation to satisfy a test.

## First physical integration feedback and follow-up

The integration owner reported the first Release run passing the refusal,
synthetic runner and deterministic phase tests. The connected test reached real
execution but failed its forwarding call discriminator: the original fixture's
late `D` is only MAY-alias with published `C`, so that transformation is not
legally eligible. The correction preserves that fixture's none/forwarding math
parity and adds a separate same-descriptor `forwarding.mdsl` positive with an
exact checked-call discriminator and requested-shape failure-prefix oracle.
This fixes the test premise; it does not weaken the production legality rule.

Core `171456c` adds typed source publication. New controls precreate all sources,
bundles and stages before admission and require specific source-pattern pairing/
digest refusals before stage IR/result changes. They cross-wire compiled A with a
different-source B and a same-source different-tile B, and modify otherwise
well-formed emitted LLVM bytes. Exact paired publication must separately execute.
These follow-up physical controls remain for the integration owner to run.
Core `e026f15` additionally owns Installation identity and the bundle-associated
output path. Tests refuse a byte-identical Installation copy and an unassociated
output, with exact diagnostics and unchanged empty publication stages. All three
legitimate source/tile instances first compile and validate their own pairing.

## Final independent classifier and refusal hardening

The integration owner reported the corrected four-test Release suite passing
in 135.64 seconds and the matching four-test ASan suite passing, including actual
requested-source DSO OOB attribution. Those are integration-reported physical
results, separate from this agent's syntax and classifier checks.

The final test-only increment checks consumed HOW structure independently:
exactly one non-comment canonical Transform tile instruction uses `[M,N,0]`,
and exactly two nested outer SCF loop step references resolve through preceding
integer-index constants to ordered `[M,N]`. The toolchain string must equal
`21.1.8`. Nine synthetic unit tests cover swapped tuple/step constants/references,
comment-only impostors, extra instructions/loops, missing step definitions,
malformed manifests/ELF and exact negative-dependency diagnostic ownership. An
already emitted 64x64 Transform/scheduled pair passed this narrow inspection;
that does not add an execution or optimizer-legality claim.

Mutation completion is now recorded only after the write/path replacement
finishes. Each phase failure must name the actual source/header/artifact path
and its expected captured-input, changed-artifact or preexisting-output category;
a hook exception cannot pass. The negative `__has_include` control specifically
requires its captured bundle-directory metadata-change diagnostic, which precedes
the generated file's formerly negative status in the current recheck order.
The final strengthened physical rerun remains integration-owned.
