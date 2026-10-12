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
