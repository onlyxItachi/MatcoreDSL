# Independent reassociate issuer controls

These fixtures are independently authored against issuer `1c03306` and its
test-only classifier correction `62379dd`. They do not change production code.
The integration owner should add `add_subdirectory(reassociate_independent)`
inside the existing generated CPU test directory, after the issuer target exists.
The included CMake file registers two tests.

`mutations.cpp` checks 47 assertions with structurally valid MLIR/LLVM mutants.
It explicitly separates narrow-envelope checks from exact-replay checks, checks
wrapper data-pointer binding and non-invented pointer guarantees, and preserves
the permitted diagnostic-location/module-comment differences. The deliberate
renamed-buffer positive is self-consistency, never source authentication.

`classifier_test.py ../expect_reassociate.sh` has 14 synthetic controls. Ten
initial refusal cases, the later-allocation-stack owner adversary, two valid
widths and a preserved skip-77 case test classification only. No synthetic text
is presented as evidence that generated instructions actually executed. The
allocation-stack case fails against the old `1c03306` classifier and passes
against `62379dd` without weakening its expected outcome.

The independent report records real generated-leaf execution separately from
these mutations and synthetic diagnostic controls. No generated artifacts or
logs are committed.
