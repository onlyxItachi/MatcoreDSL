# Independent strict generated CPU candidate review

Verdict: **ACCEPT within the closed primitive contract**, after the encoding
counterexample below was fixed. This reviewer authored neither issuer nor
lowering/verifier implementation. The independently authored numerical oracle
was subsequently retained as a separate test commit.

Reviewed implementation:
`a3548eec2bfa107de3a632d4a63fb9e8f3c994c7`, following independent-test commit
`2964e2fe8ea78bc73f139acb2856cec3258d479a` on
`mdslc/generated-strict-cpu-candidate-v1`, based on canonical
`fd64850a0c8cb7d2c0801a64dd94d515dd714130`. Worktree was clean at final review.
Full issuer, stage verification, CMake, tool, tests and implementation report
were inspected. Final issuer source SHA-256:
`3c5df7581a29217da431b321be1a34d005c0a006d5c8fd3e9b052f3690a3c403`.

## Independent falsification and execution

1. A well-formed structured module carrying an added tensor encoding
   `"forged_target_layout"` was originally accepted. The complete function,
   arguments and tensor results were changed consistently, so generic MLIR
   verification also passed. This contradicted the exact canonical primitive
   check. The implementation now rejects encoded tensor types, verifies exact
   intermediate/result types, and retains this counterexample in its tests.
   The closed issuer never accepted external stages, so this was a structural
   false accept, not an imported-IR execution bypass.
2. Nine independently authored stage mutations now reject: tensor encoding,
   nested fill-yield attributes, argument noalias claims, changed linkage and
   commuted reduction operands, across applicable tensor/buffer stages.
3. The independent generated-machine-code oracle passed **17,768 checks** over
   1,200 deterministic randomized geometries. It compares the actual generated
   object against a separate volatile-f32 increasing-K oracle, checks NaNs with
   unspecified payload and other values bitwise, covers zero/null-empty shapes,
   subnormal/infinite/signed-zero inputs, and checks output guard cells. The
   private C descriptor size, alignment and field offsets are static assertions.
   This test is retained as `compiler/tests/generated_cpu/independent_execution_test.cpp`.
4. The first standalone independent run linked the actual ASan-instrumented
   generated object and compiled its caller with ASan+UBSan: zero failures.
   At the final exact implementation checkpoint, this reviewer repeated all
   seven `generated_cpu` CTests: **7/7 Release, 0.13 s**, and **7/7 whole-issuer
   ASan+UBSan, 0.18 s**. These include normal/instrumented independent execution,
   object inspection and the deliberate generated-load heap-overflow control.
5. The issuer rejects `--input` and a requested alternative `--pipeline` with
   exit status 2. Its C++ issuance API takes only an MLIR context and sanitizer
   selection, not a source Program, imported MLIR, edited stage, or pass list.

The scratch structural harness and initial execution artifact remain under
`/tmp/mdslc-generated-candidate-review.RnXNgM/`; permanent tests, reviewed
source and the implementation report carry the durable evidence.

## Authority and numerical review

The semantic witness remains inspection-only. Execution authority belongs to
this compiler-owned primitive issuer, not to mutable hashes/attributes or a
self-consistency verifier. Canonical operand order, positive-zero overwrite,
destination identity, maps, f32 multiplication/addition and absence of added
permissions are checked before lowering. The LLVM output has the two expected
definitions and a closed call set. Ordinary object compilation may implement
zero fill through `memset`; this is not tensor allocation or a source-value copy.

Linalg's reduction iterator alone does not encode increasing-K order. Acceptance
is for the exact 21.1.8 scalar lowering pipeline and tested machine code, not
arbitrary tiling, vectorization, pass substitution or optimizer equivalence.
The leaf assumes compatible FP controls; host FP preservation remains the
adapter's responsibility. Generated LLVM obtains actual ASan function attributes;
UBSan in the issuer/caller is not claimed to add missing integer/storage guards
to arbitrary generated LLVM operations.

The identity-layout memref ABI explicitly requires offset zero and strides
`{columns,1}`: generated identity-layout code does not dynamically honor a
forged nonzero descriptor offset. Shape, byte/address, capacity, alignment,
lifetime and race predicates remain caller obligations. Inputs may overlap;
the destination must be isolated from both inputs. This is a private Linux x64
21.1.8 leaf, not Windows/accelerator support or a public ABI.

## Retained limitation and integration boundary

Independent disassembly found that zero N can still execute an outer M loop.
Very large M with an empty output therefore has impractical running time. The
region adapter must short-circuit an empty output before leaf dispatch. This is
an upstream scalar-loop optimization opportunity, not a reason to narrow valid
semantic shapes or encode target scheduling in semantic IR.

No source-to-generated-region execution, external provider coexistence,
publication/failure adapter, performance result or BLAS parity follows from this
review. Root integration must preserve the closed issuance boundary, run full
regressions and hosted CI, and connect the source and runtime predicates through
separately reviewed code. No production code was changed by this review.
