# Independent review: frozen original-host code generation

Reviewed checkpoint: `20f532e5a838fa0da3c6a530a7a42085949d2273`.
Reviewer: value/transaction advocate, not the utility's author.

**ACCEPT within the utility's bounded contract.** The utility compiles an
already-captured immutable original host TU to verified in-memory LLVM IR.
It does not admit mathematics, construct region evidence, authorize imported
IR, splice functions, select numerical implementations, or publish executables.

## Source review

All five changed files were read, including the complete implementation,
interface, test and CMake wiring. The supplied snapshot's unchanged check runs
before compilation and before the resulting LLVM string is produced. Compiler
queries use the frozen replay VFS, and its recorded failures are checked both
before and after invocation. There is no mutable-filesystem fallback or source
text substitution. Failed normal-return paths clear the caller's previous IR.

The exact recorded language/preprocessor/target context is retained. The code
sets Clang `DisableLLVMPasses` before invoking LLVM emission, takes ownership of
the module before the action dies, keeps its LLVMContext alive, verifies the
module and returns only text. No serialized or mathematical execution authority
is inferred from this compile utility.

The distinction from frontend admission remains important: disabling LLVM
passes cannot undo C++ constant evaluation performed by Clang. The separate
admission contract must reject constant-evaluable region entries and attempts
to derive mathematical semantics from the source-only placeholder's C++ object
representation. Nor does this utility itself prove a later replacement thunk's
ABI, attributes, source helper erasure or final output-publication discipline.

## Independent mechanical tests

The existing `frontend.frozen_host_codegen` passed when rerun independently:
ten checks, real original-host LLVM and ordinary-Clang executables, matching
source column/source_location/header constant/RAII, and dependency-mutation
refusal.

A separately authored throwaway executable added ten checks, all passing:

- frozen host capture, Sema and code generation succeed;
- an `always_inline` function call is still present as a call in the returned
  LLVM IR, proving that this actual boundary precedes even that inlining;
- both the emitted LLVM and the original C++ link and execute;
- `#line`, `__FILE__`, builtin line/source-location column, an ordinary function
  pointer call, a thrown/caught host exception and constructor/destructor effects
  produce identical output under direct Clang and frozen-host LLVM compilation;
- staging artifacts outside the captured search directory preserves the input
  closure;
- mutating the original main file refuses compilation and clears stale output.

The additional executable is ordinary Linux x64 execution against the reviewed
Release utility/archive, not an ASan or Windows runtime claim. Its source and
binary remain locally at:

```text
/tmp/mdslc-frozen-host-independent.jIipDE/independent.cpp
/tmp/mdslc-frozen-host-independent.jIipDE/independent
source SHA256 96b57444aae03e63e90425e93cb6e46832c209dfc3bece96254c61276dbbb13e
```

Implementation identity, unchanged during the independent run:

```text
FrozenHostCodegen.cpp a4a958431cfbe17ab38671bd060f47c9bef3064b1906e19a2a59180e6759ac7d
FrozenHostCodegen.h 048e2e3640f6d9cc412e685a9863fdac3de5c4abc48b113b0b2ddc324295bf6d
frozen_host_test.cpp 046e0607b320e9968262cc93c452a2500a4fbf82d02ce52078e68de494e97b66
```

No production edit or inspection of an unfinished emitter was used to obtain
this verdict. Final connected source-to-generated execution, exact ABI thunking,
private runtime build compatibility, packaging and CI remain independent gates.
