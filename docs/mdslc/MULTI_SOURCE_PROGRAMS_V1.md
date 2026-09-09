# Authenticated multi-source programs v1

This experimental Linux x86-64 Clang/MLIR 21.1.8 route compiles one source-owned
program from 2–8 translation units in one invocation. It extends the existing
source region driver; it does not introduce serialized semantic modules,
object-file admission, opaque mathematical imports, or cross-region optimization.

```sh
mdslc-region --program \
  --host main.cpp --host utility.cpp \
  --region first.mdsl first --region second.mdsl second \
  --candidate generated-strict -o new-program -- -I./include
./new-program
```

Each `--region SOURCE NAME` selects one region definition in that physical TU.
An ordinary `--host SOURCE` can contain utility code with no Matcore header.
Exactly one ordinary C++ `main` definition is required across the program.
Additional annotated region definitions must not be left unselected. Duplicate
physical source identities, including symlink/hard-link aliases, are rejected.
The bounded C++ options after `--` apply to all source TUs. The existing
single-source syntax and default `automatic` candidate policy remain unchanged.
`-c` emits the complete program's native relocatable object; it is not an input
module format for a later driver invocation. Outputs must be new paths.

## What crosses translation units

Ordinary C++ declarations and host-visible `Result`/`Storage`/`Shape` calls
compose independently selected regions. The compiler authenticates each actual
region declaration against the canonical physical interface headers and a
compiler-issued declaration witness. C++ name/type lookup and the ordinary LLVM
linker do not establish mathematical authority. Foreign attributes such as
`pure`/`const`, incompatible packing, competing weak/COMDAT definitions, and
alternate assembler spellings cannot replace that authentication. Direct
linker aliases, weakrefs and resolvers targeting an issued region are rejected;
the compiler also compares actual direct calls with a clean call-site witness,
including their ABI and effect attributes.

Source-visible Value/Shape mathematical helpers still require an admitted body
in the consuming TU, typically through the [header-library route](EXPERIMENTAL_REGION_FRONTEND_V1.md).
A separately compiled Value helper with an unavailable body is not an import.
Retired Value helper symbols cannot escape into another host TU. Ordinary
internal-linkage utility functions remain TU-local even when their names repeat.
An ordinary overload with a distinct linker symbol remains ordinary host code;
sharing a source name alone does not grant or reserve a region implementation.

## Authority and execution boundary

All source/include/query closures are frozen and replayed. Every raw,
unoptimized Clang module passes artifact ownership checks; all selected-entry,
foreign-interface and retired-helper checks precede cross-TU linking. A clean
declaration witness supplements source checks—it does not excuse false foreign
function effects or change the original structural ABI comparator.

Each region retains its original exact semantic pairing and generated
implementation path. Linking does not fuse regions, move checks/publications
between calls, or create a program-wide failure transaction. A later failed
region call leaves an earlier call's already completed effects and owning
observations intact under the existing contract.

The compiler uses the same pinned Clang/linker, exact runtime/private candidate
DSOs, optional exact provider, environment scrubbing, closure rechecks and atomic
no-clobber publication as single-source mode. There is no fallback to arbitrary
objects, alternate providers, or user-selected runtime libraries. This does not
make ordinary C++ host code a sandbox, provide Windows generated-region support,
or claim measured multi-source performance improvement. An arbitrary ordinary
host wrapper or indirect resolver that computes a function pointer and supplies
false `pure`/`const` promises remains an invalid ordinary C++ host contract:
this route is not a general whole-program host-effect verifier.
