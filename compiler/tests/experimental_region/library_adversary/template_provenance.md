# Pinned Clang specialization provenance counterexample

This is independent diagnostic evidence, not an executable certificate or a
new compiler authority. It explains why choosing every redeclaration's lexical
owner from `FunctionDecl::getLocation()` was too restrictive for the bounded
source-visible helper route.

The initial source had `primary.h`:

```cpp
#pragma once
namespace library {
template<class Tag> md::Value product(md::Value a, md::Value b) {
  return md::gemm(b, a, md::Numerics::strict_f32);
}
}
```

and `specialization.h`:

```cpp
#pragma once
namespace library {
template<> md::Value product<int>(md::Value a, md::Value b) {
  return md::gemm(a, b, md::Numerics::strict_f32);
}
}
```

The main file included both, then called `product<int>` and `product<long>` in
one admitted region, each followed by publication/observation. Noncommuting
inputs `{1,2,3,4}` and `{0,1,2,0}` require outputs `{4,1,8,3}` and `{3,4,2,4}`.
Both final pre-correction drivers rejected this at `primary.h:3:21` with
`closed declaration spelling is not body-source-owned`. The flattened main-file
version compiled and executed correctly. The header-only long instantiation
compiled. No unsupported-result path was mistaken for successful execution.

An independent LibTooling probe against installed Clang 21.1.8 visited concrete
template instantiations, then every canonical redeclaration. It printed raw
source locations, actual FileIDs, physical spelling line/column/offsets,
`getInnerLocStart`, declaration/type/parameter ranges, `getDefinition`, and
`doesThisDeclarationHaveABody`. The relevant actual observation was:

| Node/property | Canonical int-specialization stub | Selected int definition |
| --- | --- | --- |
| canonical declaration is self | yes | no: points to stub |
| specialization kind | `TSK_ExplicitSpecialization` | `TSK_ExplicitSpecialization` |
| own body / outer template lists | false / 0 | true / 1 |
| selected definition pointer | selected definition | self |
| name location | specialization.h offset 54, line 3 column 22 | identical |
| inner start | primary.h offset 53 | specialization.h offset 44 |
| complete source range | primary.h offsets 53..150 | specialization.h offsets 33..146 |
| outer function TypeLoc range | primary.h offsets 53..95 | specialization.h offsets 44..91 |
| parameter 0 range | primary.h offsets 71..81 | specialization.h offsets 67..77 |
| parameter 1 range | primary.h offsets 84..94 | specialization.h offsets 80..90 |
| resolved body range | specialization.h offsets 93..146 | identical |

In that captured AST, primary FileID was 1354, specialization FileID was 1355.
The stub's raw begin/end were 328199/328296; the selected name was 328355.
These numeric FileIDs/raw locations are one run's diagnostic values, not stable
identities. Source-range equality to
`getPrimaryTemplate()->getTemplatedDecl()->getSourceRange()` was true exactly,
not merely equal byte offsets in different files. Function TypeLoc and both
parameter SourceRange/TypeLoc comparisons were also exactly equal to that
primary. For the selected definition, each comparison was false as expected.

The committed regression strengthens the primary to generic `T` return and
parameter types; the same exact provenance relationships were independently
observed there, with primary lexical offsets 62..135 and parameter ranges
72..74 and 77..79. Sema substitution changes resolved types, not these spelling
ranges. Body execution still follows the concrete selected definition, never
the primary spelling used to authenticate the synthetic declaration.

Pinned primary implementation references:

- [FunctionDecl source-range implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/lib/AST/Decl.cpp#L4129-L4131)
- [Clang declaration and template APIs](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/include/clang/AST/Decl.h)
- [Actual versus presumed SourceManager locations](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/include/clang/Basic/SourceManager.h)

The production correction must keep this exception narrow: canonical,
nondefining, concrete explicit-specialization stub; selected-definition identity;
exact primary declaration/type/parameter spelling; and unchanged macro/range,
concrete type, attributes, selected-body ownership, effect and helper-retirement
checks. None of this permits an arbitrary mixed-file function body.
