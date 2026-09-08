# Independent source-library adversaries

These tests exercise the actual authenticated source driver. They do not
implement admission, reinterpret a semantic graph, or accept any generic
compilation failure as a successful negative control.

Run each case through the selected built compiler:

```sh
cmake -DDRIVER=/absolute/path/to/bin/mdslc-region \
  -DCASE=repeated_include -P /path/to/library_adversary/run.cmake
```

The other cases are `escaped_address` and `escaped_wrapper`. The integration
owner registers them; this independent lane edits no production implementation
or existing test registration.

## What each case proves

`repeated_include.mdsl` includes the same unguarded `shared_body.h` into two
namespaces. Each inclusion has a distinct Clang FileID even though its physical
bytes/path can share one semantic source-file record. The nested helper in the
second inclusion must use that actual body owner, not recover the first
inclusion through `translateFile(FileEntry)`. Clang documents the
[first-inclusion behavior](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/clang/include/clang/Basic/SourceManager.h#L1490-L1500).

The header deliberately uses `#line 700 "forged-main.mdsl"`. Its presumed
coordinates must not become physical source identity. The selected region
executes helpers from both inclusions and returns an old immutable Value after
a nested GEMM whose result is unused. On success, old contents are restored and
both owning observations survive subsequent host writes. With truthful but
incompatible late-input dimensions, the unused GEMM must fail at frontier 8,
after publication/observation frontier 6, leaving exactly that earlier effect
prefix. The reported location must be the physical outer main caller at line
29, column 17, not the header or its spoofed filename. Shape-only helpers also
have ordinary host uses that must remain supported. The executable checks
**67 assertions** with exact output and empty stderr.

`escaped_address.mdsl` puts a source-Value helper address in an externally
visible global. `escaped_wrapper.mdsl` calls it from an ordinary host function
outside the selected expansion closure. Both must reach the specific
`retired Value helper has a remaining nonhelper use:` rejection and publish no
artifact. An early admission error, timeout, crash, generic nonzero status or
zero exit without an artifact does not satisfy these tests.

## Observed evidence

The implementation worktree was based on canonical
`0e8c040809ae1fe6b0f1c01e41fc509b07212be7` plus the owning lane's dirty H1
prototype. All three fixtures independently passed through its actual Release,
OpenBLAS-enabled, coherent 21.1.8 driver, SHA-256
`9ba071d2f2a9abfec9663686950fcc118d377f0ead968d3682d0034f3f0447b2`.
The five-file production diff (admission, semantic header/implementation,
experimental emitter, builtin candidate) had SHA-256
`12f7140f340993b605ca1205e8266dd8e15bdb0e9a9c454ad22a8a116c32691a`.
The source tests force `generated-strict`; they do not exercise OpenBLAS or
claim provider comparison. This prototype uses the original scalar generated
schedule, not the separate row/cache integration.

All three sources also passed ordinary Clang 21 C++ syntax checking. The older
scalar driver correctly rejects header helpers at admission; importantly that
error fails, rather than satisfies, the new retirement classifier. `/bin/false`
fails the retirement classifier and `/bin/true` without an artifact fails the
positive runner. Each invocation uses its own fresh working directory.

These are local independent Release results, not final-head integration,
package, hosted-CI, sanitizer or performance evidence. No additional template
grammar is introduced: the existing experimental admission test already
supports bounded Sema-resolved template specializations. The new source table
and active physical owner must preserve their concrete bodies and symbols.
