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

## Final H1 review and specialization regression

The original three cases were independently rerun under ASan+UBSan at clean
H1 validation commit `8b5b8c31ed8b389293aef4179214a06e192a811e`, using driver
SHA-256 `81b670b1892d5dabd33e1c9db8b73ab1cedf4380a7313d63f8811881417b2f6b`.
All passed: the repeated-inclusion program completed 67 checks, and both
escaped-helper programs reached their exact retirement rejection.

An additional independent probe found a safe compatibility rejection: a primary
template in one header and an explicit specialization in another failed because
Clang's synthesized canonical specialization declaration has its name location
in the selected header but its lexical/type spelling in the primary header.
The same source flattened into the main file compiled and executed its
noncommuting-matrix oracle; a header-only primary instantiation compiled.
The failed cross-header probes used the preceding ASan driver and Release
SHA-256 `908dc80083d5801016d4717b0ded1d00e82f7a93601c60b7d32e6028c0af6850`.
This was not wrong-body execution or an authority bypass. The owning lane then
implemented a narrow exact-primary-provenance correction. See the pinned
[Clang provenance evidence](template_provenance.md).

The six new cases are run with `template_run.cmake` and these `CASE` values:

- `template_specialization`: a generic `T` primary reverses multiplication;
  the separately defined `int` specialization preserves operand order. Both
  concrete functions must retire, and the actual generated executable must
  pass 59 checks for noncommuting products, two publication/observation pairs,
  owning observations, and a late primary-only shape failure preserving the
  first publication and physical outer-caller coordinates.
- `template_macro_type`: macro-derived primary type spelling must not qualify
  as the exact synthesized-declaration provenance exception.
- `template_macro_body`: selected-body macro expansion remains rejected.
- `template_hidden_effect`: the selected body cannot conceal observation.
- `template_escaped_specialization`: an ordinary global taking the selected
  concrete specialization's address must reach exact helper retirement failure.
- `template_split_body`: an ordinary function whose name and actual body are in
  different physical files remains rejected; this is not a blanket cross-file
  declaration/body exception.

All six independently passed through each corrected driver at clean production
commit `c6ff41b763fc97c7cb016ee20fe764c76f7f968e`:

| Profile | Driver SHA-256 | Actual result |
| --- | --- | --- |
| Release | `6ebd962629218e2dbf3f7291be0948c0245994c66711a99e4f4347031580eda0` | 59-check executable plus five exact rejections |
| ASan+UBSan | `61ddbb12e17de334819c9b98643c301fedcb1d042ce6cbb245ead78e2ee29341` | 59-check executable plus five exact rejections |

The sanitizer run used `DEBUGINFOD_URLS=''`,
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`,
and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. These are independent
source-execution/negative-control results, not the owning lane's complete
Release/ASan CTest gates, hosted CI, installed-package evidence, performance
measurements, or support for arbitrary templates. The runner requires exact
diagnostic text and no output artifact for each rejection; generic nonzero exit
does not pass. The positive runner requires the actual executable and exact
59-check output with empty stderr.
