# Independent public-driver storage conformance

Run against a built, frozen experimental region driver; this test does not
build or modify the compiler:

```sh
python3 compiler/tests/driver_storage_conformance/run.py \
  /absolute/build/bin/mdslc-region --expected-sha256 DRIVER_SHA256
```

The runner creates a new temporary artifact directory and prints its
`evidence.json` path. `--output` accepts an existing empty directory.
`--library-directory` defaults to the driver's sibling `../lib`; the ordinary
object-link check uses its isolated candidate DSO and public runtime DSO.
`--clangxx` defaults to the repository's coherent `/usr/bin/clang++-21`.
Driver and both DSO hashes are checked before every subprocess. Commands,
return codes, stdout, stderr, fixture hashes and library hashes are recorded.
The runner mechanically requires both Matcore shared-library `NEEDED` entries
in every executable and ELF64 x86-64 `REL` metadata in the emitted object.
Readelf runs under the C locale, and validation results are recorded separately
from subprocess exit status.

For an ASan+UBSan compiler installation, pass `--sanitized`. The runner checks
the driver banner against this request and supplies `-fsanitize=address,undefined`
to its ordinary final link. This option does not build or sanitize an
uninstrumented driver, and a banner alone is not instrumentation proof.

The source is ordinary C++ using only `<matcore/region.h>` in the mathematical
region. The host oracle uses an independent double-precision scalar multiply
and full 192-element arena comparison, plus hard-coded rectangular products.
All inputs are small integers, so every multiplication and accumulated sum is
exactly representable in f32; float-bit equality also checks positive zero.
The seven alias layouts include differently shaped descriptors with the same
pointer and offset-overlapping publications. No no-alias assumption is made.

Each carry direction executes 22 host cases under `native-strict`,
`generated-strict` and `automatic`. `existing-native` is separately expected
to reject `strict_f32` with `candidate_incompatible` before publication, including
empty math. It is not treated as a strict numerical implementation.
The runner also copies both source files and the identical host header to a
temporary subdirectory, changing only the two per-operation numerical profiles
to `reassociate_f32`; original filenames and operation line locations remain.
These copies execute the same oracle under requested `existing-native` policy.
Pass `--has-openblas` only for a provider-enabled installation to additionally
execute both reassociated sources under requested `openblas` policy. Without
that option the evidence explicitly says OpenBLAS was not requested or tested;
it does not infer that the installation was built without the provider. Failed
requested provider execution is not converted into a skip or fallback success.

The `lhs.mdsl` and `rhs.mdsl` line numbers are intentional diagnostic oracles.
Changing their operation locations requires reviewing the expected source lines
in the host header. Frontiers are compiler-issued result diagnostics, never
source inputs. Unsupported nested reads and arbitrary host effects must reject
at their original source locations without publishing an object.

Public Result does not identify the implementation actually selected. These
tests make no implementation-identity, benchmark, broad floating-point edge-case,
GPU, cross-platform, installed-package, or private-DSO ownership claim. Sanitizer
or provider execution claims require an actual matching run, not just these
runner options. `python3 compiler/tests/driver_storage_conformance/runner_test.py`
checks parser negative controls and source-copy invariants without any compiler
execution or build.
