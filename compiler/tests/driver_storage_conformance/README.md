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

The `lhs.mdsl` and `rhs.mdsl` line numbers are intentional diagnostic oracles.
Changing their operation locations requires reviewing the expected source lines
in the host header. Frontiers are compiler-issued result diagnostics, never
source inputs. Unsupported nested reads and arbitrary host effects must reject
at their original source locations without publishing an object.

No benchmark, broad floating-point edge-case coverage, sanitizer, GPU,
cross-platform, installed-package, or private-DSO ownership claim follows.
