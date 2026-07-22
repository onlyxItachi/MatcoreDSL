# Installed MatcoreDSL consumer

This project is intentionally outside the producer build graph. It locates an
installed MatcoreDSL package and uses `matcoredsl_add_executable` to turn one
valid-C++ `.mdsl` source into a normal generated object. CMake then performs the
ordinary executable link against `MatcoreDSL::Runtime`.

```sh
cmake -S compiler -B build-mdslc -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21
cmake --build build-mdslc -- -j2
cmake --install build-mdslc --prefix /tmp/matcoredsl-install

cmake -S compiler/tests/consumer -B /tmp/matcoredsl-consumer -G Ninja \
  -DCMAKE_PREFIX_PATH=/tmp/matcoredsl-install \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21
cmake --build /tmp/matcoredsl-consumer -- -j2
/tmp/matcoredsl-consumer/matcore_consumer
```

The helper accepts one `SOURCE`, the bootstrap `cpu` target, a `FRONTEND`
(`native` by default or explicit `ast-json-bootstrap` compatibility mode),
optional `COMPILE_OPTIONS`, and optional `LINK_LIBRARIES`. Frontend failure is
terminal; the helper and driver never retry through the compatibility frontend.
Ninja depfile integration tracks both the `.mdsl` source and its included user
headers. Editing either regenerates its object and relinks the consumer; an
unchanged subsequent build is a Ninja no-op.

On Windows, the installed-consumer validation recursively authenticates the PE
import closure before executing an installed tool. Project and Clang frontend
DLLs must be adjacent in the relocated package; the test removes every inherited
Clang/LLVM directory from `PATH` and later uses a compiler-only clang-cl,
llvm-lib, and resource-header staging directory. Windows API sets and UCRT are
recorded as operating-system components. Imports such as `vcruntime140*.dll`
and `msvcp140*.dll` are instead
reported as the external Microsoft Visual C++ 2015-2022 Redistributable (x64)
prerequisite; they are not mislabeled as OS DLLs or silently copied into the
package. The installed `share/MatcoreDSL/MatcoreDSLWindowsRuntimePrerequisites.json`
manifest records that distribution contract. Debug CRT imports are rejected.
CI may pass `--windows-import-report-out` to retain the deterministically
serialized installed-tool and external-consumer PE import graph beside the ZIP
artifact.
