# Windows LLVM 21.1.8 Package Audit

Date: 2026-07-22

Lane: official Windows LLVM development-package and CMake export audit

## Authenticated input

The workflow uses the official LLVM release asset:

```text
clang+llvm-21.1.8-x86_64-pc-windows-msvc.tar.xz
size:   942572476 bytes
sha256: 749d22f565fcd5718dbed06512572d0e5353b502c03fe1f7f17ee8b8aca21a47
```

The independently downloaded asset matched both pinned values exactly. Only
its CMake exports and three relevant static archives were extracted for this
audit.

## Export defect and dependency path

`lib/cmake/llvm/LLVMExports.cmake` exports `LLVMDebugInfoPDB` as a static
imported target whose `INTERFACE_LINK_LIBRARIES` begins with this build-host
path:

```text
C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/
DIA SDK/lib/amd64/diaguids.lib
```

The representative transitive path from the required frontend is:

```text
clangTooling
  -> clangDriver
  -> LLVMProfileData
  -> LLVMSymbolize
  -> LLVMDebugInfoPDB
  -> stale diaguids.lib path
```

This is not an optional cosmetic dependency. `llvm-nm-21` inspection of the
packaged `LLVMDebugInfoPDB.lib` found unresolved `CLSID_DiaSource` and
`IID_IDiaDataSource` references. LLVM 21.1.8 also deliberately attaches the
DIA GUID library to this target when `LLVM_ENABLE_DIA_SDK` is enabled.
Consequently the stale path must be replaced with the coherent active Visual
Studio x64 DIA SDK `diaguids.lib`; simply deleting the dependency could leave
unresolved DIA GUID symbols.

## Link model

This exact archive sets `CLANG_LINK_CLANG_DYLIB=OFF` and contains no
`clang-cpp.dll/.lib` or C++ `LLVM.dll/.lib` imported target. It provides static
Clang/LLVM component libraries and the separate `LLVM-C.dll/.lib` C API.
`LLVM-C` cannot satisfy the Clang C++ Tooling ABI.

MDSLC must therefore retain the supported imported component-target graph for
this package. The narrow Windows repair rebases only the absolute
`diaguids.lib` item on `LLVMDebugInfoPDB` to the `vswhere`-selected Visual
Studio installation, preserves every other interface dependency, and fails
closed when no valid absolute x64 DIA GUID library is available.

## Review disposition

The first patch revision rebased only when the exported VS2019 path was
missing. That medium coherence concern was resolved: an explicitly selected
current-toolchain DIA library now replaces the build-host path even if the old
path happens to exist, and the override is validated as an existing absolute
file named `diaguids.lib`.

Final audit disposition: **no unresolved high or medium finding** for the
audited `x86_64-pc-windows-msvc` LLVM 21.1.8 package integration. This is a
package/link-interface audit; it does not by itself claim hosted Windows build
or runtime validation.
