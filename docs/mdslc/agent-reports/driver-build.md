# Driver and Build Agent Report

## Ownership and baseline

- Agent role: Driver and Build (C).
- Worktree: `/home/hamza-usta/MatcoreDSL-wt-driver`.
- Branch: `mdslc/driver-build`.
- Starting SHA: `351075e4d8af1880330b7c0474d701ca76776dfa`.
- Write ownership was limited to `compiler/CMakeLists.txt`,
  `compiler/cmake/**`, `compiler/tools/mdslc/**`,
  `compiler/examples/hello_host.mdsl`, and this report.
- No public header, frontend/IR/runtime implementation, legacy source, root
  CMake, or integration-branch file was changed.

The repository's existing `rules/toolchain.md` pins LLVM 18.1.3 for the legacy
MLIR/nanobind extension. The standalone task explicitly selected the coherent
`/usr/bin/clang++-21` executable for the independent, non-MLIR bootstrap.
The standalone build does not load or link LLVM, MLIR, Python, or nanobind, and
the legacy pin was not changed.

## Implementation

- Added an independent C++20 CMake project under `compiler/`.
- Required Ninja, enabled an already-installed ccache, and defaulted the host
  compiler invoked by the driver through the configurable
  `MDSLC_CLANGXX_EXECUTABLE` cache path.
- Added the compiled `mdslc++` driver. It constructs an argument vector and
  calls `fork`/`execv`; it never invokes a shell.
- The driver injects `-x c++` for positional `.mdsl` inputs, forwards ordinary
  compiler arguments and inherited stdout/stderr, propagates the child exit
  status, prints the exact argv with `--verbose`, and translates
  `--save-temps` to Clang's `-save-temps=obj`.
- Added `hello_host.mdsl`, which is ordinary valid C++20 and prints `5`.

## Validation evidence

Fresh configure and build:

```text
cmake -S compiler -B /tmp/matcoredsl-mdslc-driver-final -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21
cmake --build /tmp/matcoredsl-mdslc-driver-final -- -j2
```

Result: configured as Clang 21.1.8 and built `bin/mdslc++` successfully. The
generated Ninja file records `/usr/bin/ccache` as the launcher.

Executable proof:

```text
/tmp/matcoredsl-mdslc-driver-final/bin/mdslc++ --verbose \
  -std=c++20 compiler/examples/hello_host.mdsl \
  -o /tmp/matcoredsl-mdslc-driver-final/hello_host
/tmp/matcoredsl-mdslc-driver-final/hello_host
```

The verbose command contained
`'/usr/bin/clang++-21' '-std=c++20' '-x' 'c++'` before the `.mdsl` input.
Program output was exactly `5` and exit status was zero.

Relocatable-object and saved-temporary proof:

```text
/tmp/matcoredsl-mdslc-driver-final/bin/mdslc++ --verbose --save-temps \
  -std=c++20 -c compiler/examples/hello_host.mdsl \
  -o /tmp/matcoredsl-mdslc-driver-final/hello_host.o
file /tmp/matcoredsl-mdslc-driver-final/hello_host.o
readelf -h /tmp/matcoredsl-mdslc-driver-final/hello_host.o
nm -C /tmp/matcoredsl-mdslc-driver-final/hello_host.o
```

Results:

- `file`: ELF 64-bit LSB relocatable, x86-64, not stripped.
- `readelf`: `Type: REL (Relocatable file)`, machine x86-64.
- `nm -C`: defined `main` and weak `int host_add<int>(int, int)` symbols.
- Saved files: `hello_host.bc`, `hello_host.ii`, `hello_host.s`, and
  `hello_host.o`.

Exit-code propagation was checked with a missing `.mdsl` input. Direct
`clang++-21` and `mdslc++` both returned status 1, and the wrapper preserved
Clang's diagnostic naming the missing source file.

## Commits and handoff

- `569e14c60b4b8fa41411d99a2e9d186ebaf0c187` —
  `build(mdslc): add isolated standalone CMake skeleton`
- Driver implementation commit: this report is included in that commit; its
  SHA is supplied in the agent handoff.

LibTooling extraction, generated stubs, runtime linkage, install rules, and
consumer CMake integration are intentionally outside this Goal 1-2 slice.
