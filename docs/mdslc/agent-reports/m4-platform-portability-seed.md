# Milestone 4 platform portability seed

## Scope and ownership

This lane owns only:

- `compiler/lib/platform/**`;
- `compiler/tests/platform/**`;
- this report.

It did not modify the compiler driver, extractor, runtime, planner, code
generator, root standalone CMake project, package helper, workflow, or
repository-hygiene policy.

## Implemented contract

`PlatformRecordV1` is a pure C++20, versioned description of the compile target.
It records, in a canonical vocabulary:

- Linux, Windows, or unknown platform kind;
- x86-64, AArch64, or unknown target architecture;
- GNU-mode Clang, clang-cl, GCC, MSVC, or unknown compiler frontend;
- ELF, COFF, or unknown object format;
- ELF, PE, or unknown executable format;
- ELF shared-object, Windows DLL/import-library, or unknown runtime model;
- POSIX fork/exec, Windows CreateProcess, or unavailable process model;
- POSIX `dlopen`, Windows `LoadLibrary`, or unavailable dynamic-library model.

The record separates each modeled facility's implementation state, discovery
completeness, and runtime-validation state. A deterministic constexpr verifier
rejects unknown enum values, version mismatches, incoherent platform/format
combinations, and false runtime-validation claims. Runtime validation requires
an implemented capability with complete discovery.

Compile-time discovery marks the existing Linux compiler, artifact, runtime,
and process models as implemented. It does not infer runtime validation merely
from compiler macros. The dynamic-library model is recorded but remains
`modeled` because this seed adds no loader abstraction.

Compile-time Windows discovery deliberately records the expected clang-cl/MSVC,
COFF/PE, DLL/import-library, CreateProcess, and LoadLibrary vocabulary as
`modeled`, never implemented or runtime-validated. This permits later Windows
backends to use the same contract without implying that they exist today.

`format_platform_record_v1` emits a stable, human-readable single-line
diagnostic with fixed field and capability ordering.

## Tests

The focused test covers:

- current-host compile-time discovery;
- a valid implemented Linux/AArch64 synthetic record;
- a valid modeled, explicitly unvalidated Windows/clang-cl record;
- rejection of runtime validation on a merely modeled Windows backend;
- Windows plus ELF incoherence;
- Linux plus CreateProcess incoherence;
- incomplete records claiming complete discovery;
- unknown schema versions;
- a well-formed unknown platform;
- deterministic diagnostics and explicit Windows status text.

Validation used the audited compiler tuple:

```text
Ubuntu clang version 21.1.8 (6ubuntu1)
Target: x86_64-pc-linux-gnu
Thread model: posix
```

Commands:

```sh
/usr/bin/clang++-21 -std=c++20 \
  -Wall -Wextra -Wpedantic -Werror \
  -Icompiler/lib/platform \
  compiler/lib/platform/platform.cpp \
  compiler/tests/platform/platform_test.cpp \
  -o /tmp/mdslc-platform-v1.ifIlzx/platform_test
/tmp/mdslc-platform-v1.ifIlzx/platform_test

/usr/bin/clang++-21 -std=c++20 \
  -Wall -Wextra -Wpedantic -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Icompiler/lib/platform \
  compiler/lib/platform/platform.cpp \
  compiler/tests/platform/platform_test.cpp \
  -o /tmp/mdslc-platform-v1.ifIlzx/platform_test_san
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
  /tmp/mdslc-platform-v1.ifIlzx/platform_test_san
```

Both executions printed `platform record v1 tests PASS`. The discovered record
was Linux, x86-64, GNU-mode Clang, ELF objects/executables, ELF shared runtime,
and POSIX fork/exec. Its implementation state is explicit; its runtime
validation state remains `not-validated` in this compile-time-only record.

The same source also compiled with `/usr/bin/g++ -std=c++20 -Wall -Wextra
-Wpedantic -Werror` and passed, reporting the distinct `gcc` frontend. This is
only a portability syntax check; Clang remains the supported MDSLC frontend.

## Deferred integration and limitations

The directory contains a local CMake target definition for later integration,
but this lane intentionally did not add it to the shared root build. The lead
must make that small integration change after resolving shared-file ownership.

This seed does not provide:

- a Windows process backend;
- Windows secure file-identity or temporary-file handling;
- clang-cl command translation or response files;
- COFF object-set/archive production;
- runtime DLL discovery;
- Windows capability discovery;
- a Windows build, package, artifact, or execution result.

Therefore Windows frontend, runtime, planner, packaging, and native CPU variants
remain **unvalidated and unsupported**. The modeled record is a portability
contract, not a support claim.
