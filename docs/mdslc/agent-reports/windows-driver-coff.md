# Windows driver and COFF lane

## Scope and ownership

This lane made the native extractor and `mdslc++` driver portable to the
Windows MSVC ABI without changing the stable runtime C ABI. It owned the two
tool entry points, native frontend argument handling, generated weak-symbol
spelling, the narrow platform-support dependencies required by those tools,
and focused contract tests. Runtime DLL exports and distribution packaging are
owned by the packaging lane.

## Implemented contracts

- Windows process arguments cross one strict UTF-16-to-UTF-8 boundary through
  `wmain`; malformed Unicode fails closed.
- Filesystem paths use strict UTF-8 conversion, including paths containing
  spaces and `ü`.
- Child processes use argv vectors and `CreateProcessW`; only the three
  selected standard handles are inherited through an explicit handle list.
- Executable discovery examines explicit `PATH` entries and never searches the
  current directory or App Paths.
- The extractor authenticates Clang 21.1.8, queries and validates its adjacent
  builtin resource directory, and accepts clang-cl arguments only after an
  explicit Clang CL driver mode.
- `--frontend-info` and `--verify-ir` do not require clang-cl on `PATH` because
  compiler discovery is deferred until extraction.
- Driver-to-extractor compiler argv uses a bounded, versioned argument file on
  Windows. Compiler and linker invocations use response files.
- Windows `/c` produces a normal static `.lib` containing the generated COFF
  `.obj` files through adjacent `llvm-lib.exe`; final mode links the objects and
  runtime import library directly into a PE executable. No ELF `-r` emulation
  is used.
- `/TC`, `/Tc...`, opaque response files, incompatible `/LD`, `/link` with
  `/c`, raw dependency-output forwarding, resource-dir override, wrong output
  extensions, input/output aliasing, and unsupported CUDA requests fail
  explicitly.
- Generated files are replaced atomically on both Linux and Windows.
- Generated backend weak definitions use Clang's portable weak attribute; a
  COFF compile-only probe reported `StorageClass: WeakExternal` for the exact
  generated symbol.

## Focused commits

1. `7a0acce` strict UTF-8 path and argv boundaries
2. `2654c84` Windows process and executable-discovery hardening
3. `99762ec` portable generated weak definitions
4. `4dd1edc` bounded helper argument files
5. `33c72d1` authenticated clang-cl frontend arguments
6. `aebccb9` complete path-traversal snapshots
7. `a12d694` platform-support target wiring
8. `d215bb0` cross-platform atomic replacement
9. `733125e` Windows native extractor entry point and toolchain discovery
10. `8ef2092` Windows COFF driver pipeline
11. `d89d113` canonical `/Fo:` and `/Fe:` parsing
12. `ddf31f5` focused clang-cl/driver contract test

## Validation evidence

- Standalone platform support: 1/1 passed.
- Fresh disk-backed Debug build: 95 build steps completed.
- `frontend.native.primary`: passed.
- `frontend.native.driver_pipeline`: passed, 65 checks.
- `driver.native.selection`: passed.
- `windows.driver_coff.contract`: passed on Linux for clang-cl parsing and
  output-mutation rejection; Windows-only guard cases are registered for the
  hosted Windows job.
- `--frontend-info` and Matcore IR v1 `--verify-ir` passed with
  `PATH=/definitely/missing`.
- Repeated extraction to an existing output passed, exercising atomic replace.
- A 64,950-byte Linux driver invocation completed, while preserving the Linux
  direct-argv compatibility path.
- Generated backend cross-compilation with clang-cl for
  `x86_64-pc-windows-msvc` produced a COFF weak external symbol, confirmed by
  `llvm-readobj-21`.
- `git diff --check`: clean.

The large driver suite initially failed only because `/tmp` had a tmpfs quota;
the exact suite passed after rerunning with `TMPDIR` on the workspace disk.

## Deferred hosted gates

This Linux host cannot execute Windows binaries. The Windows workflow must
still validate actual clang-cl/MSVC linking, `llvm-lib`, COFF/PE inspection,
DLL discovery, Unicode paths, installed package relocation, external consumer,
and execution of the generated GEMM program. Until those hosted checks pass,
the Windows path is implementation-complete but not runtime-validated.
