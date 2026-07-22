# Windows platform-support lane

Date: 2026-07-22

## Scope and ownership

This lane began at `091d74072a710389b4a8e9d51f696ad9773021e6` and owns only:

- `compiler/lib/support/**`
- `compiler/tests/support/**`
- this report

It deliberately does not edit the standalone compiler's top-level CMake,
drivers, frontend, runtime, or package integration. The integration owner must
connect the support target to those call sites after reviewing its interface.

## Implemented boundary

`MatcoreDSL::PlatformSupportV1` is a versioned C++20 internal support library.
The common interface provides:

- current-executable discovery;
- shell-free vector-`argv` child execution, with an explicit working directory,
  per-child environment overrides, separate stdout/stderr capture, or normal
  stream inheritance;
- owned temporary directories with best-effort scoped removal;
- lexical or weakly-canonical path normalization;
- executable discovery;
- file snapshots containing normalized path, existence/type, size, write-time,
  and native physical file identity;
- deterministic `CreateProcessW` argument quoting;
- BOM-free UTF-8 response-file encoding and path-safe writing.

The API rejects unknown request versions, NUL-bearing arguments, malformed
environment names, unsupported response-file syntax, and invalid snapshot
inputs before reporting success.

Operating-system code is isolated in two translation units:

- Linux uses `/proc/self/exe`, `fork`/`execv`, explicit pipes, an exec-error
  pipe, `mkdtemp`, `stat`, and `PATH` discovery. Both captured output streams
  are drained concurrently.
- Windows uses `GetModuleFileNameW`, `CreateProcessW`, wide filesystem APIs,
  `GetEnvironmentStringsW`, `SearchPathW`, `GetFileInformationByHandle`, and
  `CreateDirectoryW`. Public argument and environment strings are validated
  UTF-8 and converted with strict UTF-16 APIs. No shell command is constructed.

The CMake file can be configured directly and intentionally is not yet included
by `compiler/CMakeLists.txt`.

## Focused validation

All build directories were external under `/tmp`; no generated output entered
the repository.

| Configuration | Compiler/instrumentation | Result |
| --- | --- | --- |
| Release | Clang 21.1.8 | 1/1 CTest passed |
| Debug | Clang 21.1.8 | 1/1 CTest passed |
| RelWithDebInfo | Clang 21.1.8 ASan+UBSan | 1/1 CTest passed |
| RelWithDebInfo | Clang 21.1.8 TSan | 1/1 CTest passed |
| Release | GCC 15.2.0 | 1/1 CTest passed |
| Static analysis | Clang 21.1.8 analyzer, common+Linux | no finding |

Representative commands:

```sh
cmake -S compiler/lib/support -B /tmp/matcore-win-platform-support-build \
  -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-win-platform-support-build -- -j2
ctest --test-dir /tmp/matcore-win-platform-support-build \
  --output-on-failure -j1
```

The focused executable verifies Windows quoting goldens on Linux as pure logic,
space- and UTF-8-bearing executable/working/file/response paths, physical file
identity through a hard link, environment propagation, inert shell
metacharacters, independent stdout/stderr capture, inherited streams, explicit
nonzero exit propagation, launch failures, request-version rejection, scoped
temporary cleanup, and concurrent draining of 128 KiB on each output stream.

## Windows validation status

The Windows backend is implemented but not compiled or executed in this Linux
lane. The local `clang-cl-21` executable targets MSVC but the Linux host does
not provide the matching Windows SDK and MSVC standard-library development
surface. The Windows CI lane must compile and runtime-validate this backend
with the audited Windows LLVM 21.1.8, Visual Studio Build Tools, and Windows SDK
before any Windows-support claim is made.

## Integration obligations and explicit gaps

The driver/package lane should account for these facts:

1. Add `lib/support` to the compiler build and link only the tools that consume
   it; this lane intentionally made no top-level CMake change.
2. Replace existing process helpers atomically so a tool never switches back
   to shell-built command strings.
3. Select GNU versus Windows response-file syntax from the validated target
   platform and test the produced file with the actual Clang driver.
4. File snapshots are metadata and physical identity, not a content digest.
   Trusted-input authentication must continue to hash content separately.
5. Process v1 has no stdin redirection, timeout, cancellation, or tree-kill
   contract. Callers needing those semantics must extend the versioned API,
   not infer them.
6. Temporary-directory destructor cleanup is intentionally best effort; tools
   that promise cleanup diagnostics must remove explicitly before exit.
7. Linux current-executable discovery depends on the backend-local
   `/proc/self/exe` interface. This does not leak into shared code, but a
   non-procfs Linux environment needs a documented fallback in a later API.
8. The Windows response writer emits BOM-free UTF-8. Actual `clang-cl` response
   consumption, Unicode paths, DLL-adjacent execution, and inherited-console
   behavior remain hosted-Windows acceptance gates.

No GPU, BLAS, MLIR, matrix operation, legacy Python/JIT, or runtime ABI code was
changed.
