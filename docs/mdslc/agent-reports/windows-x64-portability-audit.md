# Windows x64 portability audit

Audit base: `105ac4873f63d1737aa220766d83330a325bdd4d`

Audit date: 2026-07-22

This is a read-only implementation inventory for the deferred Windows
compatibility phase. It does **not** establish Windows support. At this
checkpoint the Windows compiler, runtime DLL, CPU variants, package, consumer,
and distribution ZIP have not been built or executed. The existing platform
record can describe Windows as a target; that model must not be confused with
a validated Windows backend.

## Scope inspected

The audit covered the standalone compiler build, driver and extractor,
generated-source pipeline, runtime and planner, platform/capability/topology
records, benchmark tool, installed CMake package, tests, and workflows. The
legacy Python/JIT/MLIR implementation is outside this portability lane and
must remain unchanged.

Relevant implementation surfaces include:

- `compiler/CMakeLists.txt` and the standalone subdirectory CMake files;
- `compiler/tools/mdslc/main.cpp`;
- `compiler/tools/matcore-extract/main.cpp` and its native/bootstrap frontends;
- `compiler/cmake/MatcoreDSLCompile.cmake.in`;
- runtime C ABI, code generation, CPU capabilities, topology, affinity,
  execution context, planner, and benchmark sources;
- package, frontend, integration, ABI, ISA-artifact, and hygiene tests;
- `.github/workflows/`.

## Foundations that can be preserved

- `.mdsl` remains valid C++, and native LibTooling remains the authenticated
  frontend.
- The trusted-header and canonical annotated-declaration checks use Clang
  `FileEntry`/`SourceManager` identity rather than textual call matching.
- Matcore IR, planner records, and platform records are versioned and
  deterministic.
- The stable runtime surface is a C ABI. `runtime_c.h` already has a suitable
  Windows DLL import/export macro seam; no C++ ABI type needs to cross it.
- The persistent executor is based primarily on standard C++ threads and can
  be retained behind platform-specific topology and affinity adapters.
- The platform model already distinguishes Linux, Windows, and unknown target
  families. Its current Windows entry is descriptive only.

## Coherent Clang/LLVM 21 Windows provider

The GitHub-hosted `windows-2025` image inventory observed during this audit
lists LLVM 20.1.8, so the runner image alone does not satisfy the project's
exact Clang/LLVM 21.1.8 requirement.

Use the official LLVM 21.1.8 Windows MSVC development archive as one coherent
compiler/header/library/CMake provider:

- release: <https://github.com/llvm/llvm-project/releases/tag/llvmorg-21.1.8>
- archive URL: <https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/clang%2Bllvm-21.1.8-x86_64-pc-windows-msvc.tar.xz>
- archive name: `clang+llvm-21.1.8-x86_64-pc-windows-msvc.tar.xz`
- byte size: `942572476`
- SHA-256: `749d22f565fcd5718dbed06512572d0e5353b502c03fe1f7f17ee8b8aca21a47`
- hosted-runner inventory consulted:
  <https://raw.githubusercontent.com/actions/runner-images/main/images/windows/Windows2025-Readme.md>

The CI lane must verify the digest before extraction, inventory the required
Tooling, ASTMatcher, Rewriter, Lexer, AST, and Frontend headers, discover the
LLVM and Clang CMake packages from that same archive, and fail closed on any
version/provider mismatch. Cache keys must include the archive digest.

The runner's Visual Studio 2022 Build Tools, Windows SDK, CMake, and Ninja may
provide the MSVC ABI/linker/SDK surface. RapidJSON must come from a pinned,
reproducible source or package. Windows OpenBLAS remains optional and should be
configured off until a coherent, licensed, reproducible CBLAS provider is
selected and authenticated.

The official archive may expose component imported targets instead of the
Linux build's monolithic `clang-cpp` target. CMake should prefer supported
imported targets such as `clangTooling`, `clangFrontend`,
`clangASTMatchers`, and `clangRewrite`, with the matching LLVM support targets,
rather than hardcoded archive paths. Exact LLVM/Clang 21.1.8 equality remains a
configure-time gate.

## Portability gaps

### Process, filesystem, and executable discovery

`mdslc++` currently depends directly on POSIX operations including
`stat`/`lstat`, `readlink`, `open`/`fstat`, `fork`/`execv`/`waitpid`, `access`,
`mkdtemp`, POSIX file identity/time types, and `/proc/self/exe`.
`matcore-extract` also assumes `/proc/self/exe` and POSIX process identity.
The bootstrap AST-JSON frontend has its own fork/pipe process path.

Introduce narrow, separately tested platform interfaces for:

- current-executable and installed-prefix discovery;
- file identity and immutable source snapshots;
- temporary directories and atomic artifact publication;
- vector-based child-process execution;
- environment and executable discovery;
- path normalization and native diagnostics.

The Windows launcher must use `CreateProcessW`, wide paths, and the documented
Windows command-line quoting rules. It must never build a shell command.
Internally generated response files are required for long commands and paths
with spaces or Unicode; existing rejection of untrusted opaque/nested user
response files must not be weakened.

The explicit AST-JSON bootstrap compatibility frontend may remain disabled on
Windows initially. Its absence must produce a clear diagnostic and must never
become a silent fallback from the native frontend.

### clang-cl argument model

The driver and LibTooling fixed compilation database need an explicit MSVC
driver mode. The Windows path must use or faithfully translate to:

- `/TP` for C++ source;
- `/std:c++20`;
- `/c`, `/Fo`, and `/Fe`;
- `/I` and `/D`;
- `--driver-mode=cl` when constructing the native Tooling command.

The current parser can misclassify `/...` compiler switches as file paths, and
GNU `-std=c++20` is not an adequate clang-cl contract. Dependency generation
can use clang-cl's Clang forwarding form, for example `/clang:-MD`,
`/clang:-MF`, and `/clang:<depfile>`, with care not to confuse Clang `-MD` with
the MSVC `/MD` runtime-library switch. Extraction and host compilation must
receive semantically identical standards, defines, include paths, target, CRT,
and SDK options.

### COFF artifact model

The Linux driver currently assumes `.o`, ELF/shared-object conventions,
`clang++ -r`, `-l`, and rpath. None is an acceptable Windows partial-link
contract.

For Windows, produce the generated host, stub, and backend `.obj` files as
declared outputs and either:

1. link those objects directly into the final PE executable; or
2. archive them into a normal static `.lib` and link it conventionally.

Do not emulate `clang++ -r`, use `/FORCE:MULTIPLE`, or label an archive as an
object file. If compile-only output cannot honestly be represented by one COFF
object, expose an explicit object-set directory or static-library artifact.
The installed layout must distinguish `bin/matcore_runtime.dll` from
`lib/matcore_runtime.lib`; Windows must not emit or depend on rpath.

Generated weak/duplicate site definitions currently rely on ELF/GNU weak
attribute behavior. Their exact clang-cl/COFF COMDAT behavior must be compiled
and co-linked in a focused test, or replaced with an explicit COFF-safe
one-definition strategy. Linker duplicate suppression is not an acceptable
substitute.

### Build and package

The standalone CMake project presently contains Linux compiler defaults and
GNU-style warning/optimization options. Root, IR, planner, platform, runtime,
frontend, and tool targets need compiler-family branches for clang-cl/MSVC.
The runtime should remain a DLL plus import library and install all required
runtime dependencies. If the extractor dynamically imports LLVM DLLs, those
DLLs and applicable license notices must be included in the distribution;
linking the official component libraries statically may avoid that runtime
dependency subject to the LLVM package's supported targets and licensing.

The installed driver must not embed the CI extraction directory or checkout
path. It should authenticate a user-selected or package-resolved `clang-cl`
and derive the trusted header relative to its relocatable installed prefix.
`find_package(MatcoreDSL REQUIRED)` must work from a prefix containing spaces.

### CPU capability, topology, and affinity

Capability v2 has Windows/MSVC intrinsic groundwork, but the clang-cl macro
combination can currently enter a Clang-specific CPUID path while planner v1
contains `_MSC_VER` exclusions. Normalize and test the clang-cl/MSVC-ABI path.
All optional feature use must still distinguish hardware support, OS-enabled
state, compiler support, implementation availability, and runtime validation.
AMX permission/state remains unknown on Windows and must fail closed.

Only Linux topology discovery is implemented today. Introduce a host topology
entry point and an isolated Windows backend using documented Windows APIs,
principally `GetLogicalProcessorInformationEx` for cores, packages, caches,
and `RelationNumaNodeEx`. Processor groups and logical-processor numbering must
be mapped deterministically; configurations above 64 processors must fail
closed until group-aware execution is complete.

Affinity and observed placement should use documented APIs such as
`GetProcessAffinityMask`, `GetThreadGroupAffinity`,
`SetThreadGroupAffinity`, and `GetCurrentProcessorNumberEx`. Do not scatter
Windows branches through shared planner/runtime code. Real multi-node NUMA
performance cannot be claimed from synthetic records or a one-node hosted
runner.

### Tests and workflows

Current integration scripts assume ELF tools and names (`readelf`, `ldd`,
`.so`, `.o`, rpath), Unix permissions/symlinks, shell execution, `taskset`, and
fixed Unix prefixes. Retain those as Linux gates and add Windows-specific or
portable tests rather than weakening them.

Windows artifact checks should use LLVM tools such as `llvm-readobj`,
`llvm-nm`, and exact-symbol `llvm-objdump --disassemble-symbols=...`. Tests must
cover native frontend authentication, IR and planner determinism, COFF object
sets, PE execution, DLL/import-library C ABI, install/consumer behavior,
incremental dependency regeneration, paths with spaces/Unicode, and installed
path leakage. AVX2/AVX-512 artifact tests must inspect only the exact
microkernel symbol.

## Four isolated implementation lanes

No two lanes should edit the same files concurrently without an explicit
handoff.

### Lane A — platform and process foundation

Ownership:

- new process/filesystem/executable-discovery interfaces and platform CMake;
- Windows `CreateProcessW` backend and command-line quoting;
- response-file, wide-path, temporary-directory, file-identity, and atomic
  publication primitives;
- driver/extractor call-site conversion to those interfaces.

### Lane B — driver and artifact pipeline

Ownership:

- `mdslc++` argument classification/translation and artifact modes;
- clang-cl/native-Tooling option parity;
- generated COFF object-set or archive orchestration;
- `MatcoreDSLCompile.cmake` Windows object wiring;
- generated-site definition strategy and driver/artifact tests.

### Lane C — Windows runtime and topology

Ownership:

- clang-cl capability detection normalization;
- Windows topology, processor-group, NUMA-record, and affinity backends;
- runtime/planner/benchmark host-discovery call sites;
- injected/synthetic records and physical hosted-runner tests;
- fail-closed unsupported feature and topology behavior.

### Lane D — build, package, validation, and CI

Ownership:

- LLVM/Clang 21.1.8 archive acquisition, digest verification, and CMake target
  discovery;
- clang-cl build flags, DLL/import library installation, licenses, package
  relocation, and external consumer;
- Windows validation scripts, paths-with-spaces test, environment/artifact
  report, and ZIP assembly;
- `windows-2025` GitHub Actions workflow and artifact upload.

## Validation boundary

An official hosted Windows runner can physically validate:

- exact clang-cl/LibTooling 21.1.8 coherence;
- trusted-header, canonical declaration, IR, and rewrite behavior;
- COFF `.obj`, PE executable, runtime DLL, import library, and strict C ABI;
- reference, tiled, compiler-vectorized, and packed AVX2 execution when the
  runner exposes the required CPU/OS capabilities;
- persistent worker correctness, explicit thread limits, clean shutdown, and
  oversubscription guards;
- build-tree and relocated installed-package consumers, including paths with
  spaces;
- one-node topology discovery and deterministic synthetic topology/NUMA logic;
- repository hygiene and a generated ZIP distribution candidate.

Hardware-dependent claims remain conditional:

- AVX-512 may be runtime-validated only when the runner exposes the exact
  required features and OS state; otherwise it is compile/disassembly plus
  synthetic-legality evidence only.
- BF16, VNNI, and AMX require their exact hardware/OS permission and successful
  execution; otherwise they are unavailable or compile-only as applicable.
- Multi-node NUMA remains synthetic-only unless run on real multi-node Windows
  hardware.
- OpenBLAS is intentionally omitted until a coherent Windows CBLAS provider is
  selected.

The Windows workflow should run on `windows-2025`, enter the Visual Studio
development environment, verify/extract the pinned LLVM archive, configure
clang-cl with Ninja, build Release and practical Debug coverage, run the
frontend/IR/planner/runtime/package matrix, execute a real `.mdsl` PE proof,
install to a fresh prefix with spaces, run the external consumer, inventory
artifacts and capabilities, and upload—but never commit—the ZIP candidate.

## Recommended implementation order

1. Land Lane A's portable process/filesystem seams without altering Linux
   behavior; keep all Linux acceptance tests green.
2. Land Lane B's clang-cl arguments and honest COFF object model, with generated
   site/link tests.
3. Land Lane D's coherent exact-version CMake discovery, DLL packaging, and
   initial hosted workflow; use its failures to close compiler/link/package
   gaps.
4. Land Lane C's Windows capability/topology/affinity backend and runtime tests.
5. Complete installed paths-with-spaces, external consumer, exact ISA artifact,
   PE execution, and ZIP validation, then run an independent Windows review.

This order follows the project's direction: complete the Linux Milestone 5
acceptance and merge first, then apply the focused Windows compatibility phase.
No GPU, NPU, DirectML, CUDA, HIP, MLIR lowering, or unrelated Windows feature
belongs in this work.

## Verdict

Windows x64 portability is **not yet supported or validated** at this audit
base. The architecture has reusable semantic, IR, C ABI, and execution pieces,
but its driver, process layer, artifact model, topology backend, CMake/package
surface, tests, and CI still contain concrete Linux/POSIX assumptions. The
official LLVM 21.1.8 archive and the four isolated lanes above provide a
reproducible, bounded route to validation without weakening the completed
Linux milestones.
