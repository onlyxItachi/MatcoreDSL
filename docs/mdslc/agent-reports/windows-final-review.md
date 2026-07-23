# Windows x64 final adversarial review

Date: 2026-07-23

Reviewed product checkpoint:
`216c81210e2dcbc4599b384e99ceb90a91aab4ba`
(`mdslc/windows-x64-v1`)

Hosted evidence: GitHub Actions run
[`29972889899`](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/29972889899),
job `Windows x64 clang-cl 21.1.8`. The run completed successfully at the exact
reviewed checkpoint.

## Verdict

**Accept the Windows x64 compiler/runtime portability lane and recommend a
normal merge.**

No unresolved high- or medium-severity finding remains. The validated scope is
Windows Server 2025, `x86_64-pc-windows-msvc`, clang-cl/LLVM 21.1.8, lld-link,
MSVC tools 14.51.36231, and Windows SDK 10.0.26100.0. It proves the native
LibTooling frontend, generated COFF/PE pipeline, runtime DLL and C import
library, deterministic CPU planner, native AVX2 implementations, installed
package, external consumer, and ZIP distribution candidate. It is not a claim
of Windows performance calibration, AVX-512 runtime support, physical
multi-node NUMA behavior, OpenBLAS integration, or clean-machine installation.

## Rejection-oriented scope

The review attempted to reject the lane on:

- destructive output, depfile, or generated-artifact aliases of authenticated
  translation-unit dependencies;
- clang-cl option-case confusion, opaque argument forwarding, response-file
  injection, and extraction/compile divergence;
- POSIX or ELF assumptions in process launch and artifact production;
- a build-tree-only frontend, DLL, or package success;
- incorrect DLL/import-library exports or C++ ABI leakage;
- missing archive members in ordinary PE final links;
- unauthenticated third-party LLVM dependencies or missing license evidence;
- path, quoting, CRLF, Unicode, and space-bearing-prefix failures;
- falsely labeled AVX2 or AVX-512 functions;
- execution of AVX-512 on unsupported hardware;
- hidden OpenBLAS selection or native/provider oversubscription;
- synthetic topology evidence presented as physical NUMA validation;
- generated binaries or raw reports entering Git; and
- unsupported sanitizer or distribution claims.

## Findings and disposition

### Resolved High: dependency inputs could be destroyed before authentication

The review found that the driver cleared a requested output or dependency file
before it had captured and authenticated the translation unit's dependency
closure. A source could therefore include a path that was also selected as the
output, depfile, or deterministic `--save-temps` destination, allowing the
driver to destroy an input before rejecting the compilation.

Commit `fdd0c7b` fixes the ordering. The driver now captures the private
compiler dependency closure first, checks the requested output, requested
depfile, and every deterministic host/overlay/IR/sites/stubs/backend/object
destination against every dependency using physical path identity, and only
then removes or publishes a destination. Rejection occurs before output
mutation.

Focused regressions preserve sentinel bytes and reject:

- a requested output that is also an included dependency;
- a requested depfile that is also an included dependency;
- a generated host source that is also an included dependency; and
- a generated host source that is a hard-link alias of an included dependency.

Commit `3acf48e` adds the hosted Windows equivalents, including a filename-case
alias on NTFS plus direct output/dependency and depfile/dependency collisions.
The exact-tip Release native pipeline executed these guards successfully, and
no rejected case published an archive or secondary generated artifact.

### Retracted concern: `/C` versus `/c`

A proposed medium finding suggested that Windows option comparison might treat
clang-cl's preprocessing modifier `/C` as compile-only `/c`. Code tracing and
the hosted driver contract disproved it: compile-mode recognition uses the
case-sensitive clang-cl grammar, while `/C` is explicitly classified and
rejected as an unsafe compiler mode. The exact-tip Windows test exercised `/C`
and received the intended nonzero rejection. The concern was withdrawn; no
defect remains.

### Sanitizer boundary reviewed and accepted

Instrumenting the packaged LLVM Tooling executables themselves caused a real
link incompatibility: LLVM's static `LLVMSupport` brings its rpmalloc allocator,
which collided with clang-cl's Windows AddressSanitizer allocator thunks. The
lane therefore uses an explicit, narrower boundary rather than masking that
failure:

- the MDSLC runtime, planner, backends, and focused runtime test are built with
  clang-cl AddressSanitizer;
- the Release `mdslc++` and native extractor orchestrate compilation of
  AddressSanitizer-instrumented generated host, stub, and backend translation
  units and the final PE executable;
- the ASan DLL, import library, and static runtime thunk are selected from one
  authenticated LLVM 21.1.8 provider; and
- the packaged LLVM Tooling executables themselves remain validated in Release
  and Debug, but are not claimed as ASan-instrumented.

The focused runtime test passed **1/1**, and the generated ASan host/stub/backend
GEMM pipeline compiled, linked, executed, and reported `Windows native pipeline
PASS`. No Windows UBSan claim is made.

## Hosted validation evidence

### Release and Debug

- Release CTest: **35 passed + 1 intentional AVX-512 hardware skip out of 36**;
  zero failures or errors.
- Focused Debug CTest: **26 passed + 1 intentional AVX-512 hardware skip out of
  27**; zero failures or errors.
- Native frontend, typed IR, source rewrite, driver, generated GEMM, runtime,
  planner, package, C17 ABI, capability, topology, affinity, and repository
  hygiene tests passed.
- The installed runtime test resolved `matcore_runtime.dll` only through the
  isolated installed-prefix `PATH` and printed `runtime CPU GEMM v0: all tests
  passed`.

The skip is the fail-closed AVX-512 runtime gate. It is evidence that the hosted
CPU lacks usable AVX-512, not a test failure or runtime-validation claim.

### Native artifacts and ABI

- Six representative COFF/PE artifacts were authenticated, including generated
  objects/archive, ordinary PE executable, runtime DLL, and import library.
- Windows `/c` uses a normal `.lib` containing generated COFF objects; the final
  PE link retains those archive members explicitly and does not emulate ELF
  `clang++ -r`.
- `matcore_runtime.dll` exports exactly **15** declared `matcore_runtime_*` C
  symbols, with no C++ type crossing the stable ABI.
- Runtime import closure was walked recursively. All non-system dependencies
  had to be present and authenticated; Debug CRT imports were forbidden.
- The native frontend remained the explicit default; the bootstrap frontend
  was not built or selected silently.

### Exact ISA evidence and runtime status

- The isolated AVX2 microkernel contains **86 YMM register operands** and
  **8 packed-FMA instruction sites**.
- The isolated AVX-512 microkernel contains **33 ZMM register operands** and
  **4 packed-FMA instruction sites**.
- Reference, tiled, compiler-vectorized AVX2/FMA, native-packed AVX2/FMA, and
  native-parallel AVX2/FMA paths were runtime-validated on the hosted CPU.
- AVX-512 packed and parallel code compiled and passed exact disassembly checks,
  but hardware/OS/runtime capability remained unavailable. AVX-512 is therefore
  **compile/disassembly-only** on Windows in this review and was never executed
  or automatically selected.
- AVX-512 BF16, VNNI, and AMX optimized implementations remain unavailable and
  are not claimed.

### Installation, consumer, and distribution

- Installation to `MDSLC Windows ünicode install`, a Unicode and
  space-containing prefix, succeeded.
- All **17 installed files** were inventoried. The relocated external
  `find_package(MatcoreDSL REQUIRED)` consumer configured, regenerated its
  dependency graph, linked, and executed successfully from the installed
  prefix.
- All 17 installed/ZIP files were scanned in UTF-8, UTF-16LE, and UTF-16BE for
  three forbidden checkout/build roots; no absolute source or build path leaked.
- The uploaded ZIP candidate is **14,191,690 bytes** with SHA-256
  `b2c633192d3084585198f24eedba3957a85552c5d483d3b656bfdeda60480cd2`.
  The downloaded artifact was independently hashed to the same value and its
  payload inventory matched the hosted report.
- Repository hygiene passed after the hosted run; no generated binary, ZIP,
  cache, or raw validation output is tracked.

## Capability, provider, and topology boundaries

- The hosted runner exposed 4 logical processors, 2 physical cores, 1 socket,
  and 1 NUMA node. Windows topology discovery and affinity were physically
  exercised for that single-node topology.
- Multi-node placement and NUMA planning have synthetic tests only. No physical
  multi-node behavior or performance is claimed.
- OpenBLAS was intentionally omitted from the Windows package. Its registry
  candidate reports the adapter as unlinked and is rejected rather than
  silently selecting an external provider.
- The package declares the Microsoft Visual C++ Redistributable for Visual
  Studio 2015-2022 x64 as a user/deployer-owned external prerequisite; it is not
  bundled. A genuinely clean Windows machine installation was not validated.
- No Windows performance or scaling result was collected, so this review makes
  no GFLOP/s, crossover, planner-regret, or thread-scaling claim.

## Final recommendation

The resolved dependency-overwrite High has focused cross-platform regression
coverage; the `/C` concern was retracted on direct code and hosted-test evidence;
and the exact reviewed tip passed its complete hosted Windows Release, Debug,
focused ASan, artifact, install, consumer, ZIP, and hygiene gates. No unresolved
high- or medium-severity finding remains.

Merge the Windows x64 lane normally, preserving its focused commit history.
Keep the explicit hardware, sanitizer, provider, deployment, NUMA, and
performance limitations above in the public status; do not broaden them into
GPU or universal Windows support claims.
