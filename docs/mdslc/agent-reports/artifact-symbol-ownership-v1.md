# Compiler artifact symbol ownership — bounded link boundary

Starting checkpoint: `c4b56c0d3e9e41f7f73005ecc66d08c7797fc75b`; isolated branch
`mdslc/artifact-symbol-ownership-v1`. This utility checks the original, freshly
Clang-produced host LLVM module against compiler-owned immutable artifact bytes.
It is not a source frontend, imported-IR execution API or runtime dispatch policy.

## Counterexamples and chosen owner

**OBSERVED:** reserving only `matcore_runtime_`, `cblas_` and `openblas_` prefixes
misses provider-owned implementation symbols. The actual OpenBLAS 0.3.32 ELF
`cblas_sgemm` contains PLT references to `blas_memory_alloc`, `blas_memory_free`,
`gotoblas_corename` and `xerbla_`, with additional exported provider data.
A host definition of `blas_memory_alloc` can preserve real provider arithmetic
while introducing an otherwise unaccounted host effect.

The retained [manual reproducer](../../../compiler/tests/artifact_ownership/provider_internal_override.cpp)
performed one real 512x512 all-ones SGEMM on the local single-threaded provider:
`correct=1 hidden_host_calls=1`. An 80x80 probe produced correct arithmetic and
zero measured calls: that shape used another provider path. This is not evidence
that every GEMM allocates or calls that symbol. The first harness incorrectly
resolved the provider allocator only in main and crashed during pre-main provider
initialization; the retained harness resolves on first use. That harness failure
is not counted as a compiler defect or a successful final counterexample.

**OBSERVED:** file-scope Clang assembly can emit
`matcore_runtime_from_module_asm` in an actual object while the LLVM function list
contains only `main`. Function instruction assembly can likewise emit global
directives. Checking only LLVM functions therefore does not bound symbol
ownership. The [LLVM module-assembly contract](https://releases.llvm.org/21.1.0/docs/LangRef.html#module-level-inline-assembly)
explicitly describes this separate representation; LLVM's
[instruction assembly](https://releases.llvm.org/21.1.0/docs/LangRef.html#inline-assembler-expressions)
is another source of emitted assembly text.

**ARCHITECTURAL IMPLICATION:** this is a compiler-link ownership check, not more
mathematical source semantics. The implementation uses LLVM 21.1.8's
[ELF dynamic symbol interface](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/include/llvm/Object/ELFObjectFile.h)
and original-host global values. No private ELF parser or provider prefix list
is introduced.

An independent reviewer then falsified the first runtime namespace filter:
`int matcore::owned_template<int>(int)` has a return type before its demangled
namespace. The runtime's weak export was skipped and a matching host definition
was actually admitted. A parsed-name prototype also covered enclosing function
templates and local static guards, but was deliberately abandoned before any
commit: canonical Runtime actually exports only 15 C functions and already owns
its visibility boundary. Uniformly reserving every exact DSO export is both
stronger and smaller. No demangler, namespace heuristic or private C++ ABI parser
survives. The template, local-static value and guard refusals remain mechanical
tests; future accidental STL export collisions must fail closed and be corrected
in the owning DSO, not hidden by compiler heuristics.

## Mechanical contract

`verifyHostArtifactSymbolOwnership` accepts immutable `MemoryBufferRef` inputs
already pinned by its compiler caller, requires exactly one runtime ELF x86-64
DSO, and optionally inspects each selected provider DSO. It reserves every
defined nonlocal dynamic symbol in every trusted DSO uniformly, including data,
weak definitions, aliases and IFUNCs. It checks
original-host definitions across LLVM's whole global-value list. Declaration
references remain legal; local symbols cannot interpose a DSO. The actual
Runtime's hidden implementation exports let ordinary host STL code coexist.

Nonempty module/instruction assembly is deliberately unsupported in this initial
link contract; empty instruction barriers cannot define a symbol and are
accepted. Symbol-key matching accounts for Clang's explicit assembler-label
prefix and ELF version suffixes. On failure no successful ownership count is
returned. Unknown targets, malformed/non-DSO artifacts, missing/duplicate runtime
ownership and artifacts without exports for the declared role fail closed.

The caller must run this against the original host before introducing legitimate
compiler-owned helper definitions, bind the same exact bytes to actual linker
inputs, and recheck its source/toolchain/artifact closure before publication.
The fixed candidate archive must still be linked whole; ordinary strong-symbol
collision diagnostics protect its definitions. This utility does not replace
that step or authorize arbitrary artifact bytes supplied by a user.

## Validation

Clang/LLVM 21.1.8, Linux x86-64, warnings as errors:

- Final standalone Release: **68 checks, zero failures**, ownership and existing
  platform-support CTests **2/2 PASS**, 0.90 s; ownership test 0.89 s.
- Final whole-utility Debug ASan+UBSan: same **2/2 PASS**, 0.97 s; ownership test
  0.93 s.
- Both Release and ASan utility runs with actual Runtime and OpenBLAS DSO inputs:
  **72 checks, zero failures**. The actual runtime had exactly 15 public exports;
  ordinary vector-owning host code passed and a host `blas_memory_alloc`
  definition was refused against the real provider's exported symbol set.
- Repository hygiene and `git diff --check`: PASS.

The fixtures are actually compiled by Clang. They cover ordinary vector-owning
host code, deliberate overlapping STL-export refusal, canonical declarations, forged
Matcore C/C++ definitions, provider allocation/name functions, provider data,
weak symbols, templates, local static guards/values, aliases, IFUNCs, alternate
assembler names, both assembly forms,
asm-goto, local same-label definitions, provider-disabled mode and malformed or
incorrect artifact roles. Synthetic DSO exports test the ownership mechanism,
not provider numerical conformance. The manual real-provider probe above is
separate from the automated suite; no performance claim is made.

```sh
cmake -S compiler/tests/artifact_ownership -B build-ownership -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm
cmake --build build-ownership -- -j2
ctest --test-dir build-ownership -R '^codegen.artifact_symbol_ownership_v1$' \
  --output-on-failure -j1
```

Debug instrumentation uses `-O1 -g -fsanitize=address,undefined
-fno-omit-frame-pointer` for C/C++ and matching executable/shared linker flags.
The child Clang fixtures only produce inspected IR/DSOs; the utility inspecting
them is instrumented. Test environment is `DEBUGINFOD_URLS=`,
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`,
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

## Deliberate limits

[ELF permits executable interposition of DSO definitions](https://sourceware.org/binutils/docs/ld/Options.html#index-Bsymbolic).
The checked host must not define selected artifact-owned symbols, but compile-time
hashing does not authenticate future dynamic loading. In particular
[RUNPATH and loader environment selection](https://man7.org/linux/man-pages/man8/ld.so.8.html)
can select a different deployed object. Execution still requires stable trusted
runtime/provider/system-library dependencies, no foreign loader interposition,
conforming allocation hooks and existing valid-memory/thread contracts. This is
not a sandbox or a self-authenticating executable. Installing or validating the
connected driver remains the integration owner's separate gate.

Actual DSO input SHA-256 identities (artifact inspection, not new benchmark
evidence): Runtime `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014`;
OpenBLAS `be2e7d119279836105e0361be92c78dbcd8a6e7357e74339bdbb32a5195ee35e`.

Final production utility SHA-256 for independent review:

- `.cpp`: `5a19b042678354a8bd3e3b92ffe92f0bc7e836aa8e459e5cc6890e6f25fc4f38`.
- `.h`: `c441aee25a3d62b1534db696cbb90a8590866b97ef55b75b94d029d28b929b99`.
