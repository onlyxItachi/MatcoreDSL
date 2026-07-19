# MDSLC bootstrap status

Status date: 2026-07-19

Verified baseline: `351075e4d8af1880330b7c0474d701ca76776dfa`

Integration branch: `mdslc/bootstrap-v0`

Final validated implementation head: `b2b4706c44009a51ebb743e09e2b973fb1fcb796`

Independent review report commit: `be687f6f26812fdab6af22ee244061899fcf09d6`

## Current verdict

The standalone CPU vertical slice, normal native-artifact path, installation,
and external-consumer proof pass. The overall architecture proof is
**partially passed** because the installed machine lacks matching Clang 21
LibTooling development headers and libraries. The implemented, explicitly
labeled `clang-ast-json-bootstrap-v0` frontend does run Clang parsing and Sema,
uses AST declaration identity, authenticates the exact public header, and is
replaceable behind the frontend interface. It cannot read the
`AnnotateAttr("matcore.op.gemm")` payload or provide literal LibTooling
`getDirectCallee()` semantics, so Goal 3 is not represented as fully complete.

No Python, nanobind, MLIR, JIT cache, or legacy native extension participates
in the standalone compiler's normal execution path.

## Acceptance status

| Goal | State | Verified evidence |
| --- | --- | --- |
| Preflight and archaeology | Complete | Baseline, dirty original checkout, toolchain, devices, dependency gaps, and legacy reuse boundaries are recorded in `PREFLIGHT.md` and `LEGACY_REUSE_MAP.md` |
| Goal 1 architecture and skeleton | Complete | Independent `compiler/` CMake project builds 15 targets/steps without root CMake, Python, nanobind, or MLIR |
| Goal 2 valid-C++ `.mdsl` proof | Complete | `hello_host.mdsl` prints `5`; executable and ELF64 x86-64 relocatable modes pass through `mdslc++ -x c++` |
| Goal 3 post-Sema extraction | Partial | Clang 21 parsing/Sema, canonical AST declaration IDs, trusted-header equality, deterministic JSON and source diagnostics pass; exact annotation-payload authentication remains blocked on LibTooling development files |
| Goal 4 CPU GEMM vertical slice | Complete | Rewrite, JSON, sites, stubs, backend, three objects, `clang++ -r`, external ordinary link, runtime resolution, and independent-oracle execution pass |
| Goal 5 install/consumer | Complete | Installed tools/headers/runtime/CMake package and external `find_package` consumer configure, build, run, regenerate on `.mdsl` or included-header edits through a stable Ninja depfile, and return to a no-op build |
| Goal 6 validation and review | Complete for implemented scope | Release and Debug suites pass; generated-host/runtime sanitizer proofs pass; all independently reported high/medium implementation findings are fixed and covered by regressions |
| Goal 7 CUDA/cuBLAS | Not attempted | Optional GPU implementation was not started because the exact LibTooling frontend gate remains partial |

## Implemented pipeline

```text
valid C++ foo.mdsl
  -> mdslc++ (forces Clang 21 -x c++)
  -> Clang parsing/Sema and bounded structural AST JSON
  -> trusted canonical matcore::mdsl declaration recognition
  -> verified deterministic Matcore JSON IR v0
  -> exact main-file CallExpr rewrite from one stable source snapshot
  -> foo.host.cpp + foo.sites.h + foo.stubs.cpp + foo.backend.cpp
  -> original-source dependency scan and stable Ninja depfile when requested
  -> three ordinary Clang C++ objects
  -> clang++ -r combined relocatable object
  -> ordinary external clang++ link against libmatcore_runtime
  -> synchronous checked CPU f32 GEMM
```

`--save-temps` produces the five requested source/IR files, the three component
objects, and the combined `.o`. Generated call-site identifiers use the stable
source identity, exact source contents, call offset, and operation kind; tests
cover relative/absolute spelling, two sites, same-spelled sources in separate
directories, and multi-TU relocatable co-linking.

## Final validation evidence

Fresh final Release build: `/tmp/matcoredsl-final-release.u4Iyvv`

- Ninja: 15/15 steps.
- CTest at the final implementation head: 4/4 targets passed in 31.16 seconds.
- Frontend suite: 44/44 checks.
- Integration matrix: 63/63 active cases, 0 failures, 6 future capabilities
  explicitly not counted as passes.
- Installed consumer: configure/build/run, `.mdsl` rebuild, included-header
  semantic rebuild, stable depfile inspection, and no-op checks passed.
- Runtime suite: three GEMM shapes plus descriptor/policy failure contracts.

Debug validation tree: `/tmp/matcoredsl-final-debug.a2lVLg`

- Ninja: fresh 15/15 steps passed.
- CTest at the final implementation head: 4/4 targets passed in 60.13 seconds.
- Runtime dynamic exports: only `matcore_runtime_gemm_f32_v0`.

Sanitizer build: `/tmp/matcoredsl-final-sanitize.eqB21D`

- ASan+UBSan runtime build and test: 1/1 passed with leak detection enabled.
- Release extraction followed by sanitizer-instrumented generated
  host/stub/backend objects and the sanitizer-built runtime ran the GEMM
  example with no reported sanitizer finding.
- Instrumenting the bootstrap extractor itself reports pointer-arithmetic UB in
  Ubuntu's RapidJSON 1.1.0 header at `rapidjson/internal/stack.h:117`. This is
  recorded rather than counted as a clean full-frontend sanitizer result; it
  does not occur in the generated host/runtime acceptance path.

CPU result for `gemm_v0.mdsl`:

```text
host-before
MDSLC CPU GEMM PASS
```

The oracle accumulates in `double` with a different loop ordering from the
runtime. Runtime unit tests additionally cover 1x1x1, 2x3x2, and 3x2x4.

Artifact inspection confirms:

- `hello_host.o`: ELF64 x86-64 relocatable with `main` and `host_add<int>`;
- `gemm_v0.o`: ELF64 x86-64 `REL`, defining `main`, one stable C++ site
  wrapper, and one generated C backend entry;
- `matcore_runtime_gemm_f32_v0` remains unresolved in the combined object and
  is resolved by the ordinary final link;
- final program: dynamically linked ELF PIE;
- `ldd`: `libmatcore_runtime.so.0` resolves from the selected build/install
  tree;
- runtime shared object: versioned SONAME and one exported C implementation
  symbol.

## Reproducible commands

```sh
cmake -S compiler -B build-mdslc -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-mdslc -- -j2
ctest --test-dir build-mdslc --output-on-failure -j1

build-mdslc/bin/mdslc++ -std=c++20 \
  compiler/examples/hello_host.mdsl -o build-mdslc/hello_host
build-mdslc/bin/mdslc++ -std=c++20 -c \
  compiler/examples/hello_host.mdsl -o build-mdslc/hello_host.o

build-mdslc/bin/mdslc++ -std=c++20 --matcore-target=cpu \
  --save-temps -c compiler/examples/gemm_v0.mdsl \
  -o build-mdslc/gemm_v0.o
/usr/bin/clang++-21 build-mdslc/gemm_v0.o \
  -Lbuild-mdslc/lib -lmatcore_runtime \
  -Wl,-rpath,"$PWD/build-mdslc/lib" -o build-mdslc/gemm_v0
build-mdslc/gemm_v0

cmake --install build-mdslc --prefix /tmp/matcoredsl-install
cmake -S compiler/tests/consumer -B /tmp/matcoredsl-consumer -G Ninja \
  -DCMAKE_PREFIX_PATH=/tmp/matcoredsl-install \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21
cmake --build /tmp/matcoredsl-consumer -- -j2
/tmp/matcoredsl-consumer/matcore_consumer
```

## Toolchain and devices

- Selected host tuple: `/usr/bin/clang-21` and `/usr/bin/clang++-21`, version
  21.1.8.
- Matching Clang Tooling/ASTMatcher/Rewriter development headers and usable
  imported libraries: missing. No package was installed.
- Build parallelism: Ninja `-j2`, CTest/link validation `-j1`; ccache 4.12.3
  was used when discovered.
- CPU: AMD Ryzen AI 9 HX 370, 12 cores/24 threads; CPU path runtime-validated.
- NVIDIA RTX 4060 Laptop GPU, compute capability 8.9, CUDA 13.3: detected only;
  MDSLC CUDA not attempted.
- AMD `gfx1150`, HIP/ROCm 7.1: detected only; MDSLC HIP not attempted.
- RyzenAI `aie2p`: detected only; no MDSLC backend attempted.

## Legacy regression and original checkout

The standalone diff changes only `AGENTS.md`, `compiler/**`, `docs/mdslc/**`,
and ADR 0001; no legacy production source is modified. Pure-Python legacy
smoke tests passed 22 selected cases. Two additional selected tests failed
only at the known import boundary because `_matcore_native` is not built. A
root legacy CMake probe cannot configure: it requests MLIR 18.1.3 while only
MLIR 22.1.2 configuration is installed. The full legacy extension was
therefore not rebuilt or claimed green.

The original `feature/device-resident-tensors` checkout remains at the baseline
SHA with its 50 tracked deletions untouched. Final inspection also found an
untracked top-level `CMakeFiles/` there; preflight originally observed no
untracked status class, so this discrepancy is recorded and left untouched
rather than deleted without ownership certainty. The integration worktree is
clean.

## Known limitations and blockers

- The frontend is the documented AST-JSON bootstrap, not LibTooling. Exact
  annotation payload authentication remains the primary blocker.
- Rewritten calls use stable global site wrappers declared by a generated,
  include-free forward-declaration preamble followed by a `#line` reset. The
  rewrite therefore does not guess at textual include locations or reorder user
  includes/macros; named namespaces, active macro includes, inert duplicate/raw
  spellings, and multiline `__LINE__` preservation are regression-tested.
- Linux, Ninja, Clang 21, one `.mdsl` input, CPU, synchronous rank-2
  row-major-contiguous `f32` GEMM only.
- Public `matrix_view` is deliberately minimal and represents host f32 storage;
  no general tensor framework exists.
- Driver-managed shared/static/PIE modes and opaque `-Wl`/`-Xlinker`
  forwarding are rejected with an instruction to emit `-c` and perform the
  desired ordinary Clang link explicitly.
- Installed CMake integration performs one controlled Clang dependency scan
  over the original `.mdsl` and publishes a stable `-MD` depfile only after
  the native object succeeds. Generated temporary files are excluded from the
  depfile and protected from dependency-output collisions.
- `gemv`, `gevm`, `relu_gemm`, CUDA/cuBLAS, HIP, Metal, MLIR lowering, cost
  planning, fusion, and autotuning are not implemented.
- Generated multi-file publication is per-file atomic; a publication error can
  leave a partial saved-temp set, but the driver stops and never links it.

## CUDA status

Not attempted. There is no compile-only or runtime-validated MDSLC CUDA result.
Explicit CUDA requests return a nonzero no-fallback diagnostic and emit no CPU
artifact.
