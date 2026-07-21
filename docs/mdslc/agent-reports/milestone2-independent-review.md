# Milestone 2 Independent Review

## Findings

1. **P1, fixed: Debug/default builds could select a variant named
   compiler-vectorized while its emitted body was scalar.** Before the fix,
   a single-config build with no `CMAKE_BUILD_TYPE` compiled
   `cpu_runtime.cpp` without an optimization flag. On an AVX2/FMA host the
   planner selected `cpu.compiler-vectorized.avx2-fma.f32.v1`, but scoped
   disassembly contained scalar `vfmadd213ss` and no YMM instructions. That
   made the selected-lowering claim false in Debug/default builds.

   The integration resolution is at
   `compiler/lib/runtime/CMakeLists.txt:38-49` in `0ff17cd`: only Debug and
   an unset single-config build type receive `-O2`; Release retains its
   configured `-O3` and other optimized configurations retain their own
   flags. `compiler/tests/cmake/expect_cpu_vectorized_runtime.cmake:18-45`
   obtains the exact mangled `compiler_vectorized_gemm` symbol with `nm`,
   disassembles only that function, and requires both YMM and packed FMA.
   I made the same focused repair and regression test separately on this
   review branch as `ceb770b`; the integration branch independently contains
   the hardened equivalent in `97160b6`, `71b3a82`, and `0ff17cd`.

No other blocking correctness, ABI, verifier, planner-determinism, packaging,
or native-lowering defect was found in the reviewed integration range.

## Scope

- Base: `c025df534d11d1bc08285a174f2cd357aecadb0e`
- Initial validation head: `1ea6c24a7135d7f8b70d4bad82ba134cb9d9d8ae`
- Reviewed integration head: `0ff17cd4ac85f6e795dedd4e9fd3c2f363de37bb`
- The advanced `1ea6c24..0ff17cd` range was audited directly with
  `git show`/`git diff` after the local review branch diverged for the focused
  repair. This includes UTF-8 validation, dynamic-symbol scoping,
  malformed-capability rejection, benchmark hardening, Ubuntu 24.04 RapidJSON
  compatibility, the object-artifact test, and the conditional optimization
  fix.

The static audit covered typed/versioned v1 IR and the v0 bridge, verifier
contracts, capability discovery and malformed-record handling, fixed registry
ordering/cost diagnostics, C ABI error/no-allocation behavior, x86 guards,
install/export behavior, and selected CPU lowering. Relevant hardened code
includes `compiler/lib/ir/matcore_ir.cpp:440-451`,
`compiler/lib/ir/matcore_ir_v1_json.cpp:476-493`,
`compiler/lib/planner/cpu_planner.h:173-207`, and
`compiler/lib/runtime/cpu_runtime.cpp:127-216`.

## Targeted Evidence

All build directories below were newly created under `/tmp`; no existing
integration build tree was reused.

1. Fresh integration-head build:

   ```sh
   git archive --format=tar --output=/tmp/matcoredsl-m2-head.dQx7i1/source.tar 0ff17cd
   mkdir /tmp/matcoredsl-m2-head.dQx7i1/source
   tar -xf /tmp/matcoredsl-m2-head.dQx7i1/source.tar \
     -C /tmp/matcoredsl-m2-head.dQx7i1/source
   cmake -S /tmp/matcoredsl-m2-head.dQx7i1/source/compiler \
     -B /tmp/matcoredsl-m2-head.dQx7i1/build-release -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
     -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
     -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
     -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
     -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
     -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
     -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang
   cmake --build /tmp/matcoredsl-m2-head.dQx7i1/build-release -- -j2
   ```

   Result: configured with Clang 21.1.8 and completed 29/29 build steps.

2. Focused integration tests:

   ```sh
   ctest --test-dir /tmp/matcoredsl-m2-head.dQx7i1/build-release \
     -R '^(ir\.v1\.core|runtime\.cpu\.(benchmark_support|gemm_v0)|planner\.cpu\.cli\.(reference|automatic|invalid_alignment)|consumer\.installed)$' \
     --output-on-failure -j1
   ```

   Result: 7/7 passed: installed consumer, v1 IR, benchmark support, C ABI
   GEMM/runtime planner, and the three CLI planner cases.

3. RapidJSON compatibility branch, forced without modifying source:

   ```sh
   cmake -S /tmp/matcoredsl-m2-head.dQx7i1/source/compiler \
     -B /tmp/matcoredsl-m2-head.dQx7i1/build-rapidjson-compat -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
     -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
     -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
     -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
     -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
     -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
     -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
     -DMDSLC_RAPIDJSON_HEADER_COMPILES=FALSE
   cmake --build /tmp/matcoredsl-m2-head.dQx7i1/build-rapidjson-compat \
     --target matcore-extract matcore_ir_v1_tests -- -j2
   ctest --test-dir /tmp/matcoredsl-m2-head.dQx7i1/build-rapidjson-compat \
     -R '^ir\.v1\.core$' --output-on-failure -j1
   ```

   Result: the delayed-template-parsing probe succeeded, CMake reported its
   source-local compatibility mode, all RapidJSON-using translation units
   compiled with the narrow flags, and `ir.v1.core` passed.

4. Emitted v1 artifact and native execution:

   ```sh
   /tmp/matcoredsl-m2-head.dQx7i1/build-release/bin/matcore-extract \
     --frontend=native --ir-version=1 \
     --input /tmp/matcoredsl-m2-head.dQx7i1/source/compiler/tests/fixtures/positive/direct_gemm.mdsl \
     --ir-out /tmp/matcoredsl-m2-head.dQx7i1/direct-extract.v1.json \
     --rewrite-out /tmp/matcoredsl-m2-head.dQx7i1/direct-host.cpp \
     --sites-out /tmp/matcoredsl-m2-head.dQx7i1/direct-sites.h \
     --stubs-out /tmp/matcoredsl-m2-head.dQx7i1/direct-stubs.cpp \
     --backend-out /tmp/matcoredsl-m2-head.dQx7i1/direct-backend.cpp \
     -- /usr/bin/clang++-21 -x c++ -std=c++20 \
     -I/tmp/matcoredsl-m2-head.dQx7i1/source/compiler/include \
     /tmp/matcoredsl-m2-head.dQx7i1/source/compiler/tests/fixtures/positive/direct_gemm.mdsl
   /tmp/matcoredsl-m2-head.dQx7i1/build-release/bin/mdslc++ \
     --matcore-target=cpu --save-temps -std=c++20 \
     /tmp/matcoredsl-m2-head.dQx7i1/source/compiler/tests/fixtures/positive/direct_gemm.mdsl \
     -o /tmp/matcoredsl-m2-head.dQx7i1/direct-gemm-driver
   /tmp/matcoredsl-m2-head.dQx7i1/direct-gemm-driver
   /tmp/matcoredsl-m2-head.dQx7i1/build-release/bin/matcore-extract \
     --verify-ir /tmp/matcoredsl-m2-head.dQx7i1/direct-extract.v1.json
   /tmp/matcoredsl-m2-head.dQx7i1/build-release/bin/matcore-extract \
     --verify-ir /tmp/matcoredsl-m2-head.dQx7i1/direct-gemm-driver.matcore.json
   ```

   Result: the generated executable exited 0; both files verified as
   `Matcore IR v1: 1 operation(s)`. The actual JSON had schema
   `matcore.ir`, version `1`, canonical `matcore::mdsl::gemm`, f32 rank-2
   M/K/N operands, host/synchronous requirements, ordered read/write effects,
   output-vs-input no-alias requirements, and CPU/error policy. Generated
   code linked the expected `matcore_runtime_gemm_f32_v0` C ABI symbol.

5. Adversarial v1 artifacts made from that emitted JSON:

   ```sh
   jq '.version = 2' /tmp/matcoredsl-m2-head.dQx7i1/direct-gemm-driver.matcore.json \
     > /tmp/matcoredsl-m2-head.dQx7i1/ir-v2.json
   jq '.unexpected = true' /tmp/matcoredsl-m2-head.dQx7i1/direct-gemm-driver.matcore.json \
     > /tmp/matcoredsl-m2-head.dQx7i1/ir-extra-root.json
   jq '(.operations[0].operands[1].shape[0].symbol) = "m"' \
     /tmp/matcoredsl-m2-head.dQx7i1/direct-gemm-driver.matcore.json \
     > /tmp/matcoredsl-m2-head.dQx7i1/ir-bad-contraction.json
   /tmp/matcoredsl-m2-head.dQx7i1/build-release/bin/matcore-extract \
     --verify-ir /tmp/matcoredsl-m2-head.dQx7i1/ir-v2.json
   /tmp/matcoredsl-m2-head.dQx7i1/build-release/bin/matcore-extract \
     --verify-ir /tmp/matcoredsl-m2-head.dQx7i1/ir-extra-root.json
   /tmp/matcoredsl-m2-head.dQx7i1/build-release/bin/matcore-extract \
     --verify-ir /tmp/matcoredsl-m2-head.dQx7i1/ir-bad-contraction.json
   ```

   Result: each was rejected (exit 1) with, respectively, `unsupported
   Matcore IR version: 2`, `Matcore IR root has unexpected or missing fields`,
   and `gemm requires exact symbolic M/K/N shape relationships`.

6. Planner, ABI, symbol, and lowering evidence:

   ```sh
   /tmp/matcoredsl-m2-head.dQx7i1/build-release/bin/matcore-plan \
     --m 128 --k 256 --n 128 --alignment 64
   ctest --test-dir /tmp/matcoredsl-m2-debug.ioAlsL/build \
     -R '^runtime\.cpu\.(gemm_v0|vectorized_object)$' --output-on-failure -j1
   ctest --test-dir /tmp/matcoredsl-m2-review.Zh4Znk/build \
     -R '^runtime\.cpu\.(gemm_v0|vectorized_object)$' --output-on-failure -j1
   cmake -DMDSLC_OBJDUMP=/usr/bin/objdump -DMDSLC_NM=/usr/bin/nm \
     -DMDSLC_RUNTIME=/tmp/matcoredsl-m2-review.Zh4Znk/build/lib/libmatcore_runtime.so \
     -P /tmp/matcoredsl-m2-review.Zh4Znk/expect_cpu_vectorized_runtime.integration.cmake
   cmake -DMDSLC_OBJDUMP=/usr/bin/objdump -DMDSLC_NM=/usr/bin/nm \
     -DMDSLC_RUNTIME=/tmp/matcoredsl-m2-debug.ioAlsL/build/lib/libmatcore_runtime.so \
     -P /tmp/matcoredsl-m2-review.Zh4Znk/expect_cpu_vectorized_runtime.integration.cmake
   /usr/bin/clang++-21 -std=c++20 -Wall -Wextra -Wpedantic -Werror \
     -Icompiler/include -Icompiler/lib/planner \
     /tmp/matcoredsl-m2-review.Zh4Znk/runtime_abi_planner_probe.cpp \
     -L/tmp/matcoredsl-m2-default.KH1D1p/build/lib -lmatcore_runtime \
     -Wl,-rpath,/tmp/matcoredsl-m2-default.KH1D1p/build/lib \
     -o /tmp/matcoredsl-m2-review.Zh4Znk/runtime_abi_planner_probe
   /tmp/matcoredsl-m2-review.Zh4Znk/runtime_abi_planner_probe
   ```

   Result: automatic planning selected
   `cpu.compiler-vectorized.avx2-fma.f32.v1` deterministically with costs
   `33554432`, `16781312`, and `8404992` for reference/tiled/vectorized.
   Debug and Release each passed 2/2 runtime tests. The exact `0ff17cd`
   function-scoped artifact script passed against both artifacts. Release's
   compile command contained `-O3` and no trailing `-O2`; Debug's contained
   `-g` and `-O2`. Scoped function disassembly contained `vbroadcastss`,
   YMM `vmovups`, and `vfmadd213ps`.

   A standalone C++ ABI probe over the runtime C ABI also passed. It used a
   global `operator new` counter and verified no C++ allocation during plan or
   execution, no exception crossing the C ABI, correct vector-plan selection,
   no planning writes, scalar-oracle output equality, rejection of nonempty
   report input, and rejection of incoherent synthetic aarch64+AVX2 records.
   Dynamic-symbol inspection exposed only
   `matcore_runtime_gemm_f32_v0` and
   `matcore_runtime_plan_gemm_f32_v1`; no `malloc`, `free`, or C++ allocation
   symbol was imported by the runtime.

## Residual Risks

- Native execution and object inspection were run on x86_64 with AVX2/FMA.
  aarch64, MSVC, and non-AVX hardware are covered by source guards and
  synthetic capability records, not by native hardware in this review.
- The RapidJSON fallback was deliberately forced on this host; it validates
  the configure branch and source-local flags but is not a live Ubuntu 24.04
  reproduction of the original incompatible header.
- The object-artifact test intentionally runs only where GNU/Clang `nm` and
  `objdump` are available on x86_64. Other toolchains need an equivalent
  artifact-level guard if they later gain a selectable vector implementation.
- No profiler output was captured. Performance profiling was optional and
  would duplicate existing benchmark work, so no perfdigest invocation was
  needed.

## Verdict

**PASS.** The one material lowering defect was fixed and independently
verified. Merge `mdslc/matcore-ir-v1-cpu-planner` at `0ff17cd` (or its
descendant containing the same changes) is recommended for the stated CPU-only
Milestone 2 scope. Do not cherry-pick reviewer commit `ceb770b` onto that
integration head: its focused runtime fix is already present there.
