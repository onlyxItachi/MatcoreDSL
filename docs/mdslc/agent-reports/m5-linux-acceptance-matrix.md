# Independent Linux Milestone 5 acceptance audit

Audit target: `37d8ea96c88c1274388b190acdb2e851e9573afd`

The integration worktree was clean at audit start. While this audit was in
flight, the lead added documentation-only commits `2d05040` and `95fdce2`.
`git diff 37d8ea9..95fdce2` contains only documentation. No production or test
source changed, and this lane made no repository edits before this report.

## Fresh Release with required OpenBLAS

Build directory: `/tmp/matcore-m5-linux-audit-37d8-release`

```sh
cmake -S compiler -B /tmp/matcore-m5-linux-audit-37d8-release -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=ON \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=ON -DMDSLC_REQUIRE_OPENBLAS=ON \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/matcore-m5-linux-audit-37d8-release -- -j2
ctest --test-dir /tmp/matcore-m5-linux-audit-37d8-release \
  --output-on-failure -j1
```

Result: Clang 21.1.8; authenticated OpenBLAS 0.3.32 via pkg-config; 89/89
Ninja actions; **42/42 CTest passed** in 65.01 s.

## Fresh Debug with required OpenBLAS

Build directory during validation:
`/tmp/matcore-m5-linux-audit-37d8-debug` (removed after completion to free the
constrained `/tmp` tmpfs; all three logs remain beside it).

The configure command was identical to Release except
`-DCMAKE_BUILD_TYPE=Debug`. Build used `-j2`; CTest used `-j1`.

Result: Clang 21.1.8; authenticated OpenBLAS 0.3.32; 89/89 Ninja actions;
**42/42 CTest passed** in 85.79 s.

## ASan and UBSan

Build directory during validation:
`/tmp/matcore-m5-linux-audit-37d8-asan` (removed after completion; logs remain).

```sh
cmake -S compiler -B /tmp/matcore-m5-linux-audit-37d8-asan -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
  -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF \
  -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
  -DMDSLC_ENABLE_OPENBLAS=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_CXX_FLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build /tmp/matcore-m5-linux-audit-37d8-asan -- -j2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir /tmp/matcore-m5-linux-audit-37d8-asan \
  -R '<supported IR/platform/runtime/planner/benchmark regex>' \
  --output-on-failure -j1
```

Result: **31/31 supported tests passed** in 2.37 s, including IR v1,
platform/capability/topology/affinity, typed reference APIs, AVX2 and AVX-512
packed correctness, persistent executor, parallel GEMM, planner resources,
workspace, public C context, legacy GEMM, planner CLI, and benchmark/provenance.
No ASan, leak, or UBSan diagnostic was emitted.

## TSan, OpenBLAS disabled

Build directory: `/tmp/matcore-m5-linux-audit-37d8-tsan`

The fresh configure used Debug, native frontend disabled, OpenBLAS disabled,
and `-O1 -g -fsanitize=thread -fno-omit-frame-pointer` on C/C++ with the
corresponding executable/shared linker flags. Only
`matcore_execution_context_test`, `matcore_planner_v3_resources_test`, and
`matcore_public_context_c_api_test` were built, using `-j1` after a first
write failed because the shared `/tmp` tmpfs reported quota exhaustion.
Removing only this lane's completed Debug and ASan build directories provided
space; the exact retry compiled normally.

```sh
TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1 \
ctest --test-dir /tmp/matcore-m5-linux-audit-37d8-tsan \
  -R '^runtime\.(cpu\.(execution_context\.v1|planner_v3_resources)|c_abi\.public_context_v1)$' \
  --output-on-failure -j1
```

Result: **3/3 passed**, no TSan diagnostic.

## OpenBLAS-disabled product matrix

Build directory: `/tmp/matcore-m5-linux-audit-37d8-openblas-off`

Fresh Release configure used Clang 21, bootstrap compatibility frontend,
`MDSLC_ENABLE_NATIVE_FRONTEND=OFF`, and `MDSLC_ENABLE_OPENBLAS=OFF`.

Result: 87/87 Ninja actions; **37/37 CTest passed**. The forced
`cpu.external.openblas.f32.v1` plan exited 1 and reported
`OpenBLAS CBLAS adapter is not linked`; it emitted no selected fallback.

## Install, package, consumer, ABI, and artifact proof

- Release `consumer.installed` plus `package.installed_c17_abi`: **2/2 passed**
  on a verbose repeat. The relocated external consumer configured with
  `find_package`, compiled `.mdsl`, linked, ran, regenerated on source/header
  changes, and returned to a no-op build. Strict C17 compile/link/run
  authenticated **15 public C exports**.
- A separate install prefix containing spaces at
  `/tmp/matcore m5 linux audit 37d8 install` contains `mdslc++`,
  `matcore-extract`, `matcore-plan`, `matcore-bench`, public headers,
  versioned `libmatcore_runtime.so`, and CMake package files. Installed CMake
  and binary scans found no checkout/build path leakage. Installed runtime has
  SONAME `libmatcore_runtime.so.0` and no absolute RUNPATH; installed
  `matcore-plan` uses `$ORIGIN/../lib`.
- Release runtime exports exactly 15 dynamic C symbols and has no Python or
  nanobind dependency/string marker.
- Exact AVX2 microkernel
  `matcore_cpu_packed_avx2_4x16_microkernel_f32_v1`: 54 YMM operands,
  8 packed FMA instruction sites, 0 ZMM operands.
- Exact AVX-512 microkernel
  `matcore_cpu_packed_avx512_4x16_microkernel_f32_v1`: 21 ZMM operands,
  4 packed FMA instruction sites, 0 YMM operands.
- Native end-to-end proof:

```sh
/tmp/matcore-m5-linux-audit-37d8-release/bin/mdslc++ \
  -std=c++20 --matcore-target=cpu --save-temps -c \
  compiler/examples/gemm_v0.mdsl \
  -o /tmp/matcore-m5-linux-audit-37d8-e2e/gemm_v0.o
/usr/bin/clang++-21 /tmp/matcore-m5-linux-audit-37d8-e2e/gemm_v0.o \
  -L/tmp/matcore-m5-linux-audit-37d8-release/lib -lmatcore_runtime \
  -Wl,-rpath,/tmp/matcore-m5-linux-audit-37d8-release/lib \
  -o /tmp/matcore-m5-linux-audit-37d8-e2e/gemm_v0
/tmp/matcore-m5-linux-audit-37d8-e2e/gemm_v0
```

Output: `host-before` then `MDSLC CPU GEMM PASS`. `file` and `readelf` identify
the combined object as ELF64 x86-64 `REL`; `nm -C` finds `main`, the generated
call-site/backend symbols, and unresolved `matcore_runtime_gemm_f32_v0` before
ordinary final link. All saved host/IR/sites/stubs/backend component sources
and objects are present. `ldd` resolves the versioned runtime and OpenBLAS.

## Hygiene and legacy checks

- `bash tests/check_repository_hygiene.sh`: passed.
- `python3 tests/test_frontend_contract.py`: passed all 14 defined legacy
  contract cases.
- `git diff --check`: passed.
- A first rebuild after the two concurrent documentation commits correctly
  refreshed benchmark source provenance and only the affected benchmark
  targets. The immediately repeated build left provenance header and benchmark
  object mtimes unchanged and executed only the provenance refresh action.

## Verdict

No product high- or medium-severity failure was found. The one observed
infrastructure interruption was a transient `/tmp` write-quota failure during
the first TSan compile; the exact single-job retry succeeded and its complete
focused test matrix passed.
