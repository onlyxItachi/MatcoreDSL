cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MatcoreRuntimeV0 LANGUAGES CXX)
  include(CTest)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/MatcoreDSLCompilerOptions.cmake")
endif()

option(
  MDSLC_BUILD_RUNTIME_BENCHMARKS
  "Build opt-in CPU runtime benchmarks (never registered with CTest)"
  OFF
)

set(MDSLC_OPENBLAS_DEFAULT ON)
if(WIN32)
  set(MDSLC_OPENBLAS_DEFAULT OFF)
endif()
option(
  MDSLC_ENABLE_OPENBLAS
  "Enable the optional authenticated OpenBLAS/CBLAS CPU variant"
  ${MDSLC_OPENBLAS_DEFAULT}
)
option(
  MDSLC_REQUIRE_OPENBLAS
  "Fail configuration when the optional OpenBLAS development surface is absent"
  OFF
)

if(WIN32 AND MDSLC_ENABLE_OPENBLAS)
  message(FATAL_ERROR
    "The Windows MDSLC lane has no authenticated CBLAS provider; configure "
    "with MDSLC_ENABLE_OPENBLAS=OFF")
endif()

set(MDSLC_HAS_OPENBLAS OFF)
if(MDSLC_ENABLE_OPENBLAS)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(MDSLC_OPENBLAS QUIET IMPORTED_TARGET openblas)
  endif()
  if(MDSLC_OPENBLAS_FOUND)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES PkgConfig::MDSLC_OPENBLAS)
    check_cxx_source_compiles(
      "#include <cblas.h>\n#include <climits>\nstatic_assert(sizeof(blasint) == sizeof(int));\nint main() { float a[1]={1}, b[1]={1}, c[1]={0}; int previous=openblas_set_num_threads_local(1); cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,1,1,1.0f,a,1,b,1,0.0f,c,1); openblas_set_num_threads_local(previous); return c[0] == 1.0f ? 0 : 1; }"
      MDSLC_OPENBLAS_CBLAS_COMPILES
    )
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(MDSLC_OPENBLAS_CBLAS_COMPILES)
      set(MDSLC_HAS_OPENBLAS ON)
      message(STATUS
        "MDSLC OpenBLAS variant: ${MDSLC_OPENBLAS_VERSION} via pkg-config")
    endif()
  endif()
endif()
if(MDSLC_REQUIRE_OPENBLAS AND NOT MDSLC_HAS_OPENBLAS)
  message(FATAL_ERROR
    "MDSLC_REQUIRE_OPENBLAS=ON requires a coherent LP64 OpenBLAS CBLAS provider")
endif()
if(NOT MDSLC_HAS_OPENBLAS)
  message(STATUS "MDSLC OpenBLAS variant: unavailable (optional)")
endif()

if(NOT TARGET matcore_cpu_planner_v1)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../planner"
    "${CMAKE_CURRENT_BINARY_DIR}/matcore_cpu_planner_v1"
  )
endif()

find_package(Threads REQUIRED)

if(MDSLC_BUILD_RUNTIME_BENCHMARKS)
  add_executable(matcore_cpu_planner_benchmark
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/cpu_planner_benchmark.cpp"
  )
  target_compile_features(matcore_cpu_planner_benchmark PRIVATE cxx_std_20)
  target_link_libraries(matcore_cpu_planner_benchmark
    PRIVATE MatcoreDSL::CpuPlannerV1
  )
  mdslc_target_enable_debug_optimization(matcore_cpu_planner_benchmark)
endif()

add_library(matcore_cpu_backends_v1 STATIC
  cpu_backend_registry.cpp
  cpu_execution_context.cpp
  cpu_numeric_reference.cpp
  cpu_openblas.cpp
  cpu_parallel_gemm.cpp
  cpu_planner_v3_resources.cpp
  cpu_runtime_validation.cpp
  cpu_gemm_backend.cpp
  cpu_packed_avx2.cpp
  cpu_packed_avx512.cpp
)
add_library(MatcoreDSL::CpuBackendsV1 ALIAS matcore_cpu_backends_v1)
target_compile_features(matcore_cpu_backends_v1 PUBLIC cxx_std_20)
target_include_directories(matcore_cpu_backends_v1 PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(matcore_cpu_backends_v1
  PUBLIC MatcoreDSL::CpuPlannerV1 MatcoreDSL::CpuPlannerV3
  PRIVATE Threads::Threads MatcoreDSL::PlatformV1
)
set_target_properties(matcore_cpu_backends_v1 PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  POSITION_INDEPENDENT_CODE ON
)

add_library(matcore_runtime SHARED cpu_runtime.cpp)
add_library(MatcoreDSL::Runtime ALIAS matcore_runtime)
set_property(TARGET matcore_runtime PROPERTY EXPORT_NAME Runtime)

target_compile_features(matcore_runtime PUBLIC cxx_std_20)
target_compile_definitions(matcore_runtime PRIVATE MATCORE_RUNTIME_BUILD)
target_link_libraries(matcore_runtime PRIVATE MatcoreDSL::CpuPlannerV1)
if(MDSLC_HAS_OPENBLAS)
  target_link_libraries(matcore_cpu_backends_v1
    PRIVATE PkgConfig::MDSLC_OPENBLAS
  )
  target_compile_definitions(matcore_cpu_backends_v1 PUBLIC
    MATCORE_MDSLC_HAS_OPENBLAS=1
    "MATCORE_MDSLC_OPENBLAS_VERSION=\"${MDSLC_OPENBLAS_VERSION}\""
  )
else()
  target_compile_definitions(matcore_cpu_backends_v1 PUBLIC
    MATCORE_MDSLC_HAS_OPENBLAS=0
  )
endif()
target_link_libraries(matcore_runtime PRIVATE MatcoreDSL::CpuBackendsV1)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Backend implementations live in a private static archive. libstdc++ marks
  # some inline templates with default visibility even under
  # -fvisibility=hidden, so Debug links can otherwise leak weak C++ symbols.
  # Keep the shared runtime boundary limited to MATCORE_RUNTIME_API C symbols.
  target_link_options(matcore_runtime PRIVATE "LINKER:--exclude-libs,ALL")
endif()
# The planner may select the AVX2/FMA implementation at runtime. Debug and an
# unset single-config build type otherwise supply no optimization flag, which
# would leave this candidate as a scalar loop. Preserve the configured level
# for Release, RelWithDebInfo, and MinSizeRel.
mdslc_target_enable_debug_optimization(matcore_runtime)
target_include_directories(matcore_runtime
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>"
    "$<INSTALL_INTERFACE:include>"
)
set_target_properties(matcore_runtime PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  OUTPUT_NAME matcore_runtime
  SOVERSION 0
  VERSION 0.0.0
)
if(WIN32)
  # MATCORE_RUNTIME_API provides the only exported DLL surface.  CMake places
  # the DLL in bin/ and its import library in lib/ through the global output
  # directories and the root install(TARGETS) contract.
  set_target_properties(matcore_runtime PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    PREFIX ""
  )
endif()

if(BUILD_TESTING)
  add_executable(matcore_numeric_reference_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/types/reference_types_test.cpp"
  )
  target_compile_features(matcore_numeric_reference_test PRIVATE cxx_std_20)
  target_include_directories(matcore_numeric_reference_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(matcore_numeric_reference_test
    PRIVATE MatcoreDSL::CpuBackendsV1
  )
  mdslc_target_enable_warnings(matcore_numeric_reference_test WERROR)
  add_test(NAME runtime.cpu.reference_types
    COMMAND matcore_numeric_reference_test
  )

  add_executable(matcore_typed_runtime_c_api_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/typed_runtime_c_api_test.cpp"
  )
  target_compile_features(matcore_typed_runtime_c_api_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_typed_runtime_c_api_test
    PRIVATE MatcoreDSL::Runtime
  )
  mdslc_target_enable_warnings(matcore_typed_runtime_c_api_test WERROR)
  add_test(NAME runtime.c_abi.typed_reference_v1
    COMMAND matcore_typed_runtime_c_api_test
  )

  add_executable(matcore_packed_avx2_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/packed/packed_gemm_test.cpp"
  )
  target_compile_features(matcore_packed_avx2_test PRIVATE cxx_std_20)
  target_include_directories(matcore_packed_avx2_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(matcore_packed_avx2_test
    PRIVATE MatcoreDSL::CpuBackendsV1
  )
  mdslc_target_enable_warnings(matcore_packed_avx2_test WERROR)
  add_test(NAME runtime.cpu.packed_avx2
    COMMAND matcore_packed_avx2_test
  )

  add_executable(matcore_packed_avx512_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/packed/packed_avx512_gemm_test.cpp"
  )
  target_compile_features(matcore_packed_avx512_test PRIVATE cxx_std_20)
  target_include_directories(matcore_packed_avx512_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(matcore_packed_avx512_test
    PRIVATE MatcoreDSL::CpuBackendsV1
  )
  mdslc_target_enable_warnings(matcore_packed_avx512_test WERROR)
  add_test(NAME runtime.cpu.packed_avx512
    COMMAND matcore_packed_avx512_test
  )
  set_tests_properties(runtime.cpu.packed_avx512 PROPERTIES
    SKIP_RETURN_CODE 77
  )

  add_executable(matcore_execution_context_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/parallel/execution_context_test.cpp"
  )
  target_compile_features(matcore_execution_context_test PRIVATE cxx_std_20)
  target_include_directories(matcore_execution_context_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  mdslc_target_enable_warnings(matcore_execution_context_test WERROR)
  target_link_libraries(matcore_execution_context_test
    PRIVATE MatcoreDSL::CpuBackendsV1 MatcoreDSL::PlatformV1
  )
  add_test(NAME runtime.cpu.execution_context.v1
    COMMAND matcore_execution_context_test
  )
  set_tests_properties(runtime.cpu.execution_context.v1 PROPERTIES
    TIMEOUT 10
  )

  add_executable(matcore_parallel_packed_gemm_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/parallel/parallel_packed_gemm_test.cpp"
  )
  target_compile_features(matcore_parallel_packed_gemm_test PRIVATE cxx_std_20)
  target_include_directories(matcore_parallel_packed_gemm_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  mdslc_target_enable_warnings(matcore_parallel_packed_gemm_test WERROR)
  target_link_libraries(matcore_parallel_packed_gemm_test
    PRIVATE MatcoreDSL::CpuBackendsV1
  )
  add_test(NAME runtime.cpu.parallel_packed.v1
    COMMAND matcore_parallel_packed_gemm_test
  )

  add_executable(matcore_planner_v3_resources_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/parallel/planner_v3_resources_test.cpp"
  )
  target_compile_features(matcore_planner_v3_resources_test PRIVATE cxx_std_20)
  target_include_directories(matcore_planner_v3_resources_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  mdslc_target_enable_warnings(matcore_planner_v3_resources_test WERROR)
  target_link_libraries(matcore_planner_v3_resources_test
    PRIVATE MatcoreDSL::CpuBackendsV1 MatcoreDSL::PlatformV1
  )
  add_test(NAME runtime.cpu.planner_v3_resources
    COMMAND matcore_planner_v3_resources_test
  )

  add_executable(matcore_workspace_runtime_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/workspace_runtime_test.cpp"
  )
  target_compile_features(matcore_workspace_runtime_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_workspace_runtime_test
    PRIVATE MatcoreDSL::Runtime
  )
  add_test(NAME runtime.cpu.workspace_v1
    COMMAND matcore_workspace_runtime_test
  )

  add_executable(matcore_openblas_adapter_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/openblas_adapter_test.cpp"
  )
  target_compile_features(matcore_openblas_adapter_test PRIVATE cxx_std_20)
  target_include_directories(matcore_openblas_adapter_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../planner"
  )
  target_link_libraries(matcore_openblas_adapter_test
    PRIVATE MatcoreDSL::CpuBackendsV1
  )
  add_test(NAME runtime.cpu.openblas_adapter
    COMMAND matcore_openblas_adapter_test
  )

  add_executable(matcore_runtime_abi_compat_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/runtime_abi_compat_test.cpp"
  )
  target_compile_features(matcore_runtime_abi_compat_test PRIVATE cxx_std_20)
  target_include_directories(matcore_runtime_abi_compat_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  )
  add_test(NAME runtime.c_abi.compatibility_v1
    COMMAND matcore_runtime_abi_compat_test
  )

  add_executable(matcore_public_context_c_api_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/public_context_c_api_test.cpp"
  )
  target_compile_features(matcore_public_context_c_api_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_public_context_c_api_test
    PRIVATE MatcoreDSL::Runtime
  )
  mdslc_target_enable_warnings(matcore_public_context_c_api_test WERROR)
  add_test(NAME runtime.c_abi.public_context_v1
    COMMAND matcore_public_context_c_api_test
  )

  add_executable(matcore_cpu_benchmark_support_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/cpu_benchmark_support_test.cpp"
  )
  target_compile_features(matcore_cpu_benchmark_support_test PRIVATE cxx_std_20)
  add_test(NAME runtime.cpu.benchmark_support
    COMMAND matcore_cpu_benchmark_support_test
  )

  add_executable(matcore_runtime_cpu_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/runtime_cpu_test.cpp"
  )
  target_compile_features(matcore_runtime_cpu_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_runtime_cpu_test
    PRIVATE MatcoreDSL::Runtime MatcoreDSL::CpuPlannerV1
  )
  mdslc_target_enable_debug_optimization(matcore_runtime_cpu_test)
  add_test(NAME runtime.cpu.gemm_v0 COMMAND matcore_runtime_cpu_test)

  if(NOT WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" AND
     NOT CMAKE_CXX_FLAGS MATCHES "fsanitize" AND
     CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    find_program(MDSLC_OBJDUMP_EXECUTABLE NAMES objdump)
    find_program(MDSLC_NM_EXECUTABLE NAMES nm llvm-nm-21)
    if(MDSLC_OBJDUMP_EXECUTABLE AND MDSLC_NM_EXECUTABLE)
      find_package(Python3 COMPONENTS Interpreter QUIET)
      add_test(
        NAME runtime.cpu.packed_avx2_object
        COMMAND
          "${CMAKE_COMMAND}"
          "-DMDSLC_OBJDUMP=${MDSLC_OBJDUMP_EXECUTABLE}"
          "-DMDSLC_NM=${MDSLC_NM_EXECUTABLE}"
          "-DMDSLC_PACKED_ARTIFACT=$<TARGET_FILE:matcore_runtime>"
          -P
          "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cmake/expect_cpu_packed_avx2_microkernel.cmake"
      )
      add_test(
        NAME runtime.cpu.vectorized_object
        COMMAND
          "${CMAKE_COMMAND}"
          "-DMDSLC_OBJDUMP=${MDSLC_OBJDUMP_EXECUTABLE}"
          "-DMDSLC_NM=${MDSLC_NM_EXECUTABLE}"
          "-DMDSLC_RUNTIME=$<TARGET_FILE:matcore_runtime>"
          -P
          "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cmake/expect_cpu_vectorized_runtime.cmake"
      )
      if(Python3_Interpreter_FOUND)
        add_test(
          NAME runtime.cpu.packed_avx512_object
          COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/packed/check_avx512_microkernel.py"
            --artifact "$<TARGET_FILE:matcore_cpu_backends_v1>"
            --objdump "${MDSLC_OBJDUMP_EXECUTABLE}"
            --nm "${MDSLC_NM_EXECUTABLE}"
        )
      endif()
    endif()
  endif()
endif()
