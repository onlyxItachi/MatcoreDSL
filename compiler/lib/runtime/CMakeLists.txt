cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MatcoreRuntimeV0 LANGUAGES CXX)
  include(CTest)
endif()

option(
  MDSLC_BUILD_RUNTIME_BENCHMARKS
  "Build opt-in CPU runtime benchmarks (never registered with CTest)"
  OFF
)

if(NOT TARGET matcore_cpu_planner_v1)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../planner"
    "${CMAKE_CURRENT_BINARY_DIR}/matcore_cpu_planner_v1"
  )
endif()

if(MDSLC_BUILD_RUNTIME_BENCHMARKS)
  add_executable(matcore_cpu_planner_benchmark
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/cpu_planner_benchmark.cpp"
  )
  target_compile_features(matcore_cpu_planner_benchmark PRIVATE cxx_std_20)
  target_link_libraries(matcore_cpu_planner_benchmark
    PRIVATE MatcoreDSL::CpuPlannerV1
  )
endif()

add_library(matcore_runtime SHARED cpu_runtime.cpp)
add_library(MatcoreDSL::Runtime ALIAS matcore_runtime)
set_property(TARGET matcore_runtime PROPERTY EXPORT_NAME Runtime)

target_compile_features(matcore_runtime PUBLIC cxx_std_20)
target_compile_definitions(matcore_runtime PRIVATE MATCORE_RUNTIME_BUILD)
target_link_libraries(matcore_runtime PRIVATE MatcoreDSL::CpuPlannerV1)
# The planner may select the AVX2/FMA implementation at runtime. Debug and an
# unset single-config build type otherwise supply no optimization flag, which
# would leave this candidate as a scalar loop. Preserve the configured level
# for Release, RelWithDebInfo, and MinSizeRel.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(matcore_runtime PRIVATE "$<$<CONFIG:Debug>:-O2>")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "")
    target_compile_options(matcore_runtime PRIVATE -O2)
  endif()
endif()
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

if(BUILD_TESTING)
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
  add_test(NAME runtime.cpu.gemm_v0 COMMAND matcore_runtime_cpu_test)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" AND
     CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    find_program(MDSLC_OBJDUMP_EXECUTABLE NAMES objdump)
    find_program(MDSLC_NM_EXECUTABLE NAMES nm llvm-nm-21)
    if(MDSLC_OBJDUMP_EXECUTABLE AND MDSLC_NM_EXECUTABLE)
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
    endif()
  endif()
endif()
