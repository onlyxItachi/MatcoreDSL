cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MatcoreBenchV1 LANGUAGES CXX)
  include(CTest)
  include(GNUInstallDirs)
endif()
if(NOT COMMAND mdslc_target_enable_warnings)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/MatcoreDSLCompilerOptions.cmake")
endif()

if(NOT TARGET matcore_cpu_planner_v1)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/planner"
    "${CMAKE_CURRENT_BINARY_DIR}/matcore_cpu_planner_v1"
  )
endif()
if(NOT TARGET matcore_cpu_backends_v1)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/runtime"
    "${CMAKE_CURRENT_BINARY_DIR}/matcore_cpu_runtime"
  )
endif()

set(MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE "" CACHE STRING
  "Exact Git commit for source-archive benchmark provenance")
set(MDSLC_BENCH_SOURCE_STATE_OVERRIDE "" CACHE STRING
  "Source-archive state override: clean, dirty, or unknown")
if((MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE AND
    NOT MDSLC_BENCH_SOURCE_STATE_OVERRIDE) OR
   (MDSLC_BENCH_SOURCE_STATE_OVERRIDE AND
    NOT MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE))
  message(FATAL_ERROR
    "MDSLC benchmark provenance overrides must set both "
    "MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE and "
    "MDSLC_BENCH_SOURCE_STATE_OVERRIDE")
endif()
if(MDSLC_BENCH_SOURCE_STATE_OVERRIDE AND
   NOT MDSLC_BENCH_SOURCE_STATE_OVERRIDE MATCHES "^(clean|dirty|unknown)$")
  message(FATAL_ERROR
    "MDSLC_BENCH_SOURCE_STATE_OVERRIDE must be clean, dirty, or unknown")
endif()
if(MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE)
  string(LENGTH "${MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE}"
    MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE_LENGTH)
  if((NOT MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE_LENGTH EQUAL 40 AND
      NOT MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE_LENGTH EQUAL 64) OR
     NOT MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE MATCHES "^[0-9A-Fa-f]+$")
    message(FATAL_ERROR
      "MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE must be an exact 40- or 64-hex Git object ID")
  endif()
endif()

find_package(Git QUIET)
set(MATCORE_BENCH_PROVENANCE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(MATCORE_BENCH_PROVENANCE_HEADER
  "${MATCORE_BENCH_PROVENANCE_DIR}/matcore_bench_provenance.h")
add_custom_target(matcore_benchmark_provenance_v5
  COMMAND "${CMAKE_COMMAND}"
    "-DMATCORE_BENCH_PROVENANCE_OUTPUT=${MATCORE_BENCH_PROVENANCE_HEADER}"
    "-DMATCORE_BENCH_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    "-DMATCORE_BENCH_GIT_EXECUTABLE=${GIT_EXECUTABLE}"
    "-DMATCORE_BENCH_SOURCE_COMMIT_OVERRIDE=${MDSLC_BENCH_SOURCE_COMMIT_OVERRIDE}"
    "-DMATCORE_BENCH_SOURCE_STATE_OVERRIDE=${MDSLC_BENCH_SOURCE_STATE_OVERRIDE}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/GenerateBenchmarkProvenance.cmake"
  BYPRODUCTS "${MATCORE_BENCH_PROVENANCE_HEADER}"
  COMMENT "Refreshing exact matcore-bench source provenance"
  VERBATIM
)

string(TOUPPER "${CMAKE_BUILD_TYPE}" MATCORE_BENCH_BUILD_TYPE_UPPER)
set(MATCORE_BENCH_RECORDED_FLAGS "${CMAKE_CXX_FLAGS}")
if(MATCORE_BENCH_BUILD_TYPE_UPPER)
  string(APPEND MATCORE_BENCH_RECORDED_FLAGS
    " ${CMAKE_CXX_FLAGS_${MATCORE_BENCH_BUILD_TYPE_UPPER}}")
endif()
foreach(variable MATCORE_BENCH_RECORDED_FLAGS)
  string(REPLACE "\\" "\\\\" ${variable} "${${variable}}")
  string(REPLACE "\"" "\\\"" ${variable} "${${variable}}")
endforeach()

add_library(matcore_benchmark_core_v1 STATIC
  benchmark.cpp
  planner_runner.cpp
)
add_dependencies(matcore_benchmark_core_v1 matcore_benchmark_provenance_v5)
target_compile_features(matcore_benchmark_core_v1 PUBLIC cxx_std_20)
mdslc_target_enable_warnings(matcore_benchmark_core_v1 WERROR)
target_include_directories(matcore_benchmark_core_v1
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE "${MATCORE_BENCH_PROVENANCE_DIR}"
)
target_compile_definitions(matcore_benchmark_core_v1 PRIVATE
  "MATCORE_BENCH_BUILD_TYPE=\"$<CONFIG>\""
  "MATCORE_BENCH_COMPILER_FLAGS=\"${MATCORE_BENCH_RECORDED_FLAGS}\""
)
target_link_libraries(matcore_benchmark_core_v1
  PUBLIC MatcoreDSL::CpuPlannerV1 MatcoreDSL::CpuPlannerV3
  PRIVATE MatcoreDSL::CpuBackendsV1
)

add_executable(matcore-bench main.cpp)
target_compile_features(matcore-bench PRIVATE cxx_std_20)
mdslc_target_enable_warnings(matcore-bench WERROR)
target_link_libraries(matcore-bench PRIVATE matcore_benchmark_core_v1)
set_target_properties(matcore-bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(COMMAND install)
  install(TARGETS matcore-bench
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
endif()

if(BUILD_TESTING)
  add_executable(matcore_benchmark_core_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/benchmark/benchmark_core_test.cpp"
  )
  target_compile_features(matcore_benchmark_core_test PRIVATE cxx_std_20)
  mdslc_target_enable_warnings(matcore_benchmark_core_test WERROR)
  target_link_libraries(matcore_benchmark_core_test
    PRIVATE matcore_benchmark_core_v1
  )
  add_test(NAME benchmark.cpu.contract COMMAND matcore_benchmark_core_test)

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME benchmark.cpu.cli_json
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/benchmark/run_benchmark_cli_tests.py"
        --bench "$<TARGET_FILE:matcore-bench>"
        --schema
          "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/benchmark/matcore-bench-v6.schema.json"
    )

    add_test(
      NAME benchmark.cpu.provenance_incremental
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/benchmark/run_benchmark_provenance_tests.py"
        --cmake "${CMAKE_COMMAND}"
        --generator-script
          "${CMAKE_CURRENT_SOURCE_DIR}/GenerateBenchmarkProvenance.cmake"
    )

    add_test(
      NAME benchmark.cpu.deep_audit_runner_contract
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/performance/run_cpu_deep_audit_tests.py"
        --runner
          "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/performance/run_cpu_deep_audit.py"
        --bench "$<TARGET_FILE:matcore-bench>"
    )
  endif()
endif()
