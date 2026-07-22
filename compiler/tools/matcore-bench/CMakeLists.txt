cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MatcoreBenchV1 LANGUAGES CXX)
  include(CTest)
  include(GNUInstallDirs)
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

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  OUTPUT_VARIABLE MATCORE_BENCH_SOURCE_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE MATCORE_BENCH_GIT_RESULT
)
if(NOT MATCORE_BENCH_GIT_RESULT EQUAL 0)
  set(MATCORE_BENCH_SOURCE_COMMIT "unknown")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" MATCORE_BENCH_BUILD_TYPE_UPPER)
set(MATCORE_BENCH_RECORDED_FLAGS "${CMAKE_CXX_FLAGS}")
if(MATCORE_BENCH_BUILD_TYPE_UPPER)
  string(APPEND MATCORE_BENCH_RECORDED_FLAGS
    " ${CMAKE_CXX_FLAGS_${MATCORE_BENCH_BUILD_TYPE_UPPER}}")
endif()
foreach(variable MATCORE_BENCH_RECORDED_FLAGS MATCORE_BENCH_SOURCE_COMMIT)
  string(REPLACE "\\" "\\\\" ${variable} "${${variable}}")
  string(REPLACE "\"" "\\\"" ${variable} "${${variable}}")
endforeach()

add_library(matcore_benchmark_core_v1 STATIC
  benchmark.cpp
  planner_runner.cpp
)
target_compile_features(matcore_benchmark_core_v1 PUBLIC cxx_std_20)
if(MSVC)
  target_compile_options(matcore_benchmark_core_v1 PRIVATE /W4 /WX)
else()
  target_compile_options(matcore_benchmark_core_v1 PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
endif()
target_include_directories(matcore_benchmark_core_v1
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(matcore_benchmark_core_v1 PRIVATE
  "MATCORE_BENCH_BUILD_TYPE=\"$<CONFIG>\""
  "MATCORE_BENCH_COMPILER_FLAGS=\"${MATCORE_BENCH_RECORDED_FLAGS}\""
  "MATCORE_BENCH_SOURCE_COMMIT=\"${MATCORE_BENCH_SOURCE_COMMIT}\""
)
target_link_libraries(matcore_benchmark_core_v1
  PUBLIC MatcoreDSL::CpuPlannerV1 MatcoreDSL::CpuPlannerV3
  PRIVATE MatcoreDSL::CpuBackendsV1
)

add_executable(matcore-bench main.cpp)
target_compile_features(matcore-bench PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(matcore-bench PRIVATE /W4 /WX)
else()
  target_compile_options(matcore-bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
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
  if(MSVC)
    target_compile_options(matcore_benchmark_core_test PRIVATE /W4 /WX)
  else()
    target_compile_options(matcore_benchmark_core_test PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
  endif()
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
          "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/benchmark/matcore-bench-v2.schema.json"
    )
  endif()
endif()
