cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MatcoreRuntimeV0 LANGUAGES CXX)
  include(CTest)
endif()

add_library(matcore_runtime SHARED cpu_runtime.cpp)
add_library(MatcoreDSL::Runtime ALIAS matcore_runtime)

target_compile_features(matcore_runtime PUBLIC cxx_std_20)
target_compile_definitions(matcore_runtime PRIVATE MATCORE_RUNTIME_BUILD)
target_include_directories(matcore_runtime
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>"
)
set_target_properties(matcore_runtime PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  OUTPUT_NAME matcore_runtime
  SOVERSION 0
  VERSION 0.0.0
)

if(BUILD_TESTING)
  add_executable(matcore_runtime_cpu_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/runtime_cpu_test.cpp"
  )
  target_compile_features(matcore_runtime_cpu_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_runtime_cpu_test PRIVATE MatcoreDSL::Runtime)
  add_test(NAME runtime.cpu.gemm_v0 COMMAND matcore_runtime_cpu_test)
endif()
