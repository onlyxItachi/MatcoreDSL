cmake_minimum_required(VERSION 3.24)

project(MatcoreDSLCompiler VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
  message(FATAL_ERROR "The standalone MDSLC bootstrap currently supports the Ninja generator only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(MDSLC_USE_CCACHE "Use ccache when it is already installed" ON)
if(MDSLC_USE_CCACHE)
  find_program(MDSLC_CCACHE_EXECUTABLE NAMES ccache)
  if(MDSLC_CCACHE_EXECUTABLE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${MDSLC_CCACHE_EXECUTABLE}")
  endif()
endif()

set(
  MDSLC_CLANGXX_EXECUTABLE
  "/usr/bin/clang++-21"
  CACHE FILEPATH
  "Absolute path to the clang++ driver invoked for .mdsl host compilation"
)

if(NOT IS_ABSOLUTE "${MDSLC_CLANGXX_EXECUTABLE}")
  message(FATAL_ERROR "MDSLC_CLANGXX_EXECUTABLE must be an absolute path")
endif()
if(NOT EXISTS "${MDSLC_CLANGXX_EXECUTABLE}")
  message(FATAL_ERROR "Configured clang++ driver does not exist: ${MDSLC_CLANGXX_EXECUTABLE}")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mdslc_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/mdslc_config.h"
  @ONLY
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/matcore")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/matcore/mdsl.h"
  "${CMAKE_BINARY_DIR}/include/matcore/mdsl.h"
  COPYONLY
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/matcore/runtime_c.h"
  "${CMAKE_BINARY_DIR}/include/matcore/runtime_c.h"
  COPYONLY
)

add_subdirectory(lib/ir)
add_subdirectory(lib/frontend)
add_subdirectory(lib/codegen)
add_subdirectory(tools/matcore-extract)
add_subdirectory(lib/runtime)

add_executable(mdslc_driver tools/mdslc/main.cpp)
set_target_properties(
  mdslc_driver
  PROPERTIES
    OUTPUT_NAME "mdslc++"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_include_directories(
  mdslc_driver
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_compile_options(
  mdslc_driver
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
add_dependencies(mdslc_driver matcore-extract matcore_runtime)

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME frontend.bootstrap
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/frontend/run_frontend_tests.py"
        --extractor "$<TARGET_FILE:matcore-extract>"
    )
    set_tests_properties(
      frontend.bootstrap
      PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )

    add_test(
      NAME integration.validation_matrix
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/run_validation_matrix.py"
        --build-dir "${CMAKE_BINARY_DIR}"
    )
    set_tests_properties(
      integration.validation_matrix
      PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
  endif()
endif()
