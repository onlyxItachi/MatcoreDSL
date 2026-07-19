cmake_minimum_required(VERSION 3.24)

project(MatcoreDSLCompiler VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
  message(FATAL_ERROR "The standalone MDSLC bootstrap currently supports the Ninja generator only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

