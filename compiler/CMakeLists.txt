cmake_minimum_required(VERSION 3.24)

project(MatcoreDSLCompiler VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

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
set_property(TARGET mdslc_driver PROPERTY EXPORT_NAME Compiler)
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

set_property(TARGET matcore-extract PROPERTY EXPORT_NAME Extractor)

install(
  TARGETS mdslc_driver matcore-extract matcore_runtime
  EXPORT MatcoreDSLTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/matcore"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.h"
)

set(MATCOREDSL_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/MatcoreDSL")
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatcoreDSLConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/MatcoreDSLConfig.cmake"
  INSTALL_DESTINATION "${MATCOREDSL_CMAKE_INSTALL_DIR}"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MatcoreDSLConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)
install(
  EXPORT MatcoreDSLTargets
  FILE MatcoreDSLTargets.cmake
  NAMESPACE MatcoreDSL::
  DESTINATION "${MATCOREDSL_CMAKE_INSTALL_DIR}"
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MatcoreDSLConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MatcoreDSLConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatcoreDSLCompile.cmake"
  DESTINATION "${MATCOREDSL_CMAKE_INSTALL_DIR}"
)

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
