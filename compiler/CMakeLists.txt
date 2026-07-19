cmake_minimum_required(VERSION 3.24)

project(MatcoreDSLCompiler VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
  message(FATAL_ERROR "The standalone MDSLC compiler currently supports the Ninja generator only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
  MDSLC_ENABLE_NATIVE_FRONTEND
  "Build the supported Clang LibTooling frontend"
  ON
)
option(
  MDSLC_ENABLE_BOOTSTRAP_FRONTEND
  "Build the compatibility-only Clang AST-JSON frontend"
  ON
)
option(
  MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE
  "Enable the driver tool-prefix override for deliberate tests"
  OFF
)
mark_as_advanced(MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE)

if(MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE is allowed only with BUILD_TESTING=ON")
endif()

if(NOT MDSLC_ENABLE_NATIVE_FRONTEND AND NOT MDSLC_ENABLE_BOOTSTRAP_FRONTEND)
  message(FATAL_ERROR "At least one MDSLC frontend must be enabled")
endif()

if(MDSLC_ENABLE_NATIVE_FRONTEND)
  find_package(LLVM 21.1.8 EXACT CONFIG REQUIRED)
  find_package(Clang 21.1.8 EXACT CONFIG REQUIRED)
  if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL "21.1.8")
    message(FATAL_ERROR
      "MDSLC native frontend requires the audited LLVM/Clang 21.1.8 tuple; "
      "found LLVM ${LLVM_PACKAGE_VERSION}")
  endif()
  if(NOT TARGET clang-cpp OR NOT TARGET LLVM)
    message(FATAL_ERROR
      "Clang 21 CMake packages do not provide the required clang-cpp/LLVM targets")
  endif()
  foreach(MDSLC_REQUIRED_CLANG_HEADER IN ITEMS
      clang/Tooling/Tooling.h
      clang/Tooling/CommonOptionsParser.h
      clang/ASTMatchers/ASTMatchFinder.h
      clang/Rewrite/Core/Rewriter.h
      clang/Frontend/FrontendActions.h
      clang/Frontend/CompilerInstance.h
      clang/Lex/Lexer.h
      clang/AST/Attr.h)
    if(NOT EXISTS "${CLANG_INCLUDE_DIRS}/${MDSLC_REQUIRED_CLANG_HEADER}")
      message(FATAL_ERROR
        "Matching Clang 21 development header is required: ${MDSLC_REQUIRED_CLANG_HEADER}")
    endif()
  endforeach()
endif()

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
if(MDSLC_ENABLE_NATIVE_FRONTEND)
  execute_process(
    COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" --version
    RESULT_VARIABLE MDSLC_CLANGXX_VERSION_STATUS
    OUTPUT_VARIABLE MDSLC_CLANGXX_VERSION_OUTPUT
    ERROR_VARIABLE MDSLC_CLANGXX_VERSION_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT MDSLC_CLANGXX_VERSION_STATUS EQUAL 0 OR
     NOT MDSLC_CLANGXX_VERSION_OUTPUT MATCHES "clang version 21\\.1\\.8")
    message(FATAL_ERROR
      "Native frontend requires clang++ 21.1.8 coherent with the linked "
      "LLVM/Clang packages; ${MDSLC_CLANGXX_EXECUTABLE} reported: "
      "${MDSLC_CLANGXX_VERSION_OUTPUT}${MDSLC_CLANGXX_VERSION_ERROR}")
  endif()
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
    if(MDSLC_ENABLE_NATIVE_FRONTEND AND MDSLC_ENABLE_BOOTSTRAP_FRONTEND)
      add_test(
        NAME frontend.native.focused
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/run_native_frontend_tests.py"
          --extractor "$<TARGET_FILE:matcore-extract>"
      )
      set_tests_properties(
        frontend.native.focused
        PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
      )

      add_test(
        NAME frontend.native.primary
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/frontend/run_frontend_tests.py"
          --extractor "$<TARGET_FILE:matcore-extract>"
      )
      set_tests_properties(
        frontend.native.primary
        PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
      )

      add_test(
        NAME frontend.native.adversarial
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_validation/run_native_validation.py"
          --suite core
          --extractor "$<TARGET_FILE:matcore-extract>"
          --clang "${MDSLC_CLANGXX_EXECUTABLE}"
      )
      set_tests_properties(
        frontend.native.adversarial
        PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
      )

      add_test(
        NAME frontend.native.driver_pipeline
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_validation/run_native_validation.py"
          --suite driver
          --extractor "$<TARGET_FILE:matcore-extract>"
          --driver "$<TARGET_FILE:mdslc_driver>"
          --clang "${MDSLC_CLANGXX_EXECUTABLE}"
      )
      set_tests_properties(
        frontend.native.driver_pipeline
        PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
      )
    elseif(MDSLC_ENABLE_BOOTSTRAP_FRONTEND)
      add_test(
        NAME frontend.native.unavailable
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_validation/run_native_validation.py"
          --suite unavailable
          --extractor "$<TARGET_FILE:matcore-extract>"
          --driver "$<TARGET_FILE:mdslc_driver>"
          --clang "${MDSLC_CLANGXX_EXECUTABLE}"
      )
      set_tests_properties(
        frontend.native.unavailable
        PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
      )
    endif()

    add_test(
      NAME driver.native.selection
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/driver_native/run_driver_tests.py"
        --driver "$<TARGET_FILE:mdslc_driver>"
    )
    set_tests_properties(
      driver.native.selection
      PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )

    if(MDSLC_ENABLE_NATIVE_FRONTEND)
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

      add_test(
        NAME consumer.installed
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer/run_installed_consumer_test.py"
          --cmake "${CMAKE_COMMAND}"
          --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer"
          --producer-build-dir "${CMAKE_BINARY_DIR}"
          --test-root "${CMAKE_BINARY_DIR}/tests/installed-consumer"
          --cxx-compiler "${CMAKE_CXX_COMPILER}"
      )
    endif()
  endif()
endif()
