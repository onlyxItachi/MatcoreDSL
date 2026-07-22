cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0091 NEW)

project(MatcoreDSLCompiler VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatcoreDSLCompilerOptions.cmake")

if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
  message(FATAL_ERROR "The standalone MDSLC compiler currently supports the Ninja generator only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(MATCORE_RAPIDJSON_INCLUDE_DIR rapidjson/document.h REQUIRED)
include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_INCLUDES "${MATCORE_RAPIDJSON_INCLUDE_DIR}")
check_cxx_source_compiles(
  "#include <rapidjson/document.h>\nint main() { rapidjson::Document document; return document.IsNull() ? 0 : 0; }"
  MDSLC_RAPIDJSON_HEADER_COMPILES
)
if(NOT MDSLC_RAPIDJSON_HEADER_COMPILES)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "The installed RapidJSON header is incompatible with ${CMAKE_CXX_COMPILER_ID}; "
      "install a patched RapidJSON 1.1 package")
  endif()
  if(MDSLC_USING_CLANG_CL)
    set(CMAKE_REQUIRED_FLAGS
      "/clang:-fdelayed-template-parsing /clang:-Wno-delayed-template-parsing-in-cxx20")
    set(MDSLC_RAPIDJSON_COMPAT_COMPILE_OPTIONS
      /clang:-fdelayed-template-parsing
      /clang:-Wno-delayed-template-parsing-in-cxx20)
  else()
    set(CMAKE_REQUIRED_FLAGS
      "-fdelayed-template-parsing -Wno-delayed-template-parsing-in-cxx20")
    set(MDSLC_RAPIDJSON_COMPAT_COMPILE_OPTIONS
      -fdelayed-template-parsing
      -Wno-delayed-template-parsing-in-cxx20)
  endif()
  check_cxx_source_compiles(
    "#include <rapidjson/document.h>\nint main() { rapidjson::Document document; return document.IsNull() ? 0 : 0; }"
    MDSLC_RAPIDJSON_HEADER_COMPILES_WITH_DELAYED_PARSING
  )
  if(NOT MDSLC_RAPIDJSON_HEADER_COMPILES_WITH_DELAYED_PARSING)
    message(FATAL_ERROR
      "The installed RapidJSON header is incompatible with Clang and the "
      "narrow delayed-template-parsing compatibility mode did not compile")
  endif()
  message(STATUS
    "Using source-local compatibility flags for the installed RapidJSON header")
endif()
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_INCLUDES)

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
  if(TARGET clang-cpp)
    if(WIN32)
      # The official LLVM 21.1.8 Windows package exports build-host-only DIA
      # SDK paths through LLVM's interface.  clang-cpp and LLVM are shared
      # imported targets, so consumers need their authenticated import
      # libraries, not those stale private build dependencies.
      set(MDSLC_CLANG_TOOLING_LIBRARIES
        "$<TARGET_LINKER_FILE:clang-cpp>")
      if(TARGET LLVM)
        list(APPEND MDSLC_CLANG_TOOLING_LIBRARIES
          "$<TARGET_LINKER_FILE:LLVM>")
      endif()
      set(MDSLC_CLANG_LINK_MODEL
        "clang-cpp/LLVM shared import libraries")
    else()
      set(MDSLC_CLANG_TOOLING_LIBRARIES clang-cpp)
      if(TARGET LLVM)
        list(APPEND MDSLC_CLANG_TOOLING_LIBRARIES LLVM)
      endif()
      set(MDSLC_CLANG_LINK_MODEL "clang-cpp shared library")
    endif()
  else()
    set(MDSLC_CLANG_TOOLING_LIBRARIES)
    foreach(MDSLC_REQUIRED_CLANG_TARGET IN ITEMS
        clangTooling
        clangASTMatchers
        clangRewrite
        clangFrontend
        clangSerialization
        clangDriver
        clangParse
        clangSema
        clangAnalysis
        clangAST
        clangLex
        clangBasic)
      if(NOT TARGET ${MDSLC_REQUIRED_CLANG_TARGET})
        message(FATAL_ERROR
          "Clang 21 CMake package provides neither clang-cpp nor required "
          "component target ${MDSLC_REQUIRED_CLANG_TARGET}")
      endif()
      list(APPEND MDSLC_CLANG_TOOLING_LIBRARIES
        ${MDSLC_REQUIRED_CLANG_TARGET})
    endforeach()
    set(MDSLC_CLANG_LINK_MODEL "supported Clang component targets")
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
  message(STATUS
    "MDSLC native frontend link model: ${MDSLC_CLANG_LINK_MODEL}")
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

if(WIN32)
  set(MDSLC_DEFAULT_CLANG_DRIVER "${CMAKE_CXX_COMPILER}")
else()
  set(MDSLC_DEFAULT_CLANG_DRIVER "/usr/bin/clang++-21")
endif()
set(MDSLC_CLANGXX_EXECUTABLE "${MDSLC_DEFAULT_CLANG_DRIVER}" CACHE FILEPATH
  "Absolute path to the Clang driver invoked for .mdsl host compilation")

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
      "Native frontend requires a Clang 21.1.8 driver coherent with the linked "
      "LLVM/Clang packages; ${MDSLC_CLANGXX_EXECUTABLE} reported: "
      "${MDSLC_CLANGXX_VERSION_OUTPUT}${MDSLC_CLANGXX_VERSION_ERROR}")
  endif()
endif()

# MDSLC_CLANGXX_EXECUTABLE is the configure-time authenticated compiler.  It
# must remain absolute so that configuration proves the linked Clang libraries
# and host driver belong to one coherent tuple.  Installed Windows binaries,
# however, must not remember the runner-local extraction directory: the
# platform process backend resolves this deliberately bare runtime tool name
# from the deployed environment.
if(WIN32)
  set(MDSLC_RUNTIME_CLANG_DRIVER "clang-cl.exe")
else()
  set(MDSLC_RUNTIME_CLANG_DRIVER "${MDSLC_CLANGXX_EXECUTABLE}")
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

add_subdirectory(lib/support)
add_subdirectory(lib/ir)
add_subdirectory(lib/frontend)
add_subdirectory(lib/codegen)
add_subdirectory(lib/platform)
add_subdirectory(tools/matcore-extract)
add_subdirectory(lib/runtime)
add_subdirectory(tools/matcore-plan)
add_subdirectory(tools/matcore-bench)

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
mdslc_target_enable_warnings(mdslc_driver)
target_link_libraries(mdslc_driver PRIVATE MatcoreDSL::PlatformSupportV1)
add_dependencies(mdslc_driver matcore-extract matcore_runtime)

set_property(TARGET matcore-extract PROPERTY EXPORT_NAME Extractor)

install(
  TARGETS mdslc_driver matcore-extract matcore_runtime matcore-plan
  EXPORT MatcoreDSLTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
if(WIN32 AND MDSLC_ENABLE_NATIVE_FRONTEND)
  # Official LLVM distributions may expose clang-cpp/LLVM as imported shared
  # targets or as static component targets.  Copy only declared imported DLL
  # runtime artifacts; a fully static component link has nothing to bundle.
  foreach(MDSLC_IMPORTED_TOOLING_TARGET IN ITEMS clang-cpp LLVM)
    if(TARGET ${MDSLC_IMPORTED_TOOLING_TARGET})
      get_target_property(MDSLC_IMPORTED_TOOLING_TYPE
        ${MDSLC_IMPORTED_TOOLING_TARGET} TYPE)
      get_target_property(MDSLC_IMPORTED_TOOLING_IMPORTED
        ${MDSLC_IMPORTED_TOOLING_TARGET} IMPORTED)
      if(MDSLC_IMPORTED_TOOLING_IMPORTED AND
         MDSLC_IMPORTED_TOOLING_TYPE STREQUAL "SHARED_LIBRARY")
        install(IMPORTED_RUNTIME_ARTIFACTS ${MDSLC_IMPORTED_TOOLING_TARGET}
          RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
      endif()
    endif()
  endforeach()
endif()
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
    elseif(WIN32 AND MDSLC_ENABLE_NATIVE_FRONTEND)
      # The supported Windows distribution deliberately excludes the
      # compatibility-only AST-JSON frontend. Keep native authentication in
      # CTest through a focused replacement instead of silently dropping all
      # frontend coverage when bootstrap parity is unavailable.
      add_test(
        NAME frontend.native.windows
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/windows/run_windows_native_pipeline.py"
          --suite frontend
          --extractor "$<TARGET_FILE:matcore-extract>"
          --driver "$<TARGET_FILE:mdslc_driver>"
          --clang "${MDSLC_CLANGXX_EXECUTABLE}"
          --build-dir "${CMAKE_BINARY_DIR}"
      )
      set_tests_properties(
        frontend.native.windows
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

    if(MDSLC_ENABLE_NATIVE_FRONTEND)
      add_test(
        NAME windows.driver_coff.contract
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/windows/run_windows_driver_contract.py"
          --extractor "$<TARGET_FILE:matcore-extract>"
          --driver "$<TARGET_FILE:mdslc_driver>"
      )
      set_tests_properties(
        windows.driver_coff.contract
        PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
      )
    endif()

    if(NOT WIN32)
      # This adversarial harness intentionally models ELF/rpath relocation and
      # POSIX executable test shims. The Windows-native pipeline test below
      # covers argv forwarding, COFF production and PE execution instead.
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
    endif()

    if(MDSLC_ENABLE_NATIVE_FRONTEND)
      if(WIN32)
        # The Linux matrix authenticates ELF partial links, SONAME/rpath and
        # POSIX process behavior. Windows has an explicit COFF/PE replacement
        # and never emulates clang++ -r.
        add_test(
          NAME integration.windows.native_pipeline
          COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/windows/run_windows_native_pipeline.py"
            --suite pipeline
            --extractor "$<TARGET_FILE:matcore-extract>"
            --driver "$<TARGET_FILE:mdslc_driver>"
            --clang "${MDSLC_CLANGXX_EXECUTABLE}"
            --build-dir "${CMAKE_BINARY_DIR}"
        )
        set_tests_properties(
          integration.windows.native_pipeline
          PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        )
      else()
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

    if(WIN32)
      find_program(MDSLC_PACKAGE_READOBJ_EXECUTABLE
        NAMES llvm-readobj llvm-readobj.exe
        HINTS "${LLVM_TOOLS_BINARY_DIR}"
        REQUIRED)
      set(MDSLC_PACKAGE_NM_EXECUTABLE "")
    else()
      find_program(MDSLC_PACKAGE_NM_EXECUTABLE NAMES nm llvm-nm-21 REQUIRED)
      set(MDSLC_PACKAGE_READOBJ_EXECUTABLE "")
    endif()
    add_test(
      NAME package.installed_c17_abi
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/run_installed_c17_abi_probe.py"
        --cmake "${CMAKE_COMMAND}"
        --producer-build-dir "${CMAKE_BINARY_DIR}"
        --test-root "${CMAKE_BINARY_DIR}/tests/installed-c17-abi"
        --c-compiler "${CMAKE_C_COMPILER}"
        --nm "${MDSLC_PACKAGE_NM_EXECUTABLE}"
        --readobj "${MDSLC_PACKAGE_READOBJ_EXECUTABLE}"
    )
  endif()
endif()
