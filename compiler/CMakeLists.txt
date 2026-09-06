cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0091 NEW)

project(MatcoreDSLCompiler VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatcoreDSLCompilerOptions.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatcoreDSLToolchainVersion.cmake")

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
  MDSLC_ENABLE_MATCORE_MLIR
  "Build the internal Matcore MLIR semantic dialect and inspection tool"
  OFF
)
option(
  MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS
  "Build the opt-in inspection-only MLIR Transform/Vector readiness seam"
  OFF
)
mark_as_advanced(MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS)
set(
  MDSLC_DEFAULT_SEMANTIC_PIPELINE
  "capture-v0"
  CACHE STRING
  "Default mdslc++ CPU semantic pipeline (capture-v0 or matcore-mlir)"
)
set_property(
  CACHE MDSLC_DEFAULT_SEMANTIC_PIPELINE
  PROPERTY STRINGS capture-v0 matcore-mlir
)
set(MDSLC_VALID_SEMANTIC_PIPELINES capture-v0 matcore-mlir)
if(NOT MDSLC_DEFAULT_SEMANTIC_PIPELINE IN_LIST MDSLC_VALID_SEMANTIC_PIPELINES)
  message(FATAL_ERROR
    "MDSLC_DEFAULT_SEMANTIC_PIPELINE must be capture-v0 or matcore-mlir; "
    "found '${MDSLC_DEFAULT_SEMANTIC_PIPELINE}'")
endif()
if(MDSLC_DEFAULT_SEMANTIC_PIPELINE STREQUAL "matcore-mlir" AND
   NOT MDSLC_ENABLE_MATCORE_MLIR)
  message(FATAL_ERROR
    "MDSLC_DEFAULT_SEMANTIC_PIPELINE=matcore-mlir requires "
    "MDSLC_ENABLE_MATCORE_MLIR=ON")
endif()
if(MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS AND
   NOT MDSLC_ENABLE_MATCORE_MLIR)
  message(FATAL_ERROR
    "MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS=ON requires "
    "MDSLC_ENABLE_MATCORE_MLIR=ON")
endif()
set(MDSLC_MATCORE_MLIR_AVAILABLE "OFF")
if(MDSLC_ENABLE_MATCORE_MLIR)
  set(MDSLC_MATCORE_MLIR_AVAILABLE "ON")
endif()
message(STATUS
  "MDSLC default semantic pipeline: ${MDSLC_DEFAULT_SEMANTIC_PIPELINE}")
option(
  MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE
  "Enable the driver tool-prefix override for deliberate tests"
  OFF
)
mark_as_advanced(MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE)

# Product and release builds remain pinned to the audited 21.1.8 tuple.  This
# internal-only escape hatch exists so a compatibility branch can measure one
# explicitly named coherent upstream tuple without weakening the default.
set(MDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION "" CACHE STRING
  "Exact non-product LLVM/Clang/MLIR version used for a portability experiment")
mark_as_advanced(MDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION)
set(MDSLC_REQUIRED_TOOLCHAIN_VERSION "21.1.8")
set(MDSLC_EXPERIMENTAL_TOOLCHAIN_ACTIVE 0)
if(NOT MDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION STREQUAL "")
  if(NOT MDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION STREQUAL "22.1.8")
    message(FATAL_ERROR
      "MDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION admits only the reviewed "
      "22.1.8 portability experiment")
  endif()
  set(MDSLC_REQUIRED_TOOLCHAIN_VERSION
    "${MDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION}")
  set(MDSLC_EXPERIMENTAL_TOOLCHAIN_ACTIVE 1)
  message(WARNING
    "MDSLC experimental toolchain ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} is "
    "compatibility evidence only; canonical product support remains 21.1.8")
endif()
if(MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS AND
   MDSLC_EXPERIMENTAL_TOOLCHAIN_ACTIVE)
  message(FATAL_ERROR
    "MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS is certified only for "
    "the exact LLVM/Clang/MLIR 21.1.8 product tuple; the experimental "
    "toolchain compatibility path is not admitted")
endif()

if(MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE is allowed only with BUILD_TESTING=ON")
endif()

if(NOT MDSLC_ENABLE_NATIVE_FRONTEND AND NOT MDSLC_ENABLE_BOOTSTRAP_FRONTEND)
  message(FATAL_ERROR "At least one MDSLC frontend must be enabled")
endif()

function(mdslc_target_match_llvm_rtti target)
  if(DEFINED LLVM_ENABLE_RTTI AND NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(${target} PRIVATE /GR-)
    else()
      target_compile_options(${target} PRIVATE -fno-rtti)
    endif()
  endif()
endfunction()

if(MDSLC_ENABLE_MATCORE_MLIR)
  if(NOT DEFINED MLIR_DIR OR MLIR_DIR STREQUAL "")
    message(FATAL_ERROR
      "MDSLC_ENABLE_MATCORE_MLIR requires an explicit MLIR_DIR for the "
      "selected coherent LLVM/Clang/MLIR tuple")
  endif()
  find_package(LLVM ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} EXACT CONFIG REQUIRED)
  find_package(Clang ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} EXACT CONFIG REQUIRED)
  find_package(MLIR ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} EXACT CONFIG REQUIRED)
  if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL MDSLC_REQUIRED_TOOLCHAIN_VERSION OR
     NOT MLIR_VERSION VERSION_EQUAL MDSLC_REQUIRED_TOOLCHAIN_VERSION)
    message(FATAL_ERROR
      "Matcore MLIR requires exact coherent LLVM/Clang/MLIR "
      "${MDSLC_REQUIRED_TOOLCHAIN_VERSION} packages; found "
      "LLVM ${LLVM_PACKAGE_VERSION} and MLIR ${MLIR_VERSION}")
  endif()
  set(MDSLC_REQUIRED_MLIR_TARGETS
      MLIRIR
      MLIRArithDialect
      MLIRBufferizationDialect
      MLIRBufferizationTransforms
      MLIRFuncDialect
      MLIRLinalgDialect
      MLIRLinalgTransforms
      MLIRMemRefDialect
      MLIRParser
      MLIRTensorTransforms
      MLIRTransforms
      MLIRDestinationStyleOpInterface
      MLIRSideEffectInterfaces)
  if(MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS)
    list(APPEND MDSLC_REQUIRED_MLIR_TARGETS
      MLIRLinalgTransformOps
      MLIRTensorDialect
      MLIRTransformDialect
      MLIRTransformDialectTransforms
      MLIRUBDialect
      MLIRVectorDialect)
  endif()
  foreach(MDSLC_REQUIRED_MLIR_TARGET IN LISTS MDSLC_REQUIRED_MLIR_TARGETS)
    if(NOT TARGET ${MDSLC_REQUIRED_MLIR_TARGET})
      message(FATAL_ERROR
        "MLIR ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} package is missing required component target "
        "${MDSLC_REQUIRED_MLIR_TARGET}")
    endif()
  endforeach()
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include(AddLLVM)
  include(AddMLIR)
  message(STATUS
    "MDSLC Matcore MLIR semantic layer: exact MLIR ${MLIR_VERSION} at ${MLIR_DIR}")
endif()

if(MDSLC_ENABLE_NATIVE_FRONTEND)
  find_package(LLVM ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} EXACT CONFIG REQUIRED)
  find_package(Clang ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} EXACT CONFIG REQUIRED)
  if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL MDSLC_REQUIRED_TOOLCHAIN_VERSION)
    message(FATAL_ERROR
      "MDSLC native frontend requires the selected coherent LLVM/Clang "
      "${MDSLC_REQUIRED_TOOLCHAIN_VERSION} tuple; "
      "found LLVM ${LLVM_PACKAGE_VERSION}")
  endif()
  if(WIN32 AND TARGET LLVMDebugInfoPDB)
    set(MDSLC_DIAGUIDS_LIBRARY "" CACHE FILEPATH
      "Coherent Visual Studio DIA SDK GUID library")
    get_target_property(MDSLC_PDB_INTERFACE_LIBRARIES
      LLVMDebugInfoPDB INTERFACE_LINK_LIBRARIES)
    set(MDSLC_PDB_REBASED_INTERFACE_LIBRARIES)
    set(MDSLC_REBASED_DIAGUIDS FALSE)
    foreach(MDSLC_PDB_INTERFACE_LIBRARY IN LISTS
        MDSLC_PDB_INTERFACE_LIBRARIES)
      get_filename_component(MDSLC_PDB_INTERFACE_NAME
        "${MDSLC_PDB_INTERFACE_LIBRARY}" NAME)
      if(IS_ABSOLUTE "${MDSLC_PDB_INTERFACE_LIBRARY}" AND
         MDSLC_PDB_INTERFACE_NAME STREQUAL "diaguids.lib")
        if(NOT "${MDSLC_DIAGUIDS_LIBRARY}" STREQUAL "")
          if(NOT IS_ABSOLUTE "${MDSLC_DIAGUIDS_LIBRARY}" OR
             NOT EXISTS "${MDSLC_DIAGUIDS_LIBRARY}" OR
             IS_DIRECTORY "${MDSLC_DIAGUIDS_LIBRARY}")
            message(FATAL_ERROR
              "MDSLC_DIAGUIDS_LIBRARY must be an existing absolute file")
          endif()
          get_filename_component(MDSLC_DIAGUIDS_NAME
            "${MDSLC_DIAGUIDS_LIBRARY}" NAME)
          if(NOT MDSLC_DIAGUIDS_NAME STREQUAL "diaguids.lib")
            message(FATAL_ERROR
              "MDSLC_DIAGUIDS_LIBRARY must name diaguids.lib")
          endif()
          list(APPEND MDSLC_PDB_REBASED_INTERFACE_LIBRARIES
            "${MDSLC_DIAGUIDS_LIBRARY}")
          set(MDSLC_REBASED_DIAGUIDS TRUE)
        elseif(NOT EXISTS "${MDSLC_PDB_INTERFACE_LIBRARY}" OR
               IS_DIRECTORY "${MDSLC_PDB_INTERFACE_LIBRARY}")
          message(FATAL_ERROR
            "LLVM ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} exports a missing "
            "build-host DIA SDK dependency "
            "(${MDSLC_PDB_INTERFACE_LIBRARY}). Set "
            "MDSLC_DIAGUIDS_LIBRARY to the coherent Visual Studio DIA SDK "
            "amd64 diaguids.lib used by this build.")
        else()
          list(APPEND MDSLC_PDB_REBASED_INTERFACE_LIBRARIES
            "${MDSLC_PDB_INTERFACE_LIBRARY}")
        endif()
      else()
        list(APPEND MDSLC_PDB_REBASED_INTERFACE_LIBRARIES
          "${MDSLC_PDB_INTERFACE_LIBRARY}")
      endif()
    endforeach()
    if(MDSLC_REBASED_DIAGUIDS)
      set_property(TARGET LLVMDebugInfoPDB PROPERTY INTERFACE_LINK_LIBRARIES
        "${MDSLC_PDB_REBASED_INTERFACE_LIBRARIES}")
      message(STATUS
        "Rebased LLVM DIA SDK dependency to ${MDSLC_DIAGUIDS_LIBRARY}")
    endif()
  endif()
  if(TARGET clang-cpp)
    if(WIN32)
      # Some Windows LLVM packages provide shared clang-cpp/LLVM imported
      # targets. Consumers need their authenticated import libraries, not
      # private build-host dependencies carried by a faulty interface graph.
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
          "Clang ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} CMake package provides "
          "neither clang-cpp nor required "
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
        "Matching Clang ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} development "
        "header is required: ${MDSLC_REQUIRED_CLANG_HEADER}")
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
  mdslc_extract_clang_version_token(
    "${MDSLC_CLANGXX_VERSION_OUTPUT}\n${MDSLC_CLANGXX_VERSION_ERROR}"
    MDSLC_CLANGXX_REPORTED_VERSION)
  if(NOT MDSLC_CLANGXX_VERSION_STATUS EQUAL 0 OR
     NOT "${MDSLC_CLANGXX_REPORTED_VERSION}" STREQUAL
       "${MDSLC_REQUIRED_TOOLCHAIN_VERSION}")
    message(FATAL_ERROR
      "Native frontend requires a Clang ${MDSLC_REQUIRED_TOOLCHAIN_VERSION} "
      "driver coherent with the linked "
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
if(MDSLC_ENABLE_MATCORE_MLIR)
  add_subdirectory(lib/mlir)
endif()
add_subdirectory(lib/frontend)
add_subdirectory(lib/codegen)
add_subdirectory(lib/platform)
add_subdirectory(tools/matcore-extract)
add_subdirectory(lib/runtime)
add_subdirectory(tools/matcore-plan)
add_subdirectory(tools/matcore-bench)
if(BUILD_TESTING AND MDSLC_ENABLE_NATIVE_FRONTEND AND
   MDSLC_ENABLE_MATCORE_MLIR AND NOT WIN32)
  # Private feasibility instrument: no installed headers, driver option, or
  # execution consumer. The supported runtime/frontend targets stay unchanged.
  add_subdirectory(tests/closed_region)
endif()
if(MDSLC_ENABLE_MATCORE_MLIR)
  add_subdirectory(tools/matcore-mlir)
endif()

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

# The exact-22 lane deliberately keeps LLVM/Clang external.  Its installed
# extractor records the authenticated imported clang-cpp directory as an
# experimental-only loader search path and checks the loaded DSO at startup.
# No LLVM artifact is copied into the MatcoreDSL package.
if(NOT WIN32 AND MDSLC_EXPERIMENTAL_TOOLCHAIN_ACTIVE)
  if(NOT TARGET clang-cpp)
    message(FATAL_ERROR
      "The non-Windows experimental toolchain requires the imported "
      "clang-cpp shared runtime target")
  endif()
  get_target_property(MDSLC_EXPERIMENTAL_CLANG_CPP_TYPE clang-cpp TYPE)
  if(NOT MDSLC_EXPERIMENTAL_CLANG_CPP_TYPE STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR
      "The non-Windows experimental toolchain requires a shared clang-cpp "
      "runtime; found ${MDSLC_EXPERIMENTAL_CLANG_CPP_TYPE}")
  endif()
  set_property(TARGET matcore-extract PROPERTY INSTALL_RPATH
    "$<TARGET_FILE_DIR:clang-cpp>")
  target_compile_definitions(matcore-extract PRIVATE
    "MDSLC_EXPECTED_CLANG_CPP_RUNTIME=\"$<TARGET_FILE:clang-cpp>\"")
endif()

install(
  TARGETS mdslc_driver matcore-extract matcore_runtime matcore-plan
  EXPORT MatcoreDSLTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
if(MDSLC_ENABLE_MATCORE_MLIR)
  # The semantic inspection tool is installed as a leaf executable. Its
  # private MLIR implementation libraries are deliberately not exported, so
  # installed MatcoreDSL consumers acquire no MLIR dependency.
  install(TARGETS matcore-mlir
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
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
if(WIN32)
  install(
    FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatcoreDSLWindowsRuntimePrerequisites.json"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/MatcoreDSL"
  )
endif()

if(BUILD_TESTING)
  add_test(
    NAME toolchain.version_token.contract
    COMMAND "${CMAKE_COMMAND}"
      -DMDSLC_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/toolchain_version_contract.cmake"
  )
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    if(MDSLC_ENABLE_NATIVE_FRONTEND)
      set(MDSLC_REGION_TEST_AVAILABLE 0)
      if(TARGET matcore_mlir_two_gemm_region)
        set(MDSLC_REGION_TEST_AVAILABLE 1)
      endif()
      add_test(NAME integration.two_gemm_region
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/run_two_gemm_region_tests.py"
          --extractor "$<TARGET_FILE:matcore-extract>"
          --driver "$<TARGET_FILE:mdslc_driver>"
          --clangxx "${MDSLC_CLANGXX_EXECUTABLE}"
          --repository-root "${CMAKE_CURRENT_SOURCE_DIR}/.."
          --region-available "${MDSLC_REGION_TEST_AVAILABLE}")
      set_tests_properties(integration.two_gemm_region PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.." TIMEOUT 300)
    endif()
    if(MDSLC_ENABLE_NATIVE_FRONTEND AND MDSLC_ENABLE_BOOTSTRAP_FRONTEND)
      add_test(
        NAME frontend.native.focused
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/run_native_frontend_tests.py"
          --extractor "$<TARGET_FILE:matcore-extract>"
          --expected-toolchain-version "${MDSLC_REQUIRED_TOOLCHAIN_VERSION}"
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
          --clang "${MDSLC_CLANGXX_EXECUTABLE}"
      )
      set_tests_properties(
        frontend.native.primary
        PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
      )

      if(NOT WIN32)
        add_test(
          NAME frontend.native.recovered_gemm_inspection
          COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/frontend/run_recovered_gemm_tests.py"
            --extractor "$<TARGET_FILE:matcore-extract>"
            --clang "${MDSLC_CLANGXX_EXECUTABLE}"
        )
        set_tests_properties(
          frontend.native.recovered_gemm_inspection
          PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        )
      endif()

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
      if(MDSLC_ENABLE_MATCORE_MLIR)
        if(NOT WIN32)
          add_test(
            NAME integration.semantic_mlir_cpu_pipeline
            COMMAND
              "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/run_semantic_cpu_pipeline_tests.py"
              --extractor "$<TARGET_FILE:matcore-extract>"
              --driver "$<TARGET_FILE:mdslc_driver>"
              --clangxx "${MDSLC_CLANGXX_EXECUTABLE}"
              --repository-root "${CMAKE_CURRENT_SOURCE_DIR}/.."
              --build-dir "${CMAKE_BINARY_DIR}"
              --expected-default-semantic-pipeline
                "${MDSLC_DEFAULT_SEMANTIC_PIPELINE}"
          )
          set_tests_properties(
            integration.semantic_mlir_cpu_pipeline
            PROPERTIES
              WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
              TIMEOUT 300
          )
        endif()
      else()
        # The unavailable-path contract is platform-neutral. In particular,
        # Windows distributions that do not yet contain the audited MLIR tuple
        # must reject an explicit semantic-pipeline request without artifacts.
        add_test(
          NAME integration.semantic_mlir_unavailable
          COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/run_semantic_cpu_pipeline_tests.py"
            --expect-unavailable
            --extractor "$<TARGET_FILE:matcore-extract>"
            --driver "$<TARGET_FILE:mdslc_driver>"
            --clangxx "${MDSLC_CLANGXX_EXECUTABLE}"
            --repository-root "${CMAKE_CURRENT_SOURCE_DIR}/.."
            --build-dir "${CMAKE_BINARY_DIR}"
            --expected-default-semantic-pipeline
              "${MDSLC_DEFAULT_SEMANTIC_PIPELINE}"
        )
        set_tests_properties(
          integration.semantic_mlir_unavailable
          PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
            TIMEOUT 180
        )
      endif()

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
          --matcore-mlir-available "${MDSLC_MATCORE_MLIR_AVAILABLE}"
          --default-semantic-pipeline "${MDSLC_DEFAULT_SEMANTIC_PIPELINE}"
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

    if(NOT WIN32 AND MDSLC_ENABLE_NATIVE_FRONTEND)
      find_program(MDSLC_PACKAGE_GIT_EXECUTABLE NAMES git REQUIRED)
      get_filename_component(MDSLC_PACKAGE_REPOSITORY_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
      execute_process(
        COMMAND
          "${MDSLC_PACKAGE_GIT_EXECUTABLE}" -C
          "${MDSLC_PACKAGE_REPOSITORY_ROOT}" rev-parse --verify HEAD
        RESULT_VARIABLE MDSLC_PACKAGE_SOURCE_COMMIT_RESULT
        OUTPUT_VARIABLE MDSLC_PACKAGE_SOURCE_COMMIT
        ERROR_VARIABLE MDSLC_PACKAGE_SOURCE_COMMIT_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      string(LENGTH "${MDSLC_PACKAGE_SOURCE_COMMIT}"
        MDSLC_PACKAGE_SOURCE_COMMIT_LENGTH)
      if(NOT MDSLC_PACKAGE_SOURCE_COMMIT_RESULT EQUAL 0 OR
         (NOT MDSLC_PACKAGE_SOURCE_COMMIT_LENGTH EQUAL 40 AND
          NOT MDSLC_PACKAGE_SOURCE_COMMIT_LENGTH EQUAL 64) OR
         NOT MDSLC_PACKAGE_SOURCE_COMMIT MATCHES "^[0-9a-fA-F]+$")
        message(FATAL_ERROR
          "Cannot authenticate the source-inaccessible package-test commit: "
          "${MDSLC_PACKAGE_SOURCE_COMMIT_ERROR}")
      endif()
      string(TOLOWER "${MDSLC_PACKAGE_SOURCE_COMMIT}"
        MDSLC_PACKAGE_SOURCE_COMMIT)
      execute_process(
        COMMAND
          "${MDSLC_PACKAGE_GIT_EXECUTABLE}" -C
          "${MDSLC_PACKAGE_REPOSITORY_ROOT}" status --porcelain=v1
          --untracked-files=all
        RESULT_VARIABLE MDSLC_PACKAGE_SOURCE_STATUS_RESULT
        OUTPUT_VARIABLE MDSLC_PACKAGE_SOURCE_STATUS
        ERROR_VARIABLE MDSLC_PACKAGE_SOURCE_STATUS_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(NOT MDSLC_PACKAGE_SOURCE_STATUS_RESULT EQUAL 0)
        message(FATAL_ERROR
          "Cannot authenticate source-inaccessible package-test cleanliness: "
          "${MDSLC_PACKAGE_SOURCE_STATUS_ERROR}")
      endif()
      if(MDSLC_PACKAGE_SOURCE_STATUS STREQUAL "")
        set(MDSLC_PACKAGE_SOURCE_WAS_CLEAN ON)
      else()
        set(MDSLC_PACKAGE_SOURCE_WAS_CLEAN OFF)
      endif()
      set(MDSLC_SOURCE_INACCESSIBLE_MLIR_ARGUMENTS)
      if(MDSLC_ENABLE_MATCORE_MLIR)
        list(APPEND MDSLC_SOURCE_INACCESSIBLE_MLIR_ARGUMENTS
          --mlir-dir "${MLIR_DIR}")
      endif()
      set(MDSLC_SOURCE_INACCESSIBLE_RUNTIME_ARGUMENTS)
      if(NOT WIN32 AND MDSLC_EXPERIMENTAL_TOOLCHAIN_ACTIVE)
        list(APPEND MDSLC_SOURCE_INACCESSIBLE_RUNTIME_ARGUMENTS
          --expected-clang-runtime "$<TARGET_FILE:clang-cpp>")
      endif()
      add_test(
        NAME package.installed_source_inaccessible
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/run_source_inaccessible_consumer.py"
          --cmake "${CMAKE_COMMAND}"
          --git "${MDSLC_PACKAGE_GIT_EXECUTABLE}"
          --repository-root "${MDSLC_PACKAGE_REPOSITORY_ROOT}"
          --expected-source-commit "${MDSLC_PACKAGE_SOURCE_COMMIT}"
          --configured-source-clean "${MDSLC_PACKAGE_SOURCE_WAS_CLEAN}"
          --expected-build-root "${CMAKE_BINARY_DIR}"
          --test-root
            "${CMAKE_BINARY_DIR}/tests/installed-source-inaccessible"
          --c-compiler "${CMAKE_C_COMPILER}"
          --cxx-compiler "${CMAKE_CXX_COMPILER}"
          --clangxx "${MDSLC_CLANGXX_EXECUTABLE}"
          --llvm-dir "${LLVM_DIR}"
          --clang-dir "${Clang_DIR}"
          --matcore-mlir-available "${MDSLC_MATCORE_MLIR_AVAILABLE}"
          --default-semantic-pipeline "${MDSLC_DEFAULT_SEMANTIC_PIPELINE}"
          --experimental-toolchain-version
            "${MDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION}"
          ${MDSLC_SOURCE_INACCESSIBLE_MLIR_ARGUMENTS}
          ${MDSLC_SOURCE_INACCESSIBLE_RUNTIME_ARGUMENTS}
      )
      set_tests_properties(
        package.installed_source_inaccessible
        PROPERTIES
          RUN_SERIAL TRUE
          TIMEOUT 600
      )
      add_test(
        NAME package.installed_source_inaccessible_safety
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/run_source_inaccessible_consumer_safety_tests.py"
          --script
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/run_source_inaccessible_consumer.py"
          --git "${MDSLC_PACKAGE_GIT_EXECUTABLE}"
      )
    endif()
  endif()
endif()
