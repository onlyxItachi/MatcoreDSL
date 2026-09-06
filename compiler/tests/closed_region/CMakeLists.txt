# Deliberately test-only. These interfaces are neither public source syntax nor
# an installed compiler feature, and never link into the executable CPU route.
add_library(matcore_closed_region_semantics STATIC
  ../../lib/mlir/MatcoreClosedRegion.cpp)
target_compile_features(matcore_closed_region_semantics PUBLIC cxx_std_20)
target_include_directories(matcore_closed_region_semantics PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/mlir")
target_link_libraries(matcore_closed_region_semantics PUBLIC
  matcore_mlir_semantics PRIVATE MLIRParser MLIRArithDialect MLIRFuncDialect)
mdslc_target_enable_warnings(matcore_closed_region_semantics)
mdslc_target_match_llvm_rtti(matcore_closed_region_semantics)

add_library(matcore_closed_region_admission STATIC
  ../../lib/frontend/ClosedRegionAdmission.cpp
  ../../lib/frontend/ClosedRegionHostAdmission.cpp
  ../../lib/frontend/ClosedRegionHostInputs.cpp)
set(region_include "${CMAKE_CURRENT_SOURCE_DIR}/../../include/matcore")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${region_include}/region.h" "${region_include}/detail/region_storage.h")
file(READ "${region_include}/region.h" MDSLC_EXPERIMENTAL_REGION_SOURCE)
file(READ "${region_include}/detail/region_storage.h" MDSLC_EXPERIMENTAL_REGION_STORAGE_SOURCE)
configure_file(../../lib/frontend/ExperimentalRegionHeaders.inc.in
  "${CMAKE_CURRENT_BINARY_DIR}/ExperimentalRegionHeaders.inc" @ONLY)
target_compile_features(matcore_closed_region_admission PUBLIC cxx_std_20)
target_include_directories(matcore_closed_region_admission PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/frontend"
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(matcore_closed_region_admission SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_compile_definitions(matcore_closed_region_admission PRIVATE
  ${LLVM_DEFINITIONS})
execute_process(COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
  OUTPUT_VARIABLE MDSLC_CLOSED_REGION_HOST_RESOURCE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(MDSLC_CLOSED_REGION_HOST_RUNTIME "")
if(TARGET clang-cpp)
  set(MDSLC_CLOSED_REGION_HOST_RUNTIME "$<TARGET_FILE:clang-cpp>")
endif()
target_compile_definitions(matcore_closed_region_admission PRIVATE
  "MDSLC_HOST_CLANG_EXECUTABLE=\"${MDSLC_CLANGXX_EXECUTABLE}\""
  "MDSLC_HOST_CLANG_RESOURCE_DIRECTORY=\"${MDSLC_CLOSED_REGION_HOST_RESOURCE_DIR}\""
  "MDSLC_HOST_TOOLCHAIN_VERSION=\"${MDSLC_REQUIRED_TOOLCHAIN_VERSION}\""
  "MDSLC_HOST_CLANG_CPP_RUNTIME=\"${MDSLC_CLOSED_REGION_HOST_RUNTIME}\"")
target_link_libraries(matcore_closed_region_admission PUBLIC
  matcore_closed_region_semantics PRIVATE ${MDSLC_CLANG_TOOLING_LIBRARIES}
  matcore_frontend_runtime_identity MatcoreDSL::PlatformSupportV1)
mdslc_target_enable_warnings(matcore_closed_region_admission)
mdslc_target_match_llvm_rtti(matcore_closed_region_admission)

add_executable(matcore_closed_region_semantic_tests
  ../mlir/matcore_closed_region_test.cpp)
target_link_libraries(matcore_closed_region_semantic_tests PRIVATE
  matcore_closed_region_semantics MLIRParser)
mdslc_target_match_llvm_rtti(matcore_closed_region_semantic_tests)
add_test(NAME mlir.closed_region_semantics
  COMMAND matcore_closed_region_semantic_tests)

add_executable(matcore_closed_region_admission_tests admission_test.cpp)
target_link_libraries(matcore_closed_region_admission_tests PRIVATE
  matcore_closed_region_admission MLIRParser)
mdslc_target_enable_warnings(matcore_closed_region_admission_tests)
mdslc_target_match_llvm_rtti(matcore_closed_region_admission_tests)
add_test(NAME frontend.closed_region_admission
  COMMAND matcore_closed_region_admission_tests)
add_test(NAME frontend.closed_region_ordinary_rejected
  COMMAND "${CMAKE_COMMAND}"
    "-DCLANGXX=${MDSLC_CLANGXX_EXECUTABLE}"
    "-DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixture_language.h"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/ordinary_rejected.cmake")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET clang-cpp)
  add_executable(matcore_experimental_region_admission_tests ../experimental_region/admission_test.cpp)
  target_link_libraries(matcore_experimental_region_admission_tests PRIVATE
    matcore_closed_region_admission MatcoreDSL::PlatformSupportV1 MLIRParser)
  mdslc_target_enable_warnings(matcore_experimental_region_admission_tests)
  mdslc_target_match_llvm_rtti(matcore_experimental_region_admission_tests)
  add_test(NAME frontend.experimental_region_admission
    COMMAND matcore_experimental_region_admission_tests
      "${MDSLC_CLANGXX_EXECUTABLE}" "${MDSLC_CLOSED_REGION_HOST_RESOURCE_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
  add_executable(matcore_closed_region_host_tests host_admission_test.cpp)
  target_link_libraries(matcore_closed_region_host_tests PRIVATE
    matcore_closed_region_admission MatcoreDSL::PlatformSupportV1 MLIRParser)
  mdslc_target_enable_warnings(matcore_closed_region_host_tests)
  mdslc_target_match_llvm_rtti(matcore_closed_region_host_tests)
  add_test(NAME frontend.closed_region_host_context
    COMMAND matcore_closed_region_host_tests
      "${MDSLC_CLANGXX_EXECUTABLE}"
      "${MDSLC_CLOSED_REGION_HOST_RESOURCE_DIR}")
endif()
