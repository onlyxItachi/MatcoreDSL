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
  ../../lib/frontend/ClosedRegionAdmission.cpp)
target_compile_features(matcore_closed_region_admission PUBLIC cxx_std_20)
target_include_directories(matcore_closed_region_admission PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/frontend")
target_include_directories(matcore_closed_region_admission SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_compile_definitions(matcore_closed_region_admission PRIVATE
  ${LLVM_DEFINITIONS})
target_link_libraries(matcore_closed_region_admission PUBLIC
  matcore_closed_region_semantics PRIVATE ${MDSLC_CLANG_TOOLING_LIBRARIES})
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
