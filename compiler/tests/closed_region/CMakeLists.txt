# Production targets are defined once in lib/regions.
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
