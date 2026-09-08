# Tests consume the production compiler components from lib/regions.
if(NOT TARGET matcore_experimental_region_compiler)
  return()
endif()
add_executable(matcore_frozen_host_codegen_test frozen_host_test.cpp)
target_include_directories(matcore_frozen_host_codegen_test SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_libraries(matcore_frozen_host_codegen_test PRIVATE
  matcore_frozen_host_codegen MatcoreDSL::PlatformSupportV1)
mdslc_target_match_llvm_rtti(matcore_frozen_host_codegen_test)
execute_process(COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
  OUTPUT_VARIABLE closed_driver_resource_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
add_test(NAME frontend.frozen_host_codegen
  COMMAND matcore_frozen_host_codegen_test
  "${MDSLC_CLANGXX_EXECUTABLE}" "${closed_driver_resource_dir}")

add_executable(matcore_frozen_host_independent_test frozen_host_independent_test.cpp)
target_include_directories(matcore_frozen_host_independent_test SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_libraries(matcore_frozen_host_independent_test PRIVATE
  matcore_frozen_host_codegen MatcoreDSL::PlatformSupportV1)
mdslc_target_match_llvm_rtti(matcore_frozen_host_independent_test)
add_test(NAME frontend.frozen_host_independent
  COMMAND matcore_frozen_host_independent_test
  "${MDSLC_CLANGXX_EXECUTABLE}" "${closed_driver_resource_dir}")

add_executable(matcore_experimental_compile_test experimental_compile_test.cpp)
target_link_libraries(matcore_experimental_compile_test PRIVATE
  matcore_experimental_region_compiler MatcoreDSL::PlatformSupportV1)
mdslc_target_enable_warnings(matcore_experimental_compile_test WERROR)
mdslc_target_match_llvm_rtti(matcore_experimental_compile_test)
add_dependencies(matcore_experimental_compile_test matcore_closed_candidates_isolated_v1)
set(experimental_compiler_test_flags)
if(MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=.*address")
  list(APPEND experimental_compiler_test_flags --asan)
endif()
add_test(NAME frontend.experimental_source_generated_execution
  COMMAND matcore_experimental_compile_test
  "${MDSLC_CLANGXX_EXECUTABLE}" "${closed_driver_resource_dir}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  "$<TARGET_FILE:matcore_closed_candidates_isolated_v1>"
  "$<TARGET_FILE:matcore_runtime>" ${experimental_compiler_test_flags})
set_tests_properties(frontend.experimental_source_generated_execution PROPERTIES TIMEOUT 180)

if(TARGET mdslc-region)
  add_test(NAME driver.experimental_region_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DDRIVER=$<TARGET_FILE:mdslc-region>"
      "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../../examples/experimental/two_gemm.mdsl"
      "-DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
      "-DSANITIZED=${experimental_compiler_test_flags}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/driver_contract.cmake")
  set_tests_properties(driver.experimental_region_contract PROPERTIES TIMEOUT 240)
endif()
