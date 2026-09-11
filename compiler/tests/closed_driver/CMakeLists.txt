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
  foreach(library_case repeated_include escaped_address escaped_wrapper)
    add_test(NAME driver.source_library_${library_case}
      COMMAND "${CMAKE_COMMAND}"
        "-DDRIVER=$<TARGET_FILE:mdslc-region>"
        "-DCASE=${library_case}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/../experimental_region/library_adversary/run.cmake")
    set_tests_properties(driver.source_library_${library_case} PROPERTIES TIMEOUT 200)
  endforeach()
  foreach(library_case template_specialization template_macro_type template_macro_body
          template_hidden_effect template_escaped_specialization template_split_body)
    add_test(NAME driver.source_library_${library_case}
      COMMAND "${CMAKE_COMMAND}"
        "-DDRIVER=$<TARGET_FILE:mdslc-region>"
        "-DCASE=${library_case}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/../experimental_region/library_adversary/template_run.cmake")
    set_tests_properties(driver.source_library_${library_case} PROPERTIES TIMEOUT 200)
  endforeach()
  set(reassociate_policies generated-reassociate generated-strict native-strict automatic)
  if(MDSLC_CLOSED_CPU_X86)
    list(APPEND reassociate_policies existing-native)
  endif()
  if(MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS)
    list(APPEND reassociate_policies openblas)
  endif()
  foreach(policy IN LISTS reassociate_policies)
    add_test(NAME driver.generated_reassociate.${policy}
      COMMAND "${CMAKE_COMMAND}" "-DDRIVER=$<TARGET_FILE:mdslc-region>"
        "-DPOLICY=${policy}" -P "${CMAKE_CURRENT_SOURCE_DIR}/generated_reassociate.cmake")
    set_tests_properties(driver.generated_reassociate.${policy} PROPERTIES TIMEOUT 180)
  endforeach()
  get_target_property(driver_provider mdslc-region MDSLC_REGION_PROVIDER_PATH)
  add_test(NAME driver.experimental_region_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DDRIVER=$<TARGET_FILE:mdslc-region>"
      "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../../examples/experimental/two_gemm.mdsl"
      "-DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
      "-DSANITIZED=${experimental_compiler_test_flags}"
      "-DPROVIDER=${driver_provider}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/driver_contract.cmake")
  set_tests_properties(driver.experimental_region_contract PROPERTIES TIMEOUT 240)
  add_test(NAME driver.implementation_ownership
    COMMAND "${CMAKE_COMMAND}"
      "-DDRIVER=$<TARGET_FILE:mdslc-region>"
      "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/implementation_ownership.mdsl"
      "-DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/implementation_ownership.cmake")
  set_tests_properties(driver.implementation_ownership PROPERTIES TIMEOUT 240)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(storage_conformance_flags)
  if(experimental_compiler_test_flags)
    list(APPEND storage_conformance_flags --sanitized)
  endif()
  if(MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS)
    list(APPEND storage_conformance_flags --has-openblas)
  endif()
  add_test(NAME driver.storage_conformance
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../driver_storage_conformance/run.py"
      "$<TARGET_FILE:mdslc-region>"
      --clangxx "${MDSLC_CLANGXX_EXECUTABLE}"
      --library-directory "$<TARGET_FILE_DIR:matcore_runtime>"
      ${storage_conformance_flags})
  add_test(NAME driver.storage_conformance_runner
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../driver_storage_conformance/runner_test.py")
  set_tests_properties(driver.storage_conformance driver.storage_conformance_runner
    PROPERTIES TIMEOUT 240 ENVIRONMENT "TMPDIR=${CMAKE_CURRENT_BINARY_DIR}")
endif()
