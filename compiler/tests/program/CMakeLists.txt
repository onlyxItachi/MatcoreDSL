find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(program_case core paths ownership)
  add_test(NAME driver.program_${program_case}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run.py"
      --driver "$<TARGET_FILE:mdslc-region>" --case "${program_case}")
  set_tests_properties(driver.program_${program_case} PROPERTIES TIMEOUT 240)
endforeach()
add_test(NAME driver.program_host_utilities
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/host_expressiveness.py"
    --driver "$<TARGET_FILE:mdslc-region>")
set_tests_properties(driver.program_host_utilities PROPERTIES TIMEOUT 180)
add_test(NAME driver.program_refusal_classifier
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/runner_test.py")

add_executable(matcore_program_input_mutation_test input_mutation_test.cpp)
target_link_libraries(matcore_program_input_mutation_test PRIVATE
  matcore_experimental_program_compiler MatcoreDSL::PlatformSupportV1)
mdslc_target_enable_warnings(matcore_program_input_mutation_test WERROR)
mdslc_target_match_llvm_rtti(matcore_program_input_mutation_test)
execute_process(COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
  OUTPUT_VARIABLE program_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(program_mutation_flags)
if(region_sanitized)
  list(APPEND program_mutation_flags --asan)
endif()
add_test(NAME frontend.program_input_mutation
  COMMAND matcore_program_input_mutation_test "${MDSLC_CLANGXX_EXECUTABLE}"
    "${program_resource_dir}" "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    "$<TARGET_FILE:matcore_closed_candidates_isolated_v1>" "$<TARGET_FILE:matcore_runtime>"
    ${program_mutation_flags})
set_tests_properties(frontend.program_input_mutation PROPERTIES TIMEOUT 240)
