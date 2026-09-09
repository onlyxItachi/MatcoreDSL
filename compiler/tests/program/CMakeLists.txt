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

add_test(NAME driver.program_installed
  COMMAND "${CMAKE_COMMAND}" "-DBINARY_DIR=${CMAKE_BINARY_DIR}"
    "-DCOMPILER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../.."
    "-DINSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/installed.cmake")
set_tests_properties(driver.program_installed PROPERTIES TIMEOUT 180)

add_test(NAME driver.program_independent
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../program_independent/run.py"
    --driver "$<TARGET_FILE:mdslc-region>" --clang "${MDSLC_CLANGXX_EXECUTABLE}"
    --compiler-source "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set_tests_properties(driver.program_independent PROPERTIES TIMEOUT 300)
add_executable(matcore_program_call_interface_test ../program_independent/call_interface_test.cpp)
target_link_libraries(matcore_program_call_interface_test PRIVATE matcore_authenticated_host_thunk LLVM)
mdslc_target_enable_warnings(matcore_program_call_interface_test WERROR)
mdslc_target_match_llvm_rtti(matcore_program_call_interface_test)
add_test(NAME codegen.program_call_interface_v1
  COMMAND "${CMAKE_COMMAND}" "-DCLANG=${MDSLC_CLANGXX_EXECUTABLE}"
    "-DCOMPILER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/../.."
    "-DTEST=$<TARGET_FILE:matcore_program_call_interface_test>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/../program_independent/call_interface.cmake")
set_tests_properties(codegen.program_call_interface_v1 PROPERTIES TIMEOUT 180)
