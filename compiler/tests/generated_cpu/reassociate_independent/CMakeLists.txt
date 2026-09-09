add_executable(matcore_reassociate_independent_mutations mutations.cpp)
target_link_libraries(matcore_reassociate_independent_mutations PRIVATE
  matcore_cpu_gemm_candidate MLIRLinalgDialect MLIRVectorDialect LLVM)
mdslc_target_match_llvm_rtti(matcore_reassociate_independent_mutations)
add_test(NAME generated_cpu.reassociate_independent_mutations
  COMMAND matcore_reassociate_independent_mutations)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME generated_cpu.reassociate_independent_classifier
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/classifier_test.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../expect_reassociate.sh")
add_test(NAME generated_cpu.reassociate_connected_runner
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/runner_test.py")

if(TARGET mdslc-region)
  set(connected_flags)
  if(MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=.*address")
    list(APPEND connected_flags --sanitized)
  endif()
  if(MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS)
    get_target_property(connected_provider mdslc-region MDSLC_REGION_PROVIDER_PATH)
    list(APPEND connected_flags --provider "${connected_provider}")
  endif()
  add_test(NAME generated_cpu.reassociate_connected_independent
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/ctest_connected.py"
      --cmake "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
      --driver "$<TARGET_FILE:mdslc-region>"
      --cxx "${MDSLC_CLANGXX_EXECUTABLE}" ${connected_flags})
  set_tests_properties(generated_cpu.reassociate_connected_independent PROPERTIES
    TIMEOUT 600 SKIP_RETURN_CODE 77)
endif()
