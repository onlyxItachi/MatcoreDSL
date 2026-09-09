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
