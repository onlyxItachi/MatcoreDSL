if(NOT TARGET matcore-cpu-fused-gemm-candidate)
  return()
endif()
add_executable(matcore_cpu_fused_gemm_candidate_test candidate_test.cpp)
target_link_libraries(matcore_cpu_fused_gemm_candidate_test PRIVATE matcore_cpu_fused_gemm_candidate matcore_cpu_gemm_candidate
  MLIRLinalgDialect MLIRSCFDialect MLIRTensorDialect MLIRMemRefDialect MLIRAffineDialect LLVM)
mdslc_target_match_llvm_rtti(matcore_cpu_fused_gemm_candidate_test)
add_test(NAME generated_cpu_fused_pair.candidate COMMAND matcore_cpu_fused_gemm_candidate_test)
add_test(NAME generated_cpu_fused_pair.cli COMMAND "${CMAKE_COMMAND}"
  "-DISSUER=$<TARGET_FILE:matcore-cpu-fused-gemm-candidate>" -P "${CMAKE_CURRENT_SOURCE_DIR}/cli_contract.cmake")
foreach(mode IN ITEMS normal asan)
  string(TOUPPER "${mode}" mode_upper)
  set(object "${MDSLC_FUSED_PAIR_${mode_upper}_OBJECT}")
  set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_executable(matcore_generated_fused_pair_${mode}_test execution_test.cpp "${object}")
  target_compile_features(matcore_generated_fused_pair_${mode}_test PRIVATE cxx_std_20)
  target_compile_options(matcore_generated_fused_pair_${mode}_test PRIVATE -ffp-contract=off -fno-fast-math -frounding-math)
  if(mode STREQUAL "asan")
    target_compile_options(matcore_generated_fused_pair_${mode}_test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(matcore_generated_fused_pair_${mode}_test PRIVATE -fsanitize=address,undefined)
  endif()
  add_dependencies(matcore_generated_fused_pair_${mode}_test matcore_generated_fused_pair_${mode}_object)
  add_test(NAME generated_cpu_fused_pair.execution.${mode} COMMAND matcore_generated_fused_pair_${mode}_test)
  add_test(NAME generated_cpu_fused_pair.object.${mode} COMMAND "${CMAKE_COMMAND}"
    "-DOBJECT=${object}" "-DNM=${CMAKE_NM}" "-DOBJDUMP=${CMAKE_OBJDUMP}" "-DMODE=${mode}"
    "-DSANITIZED=${MDSLC_FUSED_PAIR_${mode_upper}_SANITIZED}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/object_contract.cmake")
endforeach()
foreach(input IN ITEMS a b d)
  add_test(NAME generated_cpu_fused_pair.sanitizer_read_${input} COMMAND "${CMAKE_COMMAND}"
    "-DEXECUTABLE=$<TARGET_FILE:matcore_generated_fused_pair_asan_test>" "-DINPUT=${input}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_asan.cmake")
endforeach()
