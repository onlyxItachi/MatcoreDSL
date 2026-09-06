# Private built-in AOT leaf proof. No public CLI, headers, ABI or source authority.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT MDSLC_REQUIRED_TOOLCHAIN_VERSION STREQUAL "21.1.8" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  return()
endif()
add_library(matcore_cpu_gemm_candidate STATIC ../../lib/mlir/MatcoreCpuGemmCandidate.cpp)
target_include_directories(matcore_cpu_gemm_candidate PUBLIC ../../lib/mlir)
target_include_directories(matcore_cpu_gemm_candidate SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(matcore_cpu_gemm_candidate PUBLIC matcore_closed_region_semantics
  PRIVATE matcore_mlir_bufferized_handoff MLIRLinalgTransforms MLIRArithToLLVM
    MLIRAffineToStandard MLIRSCFToControlFlow MLIRMemRefToLLVM MLIRFuncToLLVM
    MLIRControlFlowToLLVM MLIRReconcileUnrealizedCasts MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation MLIRTargetLLVMIRExport)
mdslc_target_enable_warnings(matcore_cpu_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore_cpu_gemm_candidate)

add_executable(matcore-cpu-gemm-candidate ../../tools/matcore-cpu-gemm-candidate/main.cpp)
target_link_libraries(matcore-cpu-gemm-candidate PRIVATE matcore_cpu_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore-cpu-gemm-candidate)
add_executable(matcore_cpu_gemm_candidate_test candidate_test.cpp)
target_link_libraries(matcore_cpu_gemm_candidate_test PRIVATE matcore_cpu_gemm_candidate MLIRLinalgDialect)
mdslc_target_match_llvm_rtti(matcore_cpu_gemm_candidate_test)
add_test(NAME generated_cpu.strict_candidate COMMAND matcore_cpu_gemm_candidate_test)

foreach(mode IN ITEMS normal asan)
  set(issuer_args)
  set(kernel_flags -O2)
  if(mode STREQUAL "asan" OR CMAKE_CXX_FLAGS MATCHES "fsanitize=.*address")
    list(APPEND issuer_args --asan)
    list(APPEND kernel_flags -fsanitize=address)
  endif()
  set(ir "${CMAKE_CURRENT_BINARY_DIR}/strict-${mode}.ll")
  set(object "${CMAKE_CURRENT_BINARY_DIR}/strict-${mode}.o")
  add_custom_command(OUTPUT "${ir}"
    BYPRODUCTS "${ir}.manifest" "${ir}.semantic.mlir" "${ir}.structured.mlir" "${ir}.bufferized.mlir"
    COMMAND matcore-cpu-gemm-candidate --output "${ir}" ${issuer_args}
    DEPENDS matcore-cpu-gemm-candidate VERBATIM)
  add_custom_command(OUTPUT "${object}"
    COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -c -x ir "${ir}" ${kernel_flags} -o "${object}"
    DEPENDS "${ir}" VERBATIM)
  set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_executable(matcore_generated_cpu_${mode}_test execution_test.cpp "${object}")
  target_compile_features(matcore_generated_cpu_${mode}_test PRIVATE cxx_std_20)
  target_compile_options(matcore_generated_cpu_${mode}_test PRIVATE -ffp-contract=off)
  if(mode STREQUAL "asan")
    target_compile_options(matcore_generated_cpu_${mode}_test PRIVATE -fsanitize=address)
    target_link_options(matcore_generated_cpu_${mode}_test PRIVATE -fsanitize=address)
  endif()
  add_test(NAME generated_cpu.execution.${mode} COMMAND matcore_generated_cpu_${mode}_test)
  add_executable(matcore_generated_cpu_independent_${mode}_test independent_execution_test.cpp "${object}")
  target_compile_features(matcore_generated_cpu_independent_${mode}_test PRIVATE cxx_std_20)
  target_compile_options(matcore_generated_cpu_independent_${mode}_test PRIVATE -ffp-contract=off -frounding-math)
  if(mode STREQUAL "asan")
    target_compile_options(matcore_generated_cpu_independent_${mode}_test PRIVATE -fsanitize=address)
    target_link_options(matcore_generated_cpu_independent_${mode}_test PRIVATE -fsanitize=address)
  endif()
  add_test(NAME generated_cpu.independent_execution.${mode} COMMAND matcore_generated_cpu_independent_${mode}_test)
endforeach()
find_program(MDSLC_CANDIDATE_NM NAMES llvm-nm HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(MDSLC_CANDIDATE_OBJDUMP NAMES llvm-objdump HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
add_test(NAME generated_cpu.object_contract COMMAND "${CMAKE_COMMAND}"
  "-DNM=${MDSLC_CANDIDATE_NM}" "-DOBJDUMP=${MDSLC_CANDIDATE_OBJDUMP}"
  "-DOBJECT=${CMAKE_CURRENT_BINARY_DIR}/strict-normal.o"
  "-DSANITIZED=${CMAKE_CXX_FLAGS}" -P "${CMAKE_CURRENT_SOURCE_DIR}/object_contract.cmake")
add_test(NAME generated_cpu.sanitizer_negative_control COMMAND "${CMAKE_COMMAND}"
  "-DEXECUTABLE=$<TARGET_FILE:matcore_generated_cpu_asan_test>" -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_asan.cmake")
