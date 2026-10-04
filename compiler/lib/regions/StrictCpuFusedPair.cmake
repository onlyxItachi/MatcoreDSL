# Independent compiler-issued pair primitive. No Runtime, source or default
# driver binding is added by this definition. Qualification is Linux x64 only.
if(NOT MDSLC_CLOSED_CPU_X86)
  return()
endif()
add_library(matcore_cpu_fused_gemm_candidate STATIC ../mlir/MatcoreCpuFusedGemmCandidate.cpp)
target_compile_features(matcore_cpu_fused_gemm_candidate PUBLIC cxx_std_20)
target_include_directories(matcore_cpu_fused_gemm_candidate PUBLIC ../mlir)
target_include_directories(matcore_cpu_fused_gemm_candidate SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(matcore_cpu_fused_gemm_candidate PUBLIC matcore_closed_region_semantics
  PRIVATE matcore_mlir_bufferized_handoff MLIRLinalgTransforms MLIRArithToLLVM
  MLIRAffineToStandard MLIRSCFToControlFlow MLIRMemRefToLLVM MLIRFuncToLLVM
  MLIRControlFlowToLLVM MLIRReconcileUnrealizedCasts MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation MLIRTargetLLVMIRExport MLIRLinalgTransformOps
  MLIRTensorTransformOps MLIRSCFTransforms MLIRArithTransforms MLIRMemRefTransforms MLIRTransformDialect
  MLIRTransformDialectTransforms MLIRParser MLIRTransforms)
mdslc_target_enable_warnings(matcore_cpu_fused_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore_cpu_fused_gemm_candidate)
add_executable(matcore-cpu-fused-gemm-candidate ../../tools/matcore-cpu-fused-gemm-candidate/main.cpp)
target_link_libraries(matcore-cpu-fused-gemm-candidate PRIVATE matcore_cpu_fused_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore-cpu-fused-gemm-candidate)
set(fused_pair_modes normal)
if(BUILD_TESTING)
  list(APPEND fused_pair_modes asan)
endif()
foreach(mode IN LISTS fused_pair_modes)
  set(ir "${CMAKE_CURRENT_BINARY_DIR}/strict-fused-pair-${mode}.ll")
  set(object "${CMAKE_CURRENT_BINARY_DIR}/strict-fused-pair-${mode}.o")
  set(issuer_args)
  set(kernel_flags -O2 -march=x86-64 -ffp-contract=off)
  set(sanitized OFF)
  if(mode STREQUAL "asan" OR MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=.*address")
    list(APPEND issuer_args --asan)
    list(APPEND kernel_flags -O1 -g -fsanitize=address)
    set(sanitized ON)
  endif()
  add_custom_command(OUTPUT "${ir}"
    BYPRODUCTS "${ir}.manifest" "${ir}.semantic.mlir" "${ir}.structured.mlir"
      "${ir}.scheduled.mlir" "${ir}.bufferized.mlir" "${ir}.transform.mlir"
    COMMAND matcore-cpu-fused-gemm-candidate --output "${ir}" ${issuer_args}
    DEPENDS matcore-cpu-fused-gemm-candidate VERBATIM)
  add_custom_command(OUTPUT "${object}"
    COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -c -x ir "${ir}" ${kernel_flags} -o "${object}"
    COMMAND "${CMAKE_COMMAND}" "-DNM=${MDSLC_STRICT_ISA_NM}" "-DOBJDUMP=${MDSLC_STRICT_ISA_OBJDUMP}"
      "-DOBJECT=${object}" "-DSANITIZED=${sanitized}"
      -P "${CMAKE_CURRENT_LIST_DIR}/strict_fused_pair_object.cmake"
    DEPENDS "${ir}" "${CMAKE_CURRENT_LIST_DIR}/strict_fused_pair_object.cmake" VERBATIM)
  set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_custom_target(matcore_generated_fused_pair_${mode}_object DEPENDS "${object}")
  string(TOUPPER "${mode}" mode_upper)
  set(MDSLC_FUSED_PAIR_${mode_upper}_OBJECT "${object}")
  set(MDSLC_FUSED_PAIR_${mode_upper}_OBJECT "${object}" PARENT_SCOPE)
  set(MDSLC_FUSED_PAIR_${mode_upper}_SANITIZED "${sanitized}" PARENT_SCOPE)
endforeach()
