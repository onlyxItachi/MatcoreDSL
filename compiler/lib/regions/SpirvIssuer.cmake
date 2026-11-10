# Compiler-private construction and inspection only. Nothing here is installed
# or linked into a source candidate, runtime adapter, or execution registry.
if(NOT MDSLC_ENABLE_EXPERIMENTAL_SPIRV_ISSUER)
  return()
endif()
add_library(matcore_spirv_gemm_candidate STATIC ../mlir/MatcoreSpirvGemmCandidate.cpp)
target_compile_features(matcore_spirv_gemm_candidate PUBLIC cxx_std_20)
target_include_directories(matcore_spirv_gemm_candidate PUBLIC ../mlir)
target_include_directories(matcore_spirv_gemm_candidate SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(matcore_spirv_gemm_candidate PUBLIC matcore_cpu_gemm_candidate
  PRIVATE MLIRGPUTransforms MLIRSCFToGPU MLIRGPUToSPIRV MLIRSPIRVDialect
    MLIRSPIRVTransforms MLIRSPIRVSerialization MLIRSPIRVDeserialization MLIRTransforms LLVM)
mdslc_target_enable_warnings(matcore_spirv_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore_spirv_gemm_candidate)
add_executable(matcore-spirv-gemm-candidate
  ../../tools/matcore-spirv-gemm-candidate/main.cpp)
target_link_libraries(matcore-spirv-gemm-candidate PRIVATE matcore_spirv_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore-spirv-gemm-candidate)
